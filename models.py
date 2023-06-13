"""Code for pixelnerf and alternatives."""
import torch, torchvision
from torch import nn
from einops import rearrange, repeat
from torch.nn import functional as F
import numpy as np
import time
import timm
from matplotlib import cm
import kornia
from tqdm import tqdm

import conv_modules
import mlp_modules
import geometry
import renderer
from geometry import procrustes

ch_sec = lambda x: rearrange(x,"... c x y -> ... (x y) c")
ch_fst = lambda src,x=None:rearrange(src,"... (x y) c -> ... c x y",x=int(src.size(-2)**(.5)) if x is None else x)
hom    = lambda x: torch.cat((x,torch.ones_like(x[...,[0]])),-1)

class FlowCam(nn.Module):
    def __init__(self, near=1.75, far=8, n_samples=64,num_view=2,logspace=True,use_trgt_crop=False,use_midas=False):
        super().__init__()

        self.raft_midas_net = RaftAndMidas(raft=True,midas=use_midas)

        self.near,self.far,self.n_samples = near,far,n_samples
        self.logspace=logspace
        self.use_trgt_crop=use_trgt_crop

        self.num_view=num_view

        self.nerf_enc_flow = conv_modules.PixelNeRFEncoder(in_ch=5,use_first_pool=False)
        #self.nerf_enc_flow = conv_modules.PixelNeRFEncoder(in_ch=3,use_first_pool=False)
        phi_latent=64
        self.renderer = MLPImplicit(latent_in=phi_latent,inner_latent=phi_latent,white_back=False,num_first_res_block=2)

        self.pos_encoder = PositionalEncodingNoFreqFactor(3,5) 

        self.ray_comb = nn.Sequential(torch.nn.Conv2d(512+33,512,3,padding=1),nn.ReLU(),
                                      torch.nn.Conv2d(512,512,3,padding=1),nn.ReLU(),
                                      torch.nn.Conv2d(512,phi_latent-3,3,padding=1)
                                     )
        self.corr_weighter_perpoint = nn.Sequential(
                nn.Linear(128,128),nn.ReLU(),
                nn.Linear(128,128),nn.ReLU(),
                nn.Linear(128,128),nn.ReLU(),
                nn.Linear(128,1),
                )
        self.corr_weighter_perpoint.apply(mlp_modules.init_weights_normal)

    def encoder(self,model_input):
        imsize=model_input["ctxt_rgb"].shape[-2:]

        if "backbone_feats" in model_input: return model_input["backbone_feats"]
        if "bwd_flow" not in model_input: model_input = self.raft_midas_net(model_input)

        # ctxt[1:]==trgt[:-1], using this property to avoid redundant computation
        all_rgb = torch.cat((model_input["ctxt_rgb"][:,:1],model_input["trgt_rgb"]),1)
        all_flow = torch.cat((model_input["bwd_flow"][:,:1],model_input["bwd_flow"]),1)

        # Resnet rgb+flow feats
        rgb_flow = torch.cat((all_rgb,all_flow*4),2)
        rgb_flow_feats = self.nerf_enc_flow(rgb_flow,imsize)
        #rgb_flow_feats = self.nerf_enc_flow(all_rgb,imsize)

        # Add rays to features for some amount of focal length information
        rds = self.pos_encoder(geometry.get_world_rays(model_input["x_pix"], model_input["intrinsics"], None)[1])
        rds = ch_fst(torch.cat((rds[:,:1],rds),1),imsize[0])

        all_feats = self.ray_comb(torch.cat((rgb_flow_feats,rds),2).flatten(0,1)).unflatten(0,all_rgb.shape[:2])
        all_feats = torch.cat((all_feats,all_rgb),2)
        model_input["backbone_feats"] = all_feats

        return all_feats

    def forward(self, model_input, trgt_rays=None,ctxt_rays=None,poses=None):

        imsize=model_input["ctxt_rgb"].shape[-2:]
        (b,n_ctxt),n_trgt=model_input["ctxt_rgb"].shape[:2],model_input["trgt_rgb"].size(1)
        add_ctxt = lambda x: torch.cat((x[:,:1],x),1)
        if trgt_rays is None: trgt_rays,ctxt_rays = self.make_rays(model_input)

        # Encode images
        backbone_feats= self.encoder(model_input) 

        # Expand identity camera into 3d points and render rays
        ros, rds = geometry.get_world_rays(add_ctxt(model_input["x_pix"]), add_ctxt(model_input["intrinsics"]), None)
        eye_pts, z_vals = renderer.sample_points_along_rays(self.near, self.far, self.n_samples, ros, rds, device=model_input["x_pix"].device,logspace=self.logspace)
        eye_render, eye_depth, eye_weights= self.renderer( backbone_feats, eye_pts[:,:,ctxt_rays], add_ctxt(model_input["intrinsics"]), z_vals,identity=True)

        # Render out correspondence's surface point now
        corresp_uv = (model_input["x_pix"]+ch_sec(model_input["bwd_flow"]))[:,:,ctxt_rays]
        ros, rds = geometry.get_world_rays(corresp_uv, model_input["intrinsics"], None)
        corresp_pts, _ = renderer.sample_points_along_rays(self.near, self.far, self.n_samples, ros, rds, device=model_input["x_pix"].device,logspace=self.logspace)
        _, _, corresp_weights= self.renderer( backbone_feats[:,:-1], corresp_pts, model_input["intrinsics"], z_vals,identity=True)

        # Predict correspondence weights as function of source feature and correspondence
        corresp_feat = F.grid_sample(backbone_feats[:,:-1].flatten(0,1),corresp_uv.flatten(0,1).unsqueeze(1)*2-1).squeeze(-2).permute(0,2,1).unflatten(0,(b,n_trgt))
        corr_weights = self.corr_weighter_perpoint(torch.cat((corresp_feat,ch_sec(backbone_feats)[:,1:,ctxt_rays]),-1)).sigmoid()

        # Weighted procrustes on scene flow
        if poses is None:
            adj_transf = procrustes((eye_pts[:,1:,ctxt_rays]*eye_weights[:,1:]).sum(-2), (corresp_weights*corresp_pts).sum(-2), corr_weights)[1] 
            poses = adj_transf
            for i in range(n_trgt-1,0,-1):
                poses = torch.cat((poses[:,:i],poses[:,[i-1]]@poses[:,i:]),1)
        else: adj_transf = torch.cat((poses[:,:1],poses[:,:-1].inverse()@poses[:,1:]),1)

        # Render out trgt frames from [ctxt=0, ctxt=-1, ctxt=middle][: num context frames ] 
        render = self.render(model_input,poses,trgt_rays)

        # Pose induced flow using ctxt depth and then multiview rendered depth
        corresp_surf_from_pose = (torch.einsum("bcij,bcdkj->bcdki",adj_transf,hom(eye_pts[:,1:]))[:,:,ctxt_rays,...,:3]*eye_weights[:,1:]).sum(-2)
        flow_from_pose = geometry.project(corresp_surf_from_pose.flatten(0,1), model_input["intrinsics"].flatten(0,1))[0].unflatten(0,(b,n_trgt))-model_input["x_pix"][:,:,ctxt_rays]
        corresp_surf_from_pose_render = (torch.einsum("bcij,bcdkj->bcdki",adj_transf,hom(eye_pts[:,1:]))[:,:,trgt_rays,...,:3]*render["weights"]).sum(-2)
        flow_from_pose_render = geometry.project(corresp_surf_from_pose_render.flatten(0,1), model_input["intrinsics"].flatten(0,1))[0].unflatten(0,(b,n_trgt))-model_input["x_pix"][:,:,trgt_rays]

        out= {
                "rgb":render["rgb"], 
                "ctxt_rgb":eye_render[:,:-1],
                "poses":poses,
                "depth":render["depth"], 
                "ctxt_depth":eye_depth[:,:-1], 
                "corr_weights": corr_weights,
                "flow_from_pose": flow_from_pose,
                "flow_from_pose_render": flow_from_pose_render,
                "ctxt_rays": ctxt_rays.to(eye_render)[None].expand(b,-1), 
                "trgt_rays": trgt_rays.to(eye_render)[None].expand(b,-1),
                "flow_inp": model_input["bwd_flow"],
        }
        if "ctxt_depth" in model_input:
            out["ctxt_depth_inp"]=model_input["ctxt_depth"]
            out["trgt_depth_inp"]=model_input["trgt_depth"]
        return out

    def render(self,model_input,poses,trgt_rays,query_pose=None):
        if query_pose is None: query_pose=poses

        ros, rds = geometry.get_world_rays(model_input["x_pix"][:,:query_pose.size(1),trgt_rays], model_input["intrinsics"][:,:query_pose.size(1)], query_pose)
        query_pts, z_vals = renderer.sample_points_along_rays(self.near, self.far, self.n_samples, ros, rds, device=model_input["x_pix"].device,logspace=self.logspace)

        ctxt_poses = torch.cat((torch.eye(4).cuda()[None].expand(poses.size(0),-1,-1)[:,None],poses),1)
        ctxt_idxs = [0,-1,model_input["trgt_rgb"].size(1)//2][:self.num_view]
        ctxt_pts = torch.einsum("bvcij,bcdkj->bvcdki",ctxt_poses[:,ctxt_idxs].inverse().unsqueeze(2),hom(query_pts))[...,:3] # in coord system of ctxt frames
        rgb, depth, weights = self.renderer(model_input["backbone_feats"][:,ctxt_idxs], ctxt_pts,model_input["intrinsics"][:,:query_pose.size(1)],z_vals)
        return {"rgb":rgb,"depth":depth,"weights":weights}

    def make_rays(self,model_input):

        imsize=model_input["ctxt_rgb"].shape[-2:]

        # Pick random subset of rays
        crop_res=32 if self.n_samples<100 else 16
        if self.use_trgt_crop: # choose random crop of rays instead of random set
            start_x,start_y = np.random.randint(0,imsize[1]-crop_res-1),np.random.randint(0,imsize[0]-crop_res-1)
            trgt_rays = torch.arange(imsize[0]*imsize[1]).view(imsize[0],imsize[1])[start_y:start_y+crop_res,start_x:start_x+crop_res].flatten()
        else:
            trgt_rays = torch.randperm(imsize[0]*imsize[1]-2)[:crop_res**2]
            
        ctxt_rays = torch.randperm(imsize[0]*imsize[1]-2)[:(32 if torch.is_grad_enabled() else 48)**2]
        return trgt_rays,ctxt_rays

    def render_full_img(self,model_input, query_pose=None,sample_out=None):
        num_chunk=8 if self.n_samples<90 else 16

        imsize=model_input["ctxt_rgb"].shape[-2:]
        (b,n_ctxt),n_trgt=model_input["ctxt_rgb"].shape[:2],model_input["trgt_rgb"].size(1)

        if sample_out is None: sample_out = self(model_input)

        # Render out image iteratively and aggregate outputs
        outs=[]
        for j,trgt_rays in enumerate(torch.arange(imsize[0]*imsize[1]).chunk(num_chunk)):
            with torch.no_grad():
                outs.append( self(model_input,trgt_rays=trgt_rays,ctxt_rays=trgt_rays,poses=sample_out["poses"]) if query_pose is None else 
                    self.render(model_input,sample_out["poses"],trgt_rays,query_pose[:,None]))

        out_all = {}
        for k,v in outs[0].items():
            if len(v.shape)>3 and "inp" not in k and "poses" not in k: out_all[k]=torch.cat([out[k] for out in outs],2)
            else:out_all[k]=v
        out_all["depth_raw"] = out_all["depth"]
        for k,v in out_all.items():
            if "depth" in k: out_all[k] = torch.from_numpy(cm.get_cmap('magma')(v.min().item()/v.cpu().numpy())).squeeze(-2)[...,:3]

        return out_all

# Implicit function which performs pixel-aligned nerf
class MLPImplicit(nn.Module):

    def __init__(self,latent_in=512,inner_latent=128,white_back=True,num_first_res_block=2,add_view_dir=False): 
        super().__init__()
        self.white_back=white_back

        self.pos_encoder = PositionalEncodingNoFreqFactor(3,5)

        self.phi1= mlp_modules.ResFAMLP(3,latent_in,inner_latent,inner_latent,num_first_res_block,last_res=True) # 2 res blocks, outputs deep feature to be averaged over ctxts
        self.phi1.apply(mlp_modules.init_weights_normal)
        self.phi2= mlp_modules.ResMLP(inner_latent,inner_latent,4)
        self.phi2.apply(mlp_modules.init_weights_normal)

    # Note this method does not use cam2world and assumes `pos` is transformed to each context view's coordinate system and duplicated for each context view
    def forward(self,ctxt_feats,pos,intrinsics,samp_dists,flow_feats=None,identity=False):

        if identity: # 1:1 correspondence of ctxt img to trgt render, used for rendering out identity camera 
            b_org,n_trgt_org = ctxt_feats.shape[:2]
            ctxt_feats,pos,intrinsics = [x.flatten(0,1)[:,None] for x in [ctxt_feats,pos,intrinsics]]

        if len(pos.shape)==5: pos=pos.unsqueeze(1) # unsqueeze ctxt if no ctxt dim (means only 1 ctxt supplied)
        (b,n_ctxt),n_trgt=(ctxt_feats.shape[:2],pos.size(2))

        # Projection onto identity camera
        img_feat, pos_,_ = geometry.pixel_aligned_features( 
                repeat(pos,"b ctxt trgt xy d c -> (b ctxt trgt) (xy d) c"),
                repeat(torch.eye(4)[None,None].to(ctxt_feats),"1 1 x y -> (b ctxt trgt) x y",b=b,ctxt=n_ctxt,trgt=n_trgt),
                repeat(intrinsics,"b trgt x y -> (b ctxt trgt) x y",ctxt=n_ctxt),
                repeat(ctxt_feats,"b ctxt c x y -> (b ctxt trgt) c x y",trgt=n_trgt),
                )
        img_feat,pos= [rearrange(x,"(b ctxt trgt) (xy d) c -> (b trgt) xy d ctxt c",b=b,ctxt=n_ctxt,d=samp_dists.size(-1)) for x in [img_feat,pos_]]

        # Map (3d crd,projecting img feature) to (rgb,sigma)
        out = self.phi2(self.phi1(pos,img_feat).mean(dim=-2))
        rgb,sigma= out[...,:3].sigmoid(),out[...,3:4].relu()

        # Alphacomposite
        rgb, depth, weights = renderer.volume_integral(z_vals=samp_dists, sigmas=sigma, radiances=rgb, white_back=self.white_back)

        out = [x.unflatten(0,(b,n_trgt)) for x in [rgb,depth, weights]]
        return [x.squeeze(1).unflatten(0,(b_org,n_trgt_org)) for x in out] if identity else out 

# Adds RAFT flow and Midas depth to the model input
class RaftAndMidas(nn.Module):
    def __init__(self,raft=True,midas=True):
        super().__init__()
        self.run_raft,self.run_midas=raft,midas

        if self.run_raft:
            from torchvision.models.optical_flow import Raft_Small_Weights
            from torchvision.models.optical_flow import raft_small
            print("making raft")
            self.raft_transforms = Raft_Small_Weights.DEFAULT.transforms()
            self.raft = raft_small(weights=Raft_Small_Weights.DEFAULT, progress=False)
            print("done making raft")
        if self.run_midas:
            print("making midas")
            self.midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms").dpt_transform
            self.midas_large = torch.hub.load("intel-isl/MiDaS", "DPT_Hybrid")
            print("done making midas")
        else: print("no midas")

    def forward(self,model_input):

        # Estimate raft flow and midas depth if no flow/depth in dataset
        imsize=model_input["ctxt_rgb"].shape[-2:]
        (b,n_ctxt),n_trgt=model_input["ctxt_rgb"].shape[:2],model_input["trgt_rgb"].size(1)
        # Inputs should be in range [0,255]; TODO change to [-1,1] to stay consistent with other RGB range 

        with torch.no_grad():
            # Compute RAFT flow from each frame to next frame forward and backward - note image shape must be > 128 for raft to work
            if "bwd_flow" not in model_input and self.run_raft:
                raft = lambda x,y: F.interpolate(self.raft(x,y,num_flow_updates=12)[-1]/(torch.tensor(x.shape[-2:][::-1])-1).to(x)[None,:,None,None],imsize)
                #raft_rgbs = torch.cat([self.midas_transforms(raft_rgb.permute(1,2,0).cpu().numpy()) for raft_rgb in raft_rgbs]).cuda()
                raft_inputs = self.raft_transforms(model_input["trgt_rgb_large"].flatten(0,1).to(torch.uint8),model_input["ctxt_rgb_large"].flatten(0,1).to(torch.uint8))
                model_input["bwd_flow"] = raft(*raft_inputs).unflatten(0,(b,n_trgt))

            # Compute midas depth if not on a datastet with depth
            if "trgt_depth" not in model_input and self.run_midas:
                # Compute midas depth for sup.
                # TODO normalize this input correctly based on torch transform. I think it's just imagenet + [0,1] mapping but check 
                midas = lambda x: F.interpolate(1/(1e-3+self.midas_large(x)).unsqueeze(1),imsize)
                midas_rgbs = torch.cat((model_input["ctxt_rgb_med"],model_input["trgt_rgb_med"][:,-1:]),1).flatten(0,1)
                midas_rgbs = (midas_rgbs/255)*2-1
                all_depth = midas(midas_rgbs).unflatten(0,(b,n_trgt+1))
                model_input["trgt_depth"] = all_depth[:,1:]
                model_input["ctxt_depth"] = all_depth[:,:-1]
        return model_input

class PositionalEncodingNoFreqFactor(nn.Module):
    """PositionalEncoding module

    Maps v to positional encoding representation phi(v)

    Arguments:
        i_dim (int): input dimension for v
        N_freqs (int): #frequency to sample (default: 10)
    """
    def __init__(
            self,
            i_dim: int,
            N_freqs: int = 10,
    ) -> None:

        super().__init__()
        
        self.i_dim = i_dim
        self.out_dim = i_dim + (2 * N_freqs) * i_dim
        self.N_freqs = N_freqs

        a, b = 1, self.N_freqs - 1
        self.freq_bands = 2 ** torch.linspace(a, b, self.N_freqs)

    def forward(self, v):
        pe = [v]
        for freq in self.freq_bands:
            fv = freq * v
            pe += [torch.sin(fv), torch.cos(fv)]
        return torch.cat(pe, dim=-1)

