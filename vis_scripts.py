import os
import geometry
import wandb
from matplotlib import cm
from torchvision.utils import make_grid
import torch.nn.functional as F
import numpy as np
import torch
import flow_vis
import flow_vis_torch
import matplotlib.pyplot as plt; imsave = lambda x,y=0: plt.imsave("/nobackup/users/camsmith/img/tmp%s.png"%y,x.cpu().numpy()); 
from einops import rearrange, repeat
import piqa
import imageio
import splines.quaternion
from torchcubicspline import (natural_cubic_spline_coeffs, NaturalCubicSpline)

def write_video(save_dir,frames,vid_name,step,write_frames=False):
    frames = [(255*x).astype(np.uint8) for x in frames]
    if "time" in vid_name: frames = frames + frames[::-1]
    f = os.path.join(save_dir, f'{vid_name}.mp4')
    imageio.mimwrite(f, frames, fps=8, quality=7)
    wandb.log({f'vid/{vid_name}':wandb.Video(f, format='mp4', fps=8)})
    print("writing video at",f)
    if write_frames:
        for i,img in enumerate(frames):
            try: os.mkdir(os.path.join(save_dir, f'{vid_name}_{step}'))
            except:pass
            f=os.path.join(save_dir, f'{vid_name}/{i}.png');plt.imsave(f,img);print(f)

def normalize(a):
    return (a - a.min()) / (a.max() - a.min())

def cvt(a):
    a = a.permute(1, 2, 0).detach().cpu()
    a = (a - a.min()) / (a.max() - a.min())
    a = a.numpy()
    return a

ch_fst = lambda src,x=None:rearrange(src,"... (x y) c -> ... c x y",x=int(src.size(-2)**(.5)) if x is None else x)

# Renders out query frame with interpolated motion field
def render_time_interp(model_input,model,resolution,n):

    b=model_input["ctxt_rgb"].size(0)
    resolution = list(model_input["ctxt_rgb"].flatten(0,1).permute(0,2,3,1).shape)
    frames=[]
    thetas = np.linspace(0, 1, n)
    with torch.no_grad(): sample_out = model(model_input)
    if "flow_inp" in sample_out:model_input["bwd_flow"]=sample_out["flow_inp"]

    # TODO add wobble flag back in here from satori code

    all_poses=sample_out["poses"]
    pos_spline_idxs=torch.linspace(0,all_poses.size(1)-1,all_poses.size(1)) # no compression
    rot_spline_idxs=torch.linspace(0,all_poses.size(1)-1,all_poses.size(1)) # no compression
    all_pos_spline=[]
    all_quat_spline=[]
    for b_i in range(b):
        all_pos_spline.append(NaturalCubicSpline(natural_cubic_spline_coeffs(pos_spline_idxs, all_poses[b_i,pos_spline_idxs.long(),:3,-1].cpu())))
        quats=geometry.matrix_to_quaternion(all_poses[b_i,:,:3,:3])
        all_quat_spline.append(splines.quaternion.PiecewiseSlerp([splines.quaternion.UnitQuaternion.from_unit_xyzw(quat_) 
                                        for quat_ in quats[rot_spline_idxs.long()].cpu().numpy()],grid=rot_spline_idxs.tolist()))

    for t in torch.linspace(0,all_poses.size(1)-1,n):
        print(t)

        custom_poses=[]
        for b_i,(pos_spline,quat_spline_) in enumerate(zip(all_pos_spline,all_quat_spline)):
            custom_pose=torch.eye(4).cuda()
            custom_pose[:3,-1]=pos_spline.evaluate(t)
            closest_t = (custom_pose[:3,-1]-all_poses[b_i,:,:3,-1]).square().sum(-1).argmin()
            quat_eval=quat_spline_.evaluate(t.item())
            curr_quats = torch.tensor(list(quat_eval.vector)+[quat_eval.scalar])
            custom_pose[:3,:3] = geometry.quaternion_to_matrix(curr_quats)
            custom_poses.append(custom_pose)
        custom_pose=torch.stack(custom_poses)
        with torch.no_grad(): model_out = model.render_full_img(model_input,query_pose=custom_pose,sample_out=sample_out)

        rgb_pred = model_out["rgb"]
        resolution = list(model_input["ctxt_rgb"][:,:1].flatten(0,1).permute(0,2,3,1).shape)
        rgb_pred=rgb_pred[:,:1].view(resolution).permute(1,0,2,3).flatten(1,2).cpu().numpy()
        magma_depth=model_out["depth"][:,:1].view(resolution).permute(1,0,2,3).flatten(1,2).cpu()
        rgbd_im=torch.cat((torch.from_numpy(rgb_pred),magma_depth),0).numpy()
        frames.append(rgbd_im)
    return frames


    for i in range(n):
        print(i,n)
        query_pose = geometry.time_interp_poses(sample_out["poses"],i/(n-1), model_input["trgt_rgb"].size(1),None)[:,0]
        # todo fix this interpolation -- is it incorrect to interpolate here? 
        with torch.no_grad(): model_out = model.render_full_img(model_input,query_pose=query_pose,sample_out=sample_out)
        rgb_pred = model_out["rgb"]
        resolution = list(model_input["ctxt_rgb"][:,:1].flatten(0,1).permute(0,2,3,1).shape)
        rgb_pred=rgb_pred[:,:1].view(resolution).permute(1,0,2,3).flatten(1,2).cpu().numpy()
        magma_depth=model_out["depth"][:,:1].view(resolution).permute(1,0,2,3).flatten(1,2).cpu()
        rgbd_im=torch.cat((torch.from_numpy(rgb_pred),magma_depth),0).numpy()
        frames.append(rgbd_im)
    return frames

def look_at(eye, at=torch.Tensor([0, 0, 0]).cuda(), up=torch.Tensor([0, 1, 0]).cuda(), eps=1e-5):
    #at = at.unsqueeze(0).unsqueeze(0)
    #up = up.unsqueeze(0).unsqueeze(0)

    z_axis = eye - at
    #z_axis /= z_axis.norm(dim=-1, keepdim=True) + eps
    z_axis = z_axis/(z_axis.norm(dim=-1, keepdim=True) + eps)

    up = up.expand(z_axis.shape)
    x_axis = torch.cross(up, z_axis)
    #x_axis /= x_axis.norm(dim=-1, keepdim=True) + eps
    x_axis = x_axis/(x_axis.norm(dim=-1, keepdim=True) + eps)

    y_axis = torch.cross(z_axis, x_axis)
    #y_axis /= y_axis.norm(dim=-1, keepdim=True) + eps
    y_axis = y_axis/(y_axis.norm(dim=-1, keepdim=True) + eps)

    r_mat = torch.stack((x_axis, y_axis, z_axis), axis=-1)
    return r_mat

def render_cam_traj_wobble(model_input,model,resolution,n):

    c2w = torch.eye(4, device='cuda')[None]
    tmp = torch.eye(4).cuda()
    circ_scale = .1
    thetas = np.linspace(0, 2 * np.pi, n)
    frames = []

    if "ctxt_c2w" not in model_input:
        model_input["ctxt_c2w"] = torch.tensor([[-2.5882e-01, -4.8296e-01,  8.3652e-01, -2.2075e+00],
                                            [ 2.1187e-08, -8.6603e-01, -5.0000e-01,  2.3660e+00],
                                            [-9.6593e-01,  1.2941e-01, -2.2414e-01,  5.9150e-01],
                                            [ 0.0000e+00,  0.0000e+00,  0.0000e+00,  1.0000e+00]]
                                         )[None,None].expand(model_input["trgt_rgb"].size(0),model_input["trgt_rgb"].size(1),-1,-1).cuda()

    resolution = list(model_input["ctxt_rgb"].flatten(0,1).permute(0,2,3,1).shape)

    c2w = model_input["ctxt_c2w"]
    circ_scale = c2w[0,0,[0,2],-1].norm()
    #circ_scale = c2w[0,[0,1],-1].norm()
    thetas=np.linspace(0,np.pi*2,n)
    rgb_imgs=[]
    depth_imgs=[]
    start_theta=0#(model_input["ctxt_c2w"][0,0,0,-1]/circ_scale).arccos()
    with torch.no_grad(): sample_out = model(model_input)
    if "flow_inp" in sample_out: model_input["bwd_flow"]=sample_out["flow_inp"]
    step=2 if n==8 else 4
    zs = torch.cat((torch.linspace(0,-n//4,n//step),torch.linspace(-n//4,0,n//step),torch.linspace(0,n//4,n//step),torch.linspace(n//4,0,n//step)))
    for i in range(n):
        print(i,n)
        theta=float(thetas[i] + start_theta)
        x=np.cos(theta) * circ_scale * .075
        y=np.sin(theta) * circ_scale * .075
        tmp=torch.eye(4).cuda()
        newpos=torch.tensor([x,y,zs[i]*1e-1]).cuda().float()
        tmp[:3,-1] = newpos
        custom_c2w = tmp[None].expand(c2w.size(0),c2w.size(1),-1,-1)
        with torch.no_grad(): model_out = model(model_input,custom_transf=custom_c2w,full_img=True)

        resolution = [model_input["trgt_rgb"].size(0)]+list(resolution[1:])
        b = model_out["rgb"].size(0)
        rgb_pred = model_out["rgb"][:,0].view(resolution).permute(1,0,2,3).flatten(1,2).cpu().numpy()
        magma_depth = model_out["depth"][:,0].view(resolution).permute(1,0,2,3).flatten(1,2).cpu()
        #depth_pred = model_out["depth"][:,0].clone()
        #mind,maxd=sample_out["depth"].cpu().min(),sample_out["depth"].cpu().max()
        #depth_pred[0,0]=mind #normalize
        #depth_pred[0,1]=maxd #normalize
        #depth_pred = (mind/(1e-3+depth_pred).view(resolution[:-1]).permute(1,0,2).flatten(1,2).cpu().numpy())
        #magma = cm.get_cmap('magma')
        #magma_depth = torch.from_numpy(magma(depth_pred))[...,:3]
        rgbd_im=torch.cat((torch.from_numpy(rgb_pred),magma_depth),0).numpy()
        frames.append(rgbd_im)
    return frames



def render_cam_traj_time_wobble(model_input,model,resolution,n):

    c2w = torch.eye(4, device='cuda')[None]
    if "ctxt_c2w" not in model_input:
        model_input["ctxt_c2w"] = torch.tensor([[-2.5882e-01, -4.8296e-01,  8.3652e-01, -2.2075e+00],
                                            [ 2.1187e-08, -8.6603e-01, -5.0000e-01,  2.3660e+00],
                                            [-9.6593e-01,  1.2941e-01, -2.2414e-01,  5.9150e-01],
                                            [ 0.0000e+00,  0.0000e+00,  0.0000e+00,  1.0000e+00]]
                                         )[None,None].expand(model_input["trgt_rgb"].size(0),model_input["trgt_rgb"].size(1),-1,-1).cuda()

    c2w = model_input["ctxt_c2w"]
    circ_scale = c2w[0,0,[0,2],-1].norm()
    thetas=np.linspace(0,np.pi*2,n)
    rgb_imgs=[]
    depth_imgs=[]
    start_theta=0#(model_input["ctxt_c2w"][0,0,0,-1]/circ_scale).arccos()
    
    resolution = list(model_input["ctxt_rgb"].flatten(0,1).permute(0,2,3,1).shape)
    frames=[]
    thetas = np.linspace(0, 1, n)
    with torch.no_grad(): sample_out = model(model_input)
    if "flow_inp" in sample_out: model_input["bwd_flow"]=sample_out["flow_inp"]
    step=2 if n==8 else 4
    zs = torch.cat((torch.linspace(0,-n//4,n//step),torch.linspace(-n//4,0,n//step),torch.linspace(0,n//4,n//step),torch.linspace(n//4,0,n//step)))
    for i in range(n):

        print(i,n)
        theta=float(thetas[i] + start_theta)
        x=np.cos(theta) * circ_scale * .005
        y=np.sin(theta) * circ_scale * .005
        tmp=torch.eye(4).cuda()
        newpos=torch.tensor([x,y,zs[i]*2e-1]).cuda().float()
        tmp[:3,-1] = newpos
        custom_c2w = tmp[None].expand(c2w.size(0),c2w.size(1),-1,-1)

        with torch.no_grad(): model_out = model(model_input,time_i=i/(n-1),full_img=True,custom_transf=custom_c2w)
        rgb_pred = model_out["rgb"]
        same_all=True
        if same_all:
            resolution = list(model_input["ctxt_rgb"][:,:1].flatten(0,1).permute(0,2,3,1).shape)
            rgb_pred=rgb_pred[:,:1].view(resolution).permute(1,0,2,3).flatten(1,2).cpu().numpy()
        else:
            rgb_pred=rgb_pred.view(resolution).permute(1,0,2,3).flatten(1,2).cpu().numpy()
        depth_pred = model_out["depth"].clone()
        mind,maxd=sample_out["depth"].cpu().min(),sample_out["depth"].cpu().max()
        depth_pred[0,0,0]=mind #normalize
        depth_pred[0,0,1]=maxd #normalize
        if same_all:
            depth_pred = (mind/(1e-3+depth_pred[:,:1]).view(resolution[:-1]).permute(1,0,2).flatten(1,2).cpu().numpy())
        else:
            depth_pred = (mind/(1e-3+depth_pred).view(resolution[:-1]).permute(1,0,2).flatten(1,2).cpu().numpy())
        magma = cm.get_cmap('magma')
        magma_depth = torch.from_numpy(magma(depth_pred))[...,:3]
        rgbd_im=torch.cat((torch.from_numpy(rgb_pred),magma_depth),0).numpy()
        frames.append(rgbd_im)

    return frames

# Renders out context frame with novel camera pose
def render_view_interp(model_input,model,resolution,n):

    resolution = list(model_input["ctxt_rgb"].flatten(0,1).permute(0,2,3,1).shape)

    c2w = torch.eye(4, device='cuda')[None]
    tmp = torch.eye(4).cuda()
    circ_scale = .1
    thetas = np.linspace(0, 2 * np.pi, n)
    frames = []

    if "ctxt_c2w" not in model_input:
        model_input["ctxt_c2w"] = torch.tensor([[-2.5882e-01, -4.8296e-01,  8.3652e-01, -2.2075e+00],
                                            [ 2.1187e-08, -8.6603e-01, -5.0000e-01,  2.3660e+00],
                                            [-9.6593e-01,  1.2941e-01, -2.2414e-01,  5.9150e-01],
                                            [ 0.0000e+00,  0.0000e+00,  0.0000e+00,  1.0000e+00]]
                                         )[None,None].expand(model_input["trgt_rgb"].size(0),model_input["trgt_rgb"].size(1),-1,-1).cuda()

    c2w = model_input["ctxt_c2w"]
    circ_scale = c2w[0,0,[0,2],-1].norm()
    #circ_scale = c2w[0,[0,1],-1].norm()
    thetas=np.linspace(0,np.pi*2,n)
    rgb_imgs=[]
    depth_imgs=[]
    start_theta=0#(model_input["ctxt_c2w"][0,0,0,-1]/circ_scale).arccos()
    with torch.no_grad(): sample_out = model(model_input)
    if "flow_inp" in sample_out: model_input["bwd_flow"]=sample_out["flow_inp"]
    for i in range(n):
        print(i,n)
        theta=float(thetas[i] + start_theta)
        x=np.cos(theta) * circ_scale * 1
        y=np.sin(theta) * circ_scale * 1
        tmp=torch.eye(4).cuda()
        #newpos=torch.tensor([x,y,c2w[0,2,-1]]).cuda().float()
        newpos=torch.tensor([x,c2w[0,0,1,-1],y]).cuda().float()
        rot = look_at(newpos,torch.tensor([0,0,0]).cuda())
        rot[:,1:]*=-1
        tmp[:3,:3]=rot
        newpos=torch.tensor([x,c2w[0,0,1,-1],y]).cuda().float()
        tmp[:3,-1] = newpos
        #with torch.no_grad(): model_out = model(model_input,custom_transf=tmp[None].expand(c2w.size(0),-1,-1))
        custom_c2w = tmp[None].expand(c2w.size(0),c2w.size(1),-1,-1)
        #TODO make circle radius and only use first img
        #from pdb import set_trace as pdb_;pdb_() 
        if 1:
            custom_c2w = custom_c2w.inverse()@model_input["ctxt_c2w"]
            #custom_c2w = model_input["ctxt_c2w"].inverse()@custom_c2w
        with torch.no_grad(): model_out = model(model_input,custom_transf=custom_c2w,full_img=True)

        resolution = [model_input["trgt_rgb"].size(0)]+list(resolution[1:])

        b = model_out["rgb"].size(0)
        rgb_pred = model_out["rgb"][:,0].view(resolution).permute(1,0,2,3).flatten(1,2).cpu().numpy()
        magma_depth = model_out["depth"][:,0].view(resolution).permute(1,0,2,3).flatten(1,2).cpu()
        #depth_pred = model_out["depth"][:,0].clone()
        #mind,maxd=sample_out["depth"].cpu().min(),sample_out["depth"].cpu().max()
        #depth_pred[0,0]=mind #normalize
        #depth_pred[0,1]=maxd #normalize
        #depth_pred = (mind/(1e-3+depth_pred).view(resolution[:-1]).permute(1,0,2).flatten(1,2).cpu().numpy())
        #magma = cm.get_cmap('magma')
        #magma_depth = torch.from_numpy(magma(depth_pred))[...,:3]
        rgbd_im=torch.cat((torch.from_numpy(rgb_pred),magma_depth),0).numpy()
        frames.append(rgbd_im)
    return frames

def wandb_summary(loss, model_output, model_input, ground_truth, resolution,prefix=""):

    resolution = list(model_input["ctxt_rgb"].flatten(0,1).permute(0,2,3,1).shape)
    resolution[0]=ground_truth["trgt_rgb"].size(1)*ground_truth["trgt_rgb"].size(0)
    nrow=model_input["trgt_rgb"].size(1)
    imsl=model_input["ctxt_rgb"].shape[-2:]
    inv = lambda x : 1/(x+1e-8)

    depth = make_grid(model_output["depth"].cpu().flatten(0,1).permute(0,2,1).unflatten(-1,imsl).detach(),nrow=nrow)

    wandb_out = {
        "est/rgb_pred": make_grid(model_output["rgb"].cpu().flatten(0,1).permute(0,2,1).unflatten(-1,imsl).detach(),nrow=nrow),
        "ref/rgb_gt": make_grid(ground_truth["trgt_rgb"].cpu().flatten(0,1).permute(0,2,1).unflatten(-1,imsl).detach(),nrow=nrow),
        #"ref/rgb_gt": make_grid(ground_truth["trgt_rgb"].cpu().view(*resolution).detach().permute(0, -1, 1, 2),nrow=nrow),
        "ref/ctxt_img": make_grid(model_input["ctxt_rgb"][:,0].cpu().detach(),nrow=1)*.5+.5,
        "est/depth": depth,
        "est/depth_1ch":make_grid(model_output["depth_raw"].flatten(0,1).permute(0,2,1).unflatten(-1,imsl).cpu(),normalize=True,nrow=nrow),
    }

    depthgt = (ground_truth["trgt_depth"] if "trgt_depth" in ground_truth else model_output["trgt_depth_inp"] if "trgt_depth_inp" in model_output 
                    else model_input["trgt_depth"] if "trgt_depth" in model_input else None)
    #if depthgt is not None:
    #    depthgt = make_grid(inv(depthgt).cpu().view(*resolution).detach(),normalize=True,nrow=nrow)
    #    wandb_out["ref/depthgt"]= depthgt

    if "ctxt_rgb" in model_output:
        wandb_out["est/ctxt_depth"] =make_grid(model_output["ctxt_depth"].cpu().flatten(0,1).permute(0,2,1).unflatten(-1,imsl).detach(),nrow=nrow)
        wandb_out["est/ctxt_rgb_pred"] = ctxt_rgb_pred = make_grid(model_output["ctxt_rgb"].cpu().view(*resolution).detach().permute(0, -1, 1, 2),nrow=nrow)

    if "corr_weights" in model_output:
        #corr_weights = make_grid(model_output["corr_weights"].flatten(0,1)[:,:1].cpu().detach(),normalize=False,nrow=nrow)
        corr_weights = make_grid(ch_fst(model_output["corr_weights"],resolution[1]).flatten(0,1)[:,:1].cpu().detach(),normalize=False,nrow=nrow)
        wandb_out["est/corr_weights"] = corr_weights

    if "flow_from_pose" in model_output and not torch.isnan(model_output["flow_from_pose"]).any() and not torch.isnan(model_output["flow_from_pose"]).any():
        #psnr = piqa.PSNR()(ch_fst(model_output["rgb"],imsl[0]).flatten(0,1).contiguous(),ch_fst(ground_truth["trgt_rgb"],imsl[0]).flatten(0,1).contiguous())
        #wandb.log({prefix+"metrics/psnr": psnr})

        gt_flow_bwd = flow_vis_torch.flow_to_color(make_grid(model_output["flow_inp"].flatten(0,1),nrow=nrow))/255
        wandb_out["ref/flow_gt_bwd"]=gt_flow_bwd
        if "flow_from_pose" in model_output:
            wandb_out["est/flow_est_pose"] = flow_vis_torch.flow_to_color(make_grid(model_output["flow_from_pose"].flatten(0,1).permute(0,2,1).unflatten(-1,imsl),nrow=nrow))/255
        if "flow_from_pose_render" in model_output:
            wandb_out["est/flow_est_pose_render"] = flow_vis_torch.flow_to_color(make_grid(model_output["flow_from_pose_render"].flatten(0,1).permute(0,2,1).unflatten(-1,imsl),nrow=nrow))/255
    else:
        print("skipping flow plotting")
    #for k,v in wandb_out.items(): print(k,v.max(),v.min())
    #for k,v in wandb_out.items():plt.imsave("img/%s.png"%k,v.permute(1,2,0).detach().cpu().numpy().clip(0,1));
    #print("saving locally")
    #zz
    #wandb.log({"sanity/"+k+"_min":v.min() for k,v in wandb_out.items()})
    #wandb.log({"sanity/"+k+"_max":v.max() for k,v in wandb_out.items()})
    #for k,v in wandb_out.items(): print(v.shape)
    wandb_out = {prefix+k:wandb.Image(v.permute(1, 2, 0).float().detach().clip(0,1).cpu().numpy()) for k,v in wandb_out.items()}

    wandb.log(wandb_out)


def pose_summary(loss, model_output, model_input, ground_truth, resolution,prefix=""):
    # Log points and boxes in W&B
    point_scene = wandb.Object3D({
        "type": "lidar/beta",
        "points":  model_output["poses"][:,:3,-1].cpu().numpy(),
    })
    wandb.log({"camera positions": point_scene})


    
