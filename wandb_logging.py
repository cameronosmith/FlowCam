import os
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

def write_video(save_dir,frames,vid_name,step,write_frames=False):
    frames = [(255*x).astype(np.uint8) for x in frames]
    if "time" in vid_name: frames = frames + frames[::-1]
    f = os.path.join(save_dir, f'{vid_name}_{step}.mp4')
    imageio.mimwrite(f, frames, fps=8, quality=7)
    wandb.log({f'vid/{vid_name}':wandb.Video(f, format='mp4', fps=8)})
    print("writing video at",f)
    if write_frames:
        for i,img in enumerate(frames):
            try: os.mkdir(os.path.join(save_dir, f'{vid_name}_{step}'))
            except:pass
            f=os.path.join(save_dir, f'{vid_name}_{step}/{i}.png');plt.imsave(f,img);print(f)

def normalize(a):
    return (a - a.min()) / (a.max() - a.min())

def cvt(a):
    a = a.permute(1, 2, 0).detach().cpu()
    a = (a - a.min()) / (a.max() - a.min())
    a = a.numpy()
    return a

ch_fst = lambda src,x=None:rearrange(src,"... (x y) c -> ... c x y",x=int(src.size(-2)**(.5)) if x is None else x)

def _wandb_summary(loss, model_output, model_input, ground_truth, resolution,prefix=""):

    resolution = list(model_input["ctxt_rgb"].flatten(0,1).shape)

    nrow=model_input["trgt_rgb"].size(1)
    imsly,imslx=model_input["ctxt_rgb"].shape[-2:]

    resolution = list(model_input["ctxt_rgb"].flatten(0,1).permute(0,2,3,1).shape)

    rgb_gt= ground_truth["trgt_rgb"]
    rgb_pred,depth,=[model_output[x] for x in ["rgb","depth"]]

    inv = lambda x : 1/(x+1e-3)
    depth = make_grid(model_output["depth"].flatten(0,1).permute(0,2,1).unflatten(-1,(imsly,imslx)).cpu(),normalize=True,nrow=nrow)

    rgb_pred = make_grid(model_output["rgb"].flatten(0,1).permute(0,2,1).unflatten(-1,(imsly,imslx)),normalize=True,nrow=nrow)
    rgb_gt = make_grid(ground_truth["trgt_rgb"].flatten(0,1).permute(0,2,1).unflatten(-1,(imsly,imslx)),normalize=True,nrow=nrow)
    ctxt_img = make_grid(model_input["ctxt_rgb"].cpu().flatten(0,1),normalize=True,nrow=nrow)

    print("add psnr metric here")

    wandb_out = {
        "est/rgb_pred": rgb_pred,
        "ref/rgb_gt": rgb_gt,
        "ref/ctxt_img": ctxt_img,
        "est/depth": depth,
    }
    if "trgt_depth" in ground_truth:
        wandb_out["depthgt"]=make_grid(ground_truth["trgt_depth"].cpu().flatten(0,1).permute(0,2,1).unflatten(-1,(imsly,imslx)).cpu(),normalize=True,nrow=nrow)

    for k,v in wandb_out.items(): print(k,v.max(),v.min(),v.shape)
    #for k,v in wandb_out.items():plt.imsave("/nobackup/users/camsmith/img/%s.png"%k,v.permute(1,2,0).detach().cpu().numpy().clip(0,1));
    wandb.log({"sanity/"+k+"_min":v.min() for k,v in wandb_out.items()})
    wandb.log({"sanity/"+k+"_max":v.max() for k,v in wandb_out.items()})
    wandb_out = {prefix+k:wandb.Image(v.permute(1, 2, 0).detach().clip(0,1).cpu().numpy()) for k,v in wandb_out.items()}

    wandb.log(wandb_out)
    
#def dyn_wandb_summary(loss, model_output, model_input, ground_truth, resolution,prefix=""):
def wandb_summary(loss, model_output, model_input, ground_truth, resolution,prefix=""):

    resolution = list(model_input["ctxt_rgb"].flatten(0,1).permute(0,2,3,1).shape)
    nrow=model_input["trgt_rgb"].size(1)
    imsl=model_input["ctxt_rgb"].shape[-2:]
    inv = lambda x : 1/(x+1e-8)

    depth = make_grid(model_output["depth"].cpu().flatten(0,1).permute(0,2,1).unflatten(-1,imsl).detach(),nrow=nrow)

    wandb_out = {
        "est/rgb_pred": make_grid(model_output["rgb"].cpu().flatten(0,1).permute(0,2,1).unflatten(-1,imsl).detach(),nrow=nrow),
        "ref/rgb_gt": make_grid(ground_truth["trgt_rgb"].cpu().view(*resolution).detach().permute(0, -1, 1, 2),nrow=nrow),
        "ref/ctxt_img": make_grid(model_input["ctxt_rgb"][:,0].cpu().detach(),nrow=1)*.5+.5,
        "est/depth": depth,
        "est/depth_1ch":make_grid(model_output["depth_raw"].flatten(0,1).permute(0,2,1).unflatten(-1,imsl).cpu(),normalize=True,nrow=nrow),
    }

    depthgt = (ground_truth["trgt_depth"] if "trgt_depth" in ground_truth else model_output["trgt_depth_inp"] if "trgt_depth_inp" in model_output 
                    else model_input["trgt_depth"] if "trgt_depth" in model_input else None)
    if depthgt is not None:
        depthgt = make_grid(inv(depthgt).cpu().view(*resolution[:3]).detach().unsqueeze(1),normalize=True,nrow=nrow)
        wandb_out["ref/depthgt"]= depthgt

    if "fine_rgb" in model_output:
        wandb_out["est/fine_rgb_pred"] = make_grid(model_output["fine_rgb"].cpu().flatten(0,1).permute(0,2,1).unflatten(-1,imsl).detach(),nrow=nrow)
        wandb_out["est/fine_depth_pred"] = make_grid(model_output["fine_depth"].cpu().flatten(0,1).permute(0,2,1).unflatten(-1,imsl).detach(),nrow=nrow,normalize=True)

    if "ctxt_rgb" in model_output:
        wandb_out["est/ctxt_depth"] =make_grid(model_output["ctxt_depth"].cpu().flatten(0,1).permute(0,2,1).unflatten(-1,imsl).detach(),nrow=nrow)
        wandb_out["est/ctxt_rgb_pred"] = ctxt_rgb_pred = make_grid(model_output["ctxt_rgb"].cpu().view(*resolution).detach().permute(0, -1, 1, 2),nrow=nrow)

    if "corr_weights" in model_output:
        #corr_weights = make_grid(model_output["corr_weights"].flatten(0,1)[:,:1].cpu().detach(),normalize=False,nrow=nrow)
        corr_weights = make_grid(ch_fst(model_output["corr_weights"],resolution[1]).flatten(0,1)[:,:1].cpu().detach(),normalize=False,nrow=nrow)
        wandb_out["est/corr_weights"] = corr_weights

    if "flow" in model_output and not torch.isnan(model_output["flow"]).any() and not torch.isnan(model_output["flow_from_pose"]).any():
        psnr = piqa.PSNR()(ch_fst(model_output["rgb"],imsl[0]).flatten(0,1).contiguous(),ch_fst(ground_truth["trgt_rgb"],imsl[0]).flatten(0,1).contiguous())
        wandb.log({prefix+"metrics/psnr": psnr})

        gt_flow_bwd = flow_vis_torch.flow_to_color(make_grid(model_output["flow_inp"].flatten(0,1),nrow=nrow))/255
        est_flow = flow_vis_torch.flow_to_color(make_grid(model_output["flow"].flatten(0,1).permute(0,2,1).unflatten(-1,imsl),nrow=nrow))/255
        wandb_out["est/flow_est"]= est_flow
        wandb_out["ref/flow_gt_bwd"]=gt_flow_bwd
        if "flow_from_pose" in model_output:
            wandb_out["est/flow_est_pose"] = flow_vis_torch.flow_to_color(make_grid(model_output["flow_from_pose"].flatten(0,1).permute(0,2,1).unflatten(-1,imsl),nrow=nrow))/255
    elif "flow" in model_output: 
        print("skipping nan flow")
    for k,v in wandb_out.items(): print(k,v.max(),v.min())
    #for k,v in wandb_out.items():plt.imsave("/nobackup/users/camsmith/img/%s.png"%k,v.permute(1,2,0).detach().cpu().numpy().clip(0,1));
    #zz
    #wandb.log({"sanity/"+k+"_min":v.min() for k,v in wandb_out.items()})
    #wandb.log({"sanity/"+k+"_max":v.max() for k,v in wandb_out.items()})
    #for k,v in wandb_out.items(): print(v.shape)
    wandb_out = {prefix+k:wandb.Image(v.permute(1, 2, 0).float().detach().clip(0,1).cpu().numpy()) for k,v in wandb_out.items()}

    wandb.log(wandb_out)
