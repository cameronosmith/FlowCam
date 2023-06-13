#usage: python demo.py --demo_rgb /nobackup/projects/public/facebook-co3dv2/hydrant/615_99120_197713/images --intrinsics 1.7671e+03,3.1427e+03,5.3550e+02,9.5150e+02 -c ~/storage/logs/hydrant_skip123.pt --render_imgs --low_res 144 128

from run import *

import torch
import random
from torch.utils.data import Dataset
from glob import glob
import imageio
import numpy as np
import matplotlib.pyplot as plt

import torch.nn.functional as F

from einops import rearrange, repeat
hom = lambda x, i=-1: torch.cat((x, torch.ones_like(x.unbind(i)[0].unsqueeze(i))), i)
ch_sec = lambda x: rearrange(x,"... c x y -> ... (x y) c")

# A quick dummy dataset for the demo rgb folder
class SingleVid(Dataset):

    # If specified here, intrinsics should be a 4-element array of [fx,fy,cx,cy] at input image resolution
    def __init__(self, img_dir,intrinsics=None,n_trgt=6,num_skip=0,low_res=None,hi_res=None):
        self.low_res,self.intrinsics,self.n_trgt,self.num_skip,self.hi_res=low_res,intrinsics,n_trgt,num_skip,hi_res
        if self.hi_res is None:self.hi_res=[x*2 for x in self.low_res]
        self.hi_res = [(x+x%64) for x in self.hi_res]

        self.img_paths = glob(img_dir + '/*.png') + glob(img_dir + '/*.jpg')
        self.img_paths.sort()
    
    def __len__(self):
        return len(self.img_paths)-(1+self.n_trgt)*(1+self.num_skip)

    def __getitem__(self, idx):
        
        n_skip=self.num_skip+1
        paths = self.img_paths[idx:idx+self.n_trgt*n_skip:n_skip]
        imgs=torch.stack([torch.from_numpy(plt.imread(path)).permute(2,0,1) for path in paths]).float()

        imgs_large = F.interpolate(imgs,self.hi_res,antialias=True,mode="bilinear")
        frames = F.interpolate(imgs,self.low_res)

        frames = frames/255 * 2 - 1

        uv = np.mgrid[0:self.low_res[0], 0:self.low_res[1]].astype(float).transpose(1, 2, 0)
        uv = torch.from_numpy(np.flip(uv, axis=-1).copy()).long()
        uv = uv/ torch.tensor([self.low_res[1], self.low_res[0]])  # uv in [0,1]
        uv = uv[None].expand(len(frames),-1,-1,-1).flatten(1,2)

        #imgs large values in [0,255], imgs in [-1,1], gt_rgb in [0,1],

        model_input = {
                "trgt_rgb": frames[1:],
                "ctxt_rgb": frames[:-1],
                "trgt_rgb_large": imgs_large[1:],
                "ctxt_rgb_large": imgs_large[:-1],
                "x_pix": uv[1:],
                }
        gt = {
                "trgt_rgb": ch_sec(frames[1:])*.5+.5,
                "ctxt_rgb": ch_sec(frames[:-1])*.5+.5,
                "x_pix": uv[1:],
                }

        if self.intrinsics is not None:
            K = torch.eye(3)
            K[0,0],K[1,1],K[0,2],K[1,2]=[float(x) for x in self.intrinsics.strip().split(",")]
            h,w=imgs[0].shape[-2:]
            K[:2] /= torch.tensor([w, h])[:, None]
            model_input["intrinsics"] = K[None].expand(self.n_trgt-1,-1,-1)

        return model_input,gt

dataset=SingleVid(args.demo_rgb,args.intrinsics,args.vid_len,args.n_skip,args.low_res)

all_poses = torch.tensor([]).cuda()
all_render_rgb=torch.tensor([]).cuda()
all_render_depth=torch.tensor([])
for seq_i in range(len(dataset)//(dataset.n_trgt)):
    print(seq_i*(dataset.n_trgt),"/",len(dataset))
    model_input = {k:to_gpu(v)[None] for k,v in dataset.__getitem__(seq_i*(dataset.n_trgt-1))[0].items()}
    with torch.no_grad(): out = (model.forward if not args.render_imgs else model.render_full_img)(model_input)
    curr_transfs = out["poses"][0]
    if len(all_poses): curr_transfs = all_poses[[-1]] @ curr_transfs # integrate poses
    all_poses = torch.cat((all_poses,curr_transfs))
    all_render_rgb = torch.cat((all_render_rgb,out["rgb"][0]))
    all_render_depth = torch.cat((all_render_depth,out["depth"][0]))

out_dir="demo_output/"+args.demo_rgb.replace("/","_")
os.makedirs(out_dir,exist_ok=True)
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot(*all_poses[:,:3,-1].T.cpu().numpy())
ax.xaxis.set_tick_params(labelbottom=False);ax.yaxis.set_tick_params(labelleft=False);ax.zaxis.set_tick_params(labelleft=False)
ax.view_init(elev=10., azim=45)
plt.tight_layout()
fp = os.path.join(out_dir,f"pose_plot.png");plt.savefig(fp,bbox_inches='tight');plt.close()

fp = os.path.join(out_dir,f"poses.npy");np.save(fp,all_poses.cpu())
if args.render_imgs:
    out_dir=os.path.join(out_dir,"renders")
    os.makedirs(out_dir,exist_ok=True)
    for i,(rgb,depth) in enumerate(zip(all_render_rgb.unflatten(1,model_input["trgt_rgb"].shape[-2:]),all_render_depth.unflatten(1,model_input["trgt_rgb"].shape[-2:]))):
        plt.imsave(os.path.join(out_dir,"render_rgb_%04d.png"%i),rgb.clip(0,1).cpu().numpy())
        plt.imsave(os.path.join(out_dir,"render_depth_%04d.png"%i),depth.clip(0,1).cpu().numpy())

