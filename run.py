import os,random,time,datetime,glob

import numpy as np
from functools import partial
from tqdm import tqdm,trange
from einops import rearrange

import wandb
import torch

import models
import vis_scripts

from data.KITTI import KittiDataset
from data.co3d import Co3DNoCams
from data.realestate10k_dataio import RealEstate10k

def to_gpu(ob): return {k: to_gpu(v) for k, v in ob.items()} if isinstance(ob, dict) else ob.cuda()

import argparse
parser = argparse.ArgumentParser(description='simple training job')
# logging parameters
parser.add_argument('-n','--name', type=str,default="",required=False,help="wandb training name")
parser.add_argument('-c','--init_ckpt', type=str,default=None,required=False,help="File for checkpoint loading. If folder specific, will use latest .pt file")
parser.add_argument('-d','--dataset', type=str,default="hydrant")
parser.add_argument('-o','--online', default=False, action=argparse.BooleanOptionalAction)
# data/training parameters
parser.add_argument('-b','--batch_size', type=int,default=1,help="number of videos/sequences per training step")
parser.add_argument('-v','--vid_len', type=int,default=6,help="video length or number of images per batch")
parser.add_argument('--midas_sup', default=False, action=argparse.BooleanOptionalAction,help="Whether to use midas depth supervision or not")
parser.add_argument('--category', type=str,default=None,help="if want to use a specific co3d category, such as 'bicycle', specify here")
# model parameters
parser.add_argument('--n_skip', nargs="+",type=int,default=0,help="Number of frames to skip between adjacent frames in dataloader. If list, dataset randomly chooses between skips. Only used for co3d")
parser.add_argument('--n_ctxt', type=int,default=2,help="Number of context views to use. 1 is just first frame, 2 is second and last, 3 is also middle, etc")
# eval/vis 
parser.add_argument('--eval', default=False, action=argparse.BooleanOptionalAction,help="whether to train or run evaluation")
parser.add_argument('--n_eval', type=int,default=int(1e8),help="Number of eval samples to run")
parser.add_argument('--save_ind', default=False, action=argparse.BooleanOptionalAction,help="whether to save out each individual image (in rendering images) or just save the all-trajectory image")
parser.add_argument('--save_imgs', default=False, action=argparse.BooleanOptionalAction,help="whether to save out the all-trajectory images")
# demo args
parser.add_argument('--demo_rgb', default="", type=str,required=False,help="The image folder path for demo inference.")
parser.add_argument('--render_imgs', default=False, action=argparse.BooleanOptionalAction,help="whether to rerender out images (video reconstructions) during the demo inference (slower than just estimating poses)")
parser.add_argument('--intrinsics', default=None, type=str,required=False,help="The intrinsics corresponding to the image path for demo inference as fx,fy,cx,cy. To use predicted intrinsics, leave as None")
parser.add_argument('--low_res', nargs="+",type=int,default=[128,128],help="Low resolution to perform renderings at. Default (128,128)")

args = parser.parse_args()
if args.n_skip==0 and args.dataset=="realestate": args.n_skip=9 # realestate is the only dataset where 0 frameskip isn't meaningful

# Wandb init: install wandb and initialize via wandb.login() before running
run = wandb.init(project="flowcam",mode="online" if args.online else "disabled",name=args.name)
wandb.run.log_code(".")
save_dir = os.path.join(os.environ.get('LOGDIR', "") , run.name)
print(save_dir)
os.makedirs(save_dir,exist_ok=True)
wandb.save(os.path.join(save_dir, "checkpoint*"))
wandb.save(os.path.join(save_dir, "video*"))

# Make dataset
get_dataset = lambda val=False: ( Co3DNoCams(num_trgt=args.vid_len+1,low_res=(156,102),num_cat=1 if args.dataset=="hydrant" else 10 if args.dataset=="10cat" else 30,
                                             n_skip=args.n_skip,val=val,category=args.category) if args.dataset in ["hydrant","10cat","allcat"] 
                             else RealEstate10k(imsl=128, num_ctxt_views=2, num_query_views=args.vid_len+1, val=val, n_skip = args.n_skip) if args.dataset == "realestate" 
                             else KittiDataset(num_context=1,num_trgt=args.vid_len+1,low_res=(76,250),val=val,n_skip=args.n_skip)
            )
get_dataloader = lambda dataset: iter(torch.utils.data.DataLoader(dataset, batch_size=args.batch_size*torch.cuda.device_count(),num_workers=args.batch_size,shuffle=True,pin_memory=True))

# Make model + optimizer
model = models.FlowCam(near=1,far=8,num_view=args.n_ctxt,use_midas=args.midas_sup).cuda()
if args.init_ckpt is not None:
    ckpt_file = args.init_ckpt if os.path.isfile(os.path.expanduser(args.init_ckpt)) else max(glob.glob(os.path.join(args.init_ckpt,"*.pt")), key=os.path.getctime)
    model.load_state_dict(torch.load(ckpt_file)["model_state_dict"],strict=False)
optim = torch.optim.Adam(lr=1e-4, params=model.parameters())
