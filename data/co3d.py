# note for davis dataloader later: temporally consistent depth estimator: https://github.com/yu-li/TCMonoDepth
# note for cool idea of not even downloading data and just streaming from youtube:https://gist.github.com/Mxhmovd/41e7690114e7ddad8bcd761a76272cc3
import matplotlib.pyplot as plt; 
import cv2
import os
import multiprocessing as mp
import torch.nn.functional as F
import torch
import random
import imageio
import numpy as np
from glob import glob
from collections import defaultdict
from pdb import set_trace as pdb
from itertools import combinations
from random import choice
import matplotlib.pyplot as plt
import imageio.v3 as iio

from torchvision import transforms

import sys

from glob import glob
import os
import gzip
import json
import numpy as np

from PIL import Image
def _load_16big_png_depth(depth_png) -> np.ndarray:
    with Image.open(depth_png) as depth_pil:
        # the image is stored with 16-bit depth but PIL reads it as I (32 bit).
        # we cast it to uint16, then reinterpret as float16, then cast to float32
        depth = (
            np.frombuffer(np.array(depth_pil, dtype=np.uint16), dtype=np.float16)
            .astype(np.float32)
            .reshape((depth_pil.size[1], depth_pil.size[0]))
        )
    return depth
def _load_depth(path, scale_adjustment) -> np.ndarray:
    d = _load_16big_png_depth(path) * scale_adjustment
    d[~np.isfinite(d)] = 0.0
    return d[None]  # fake feature channel

# Geometry functions below used for calculating depth, ignore
def glob_imgs(path):
    imgs = []
    for ext in ["*.png", "*.jpg", "*.JPEG", "*.JPG"]:
        imgs.extend(glob(os.path.join(path, ext)))
    return imgs


def pick(list, item_idcs):
    if not list:
        return list
    return [list[i] for i in item_idcs]


def parse_intrinsics(intrinsics):
    fx = intrinsics[..., 0, :1]
    fy = intrinsics[..., 1, 1:2]
    cx = intrinsics[..., 0, 2:3]
    cy = intrinsics[..., 1, 2:3]
    return fx, fy, cx, cy


from einops import rearrange, repeat
ch_sec = lambda x: rearrange(x,"... c x y -> ... (x y) c")
hom = lambda x, i=-1: torch.cat((x, torch.ones_like(x.unbind(i)[0].unsqueeze(i))), i)


def expand_as(x, y):
    if len(x.shape) == len(y.shape):
        return x

    for i in range(len(y.shape) - len(x.shape)):
        x = x.unsqueeze(-1)

    return x


def lift(x, y, z, intrinsics, homogeneous=False):
    """

    :param self:
    :param x: Shape (batch_size, num_points)
    :param y:
    :param z:
    :param intrinsics:
    :return:
    """
    fx, fy, cx, cy = parse_intrinsics(intrinsics)

    x_lift = (x - expand_as(cx, x)) / expand_as(fx, x) * z
    y_lift = (y - expand_as(cy, y)) / expand_as(fy, y) * z

    if homogeneous:
        return torch.stack((x_lift, y_lift, z, torch.ones_like(z).to(x.device)), dim=-1)
    else:
        return torch.stack((x_lift, y_lift, z), dim=-1)


def world_from_xy_depth(xy, depth, cam2world, intrinsics):
    batch_size, *_ = cam2world.shape

    x_cam = xy[..., 0]
    y_cam = xy[..., 1]
    z_cam = depth

    pixel_points_cam = lift(
        x_cam, y_cam, z_cam, intrinsics=intrinsics, homogeneous=True
    )
    world_coords = torch.einsum("b...ij,b...kj->b...ki", cam2world, pixel_points_cam)[
        ..., :3
    ]

    return world_coords


def get_ray_directions(xy, cam2world, intrinsics, normalize=True):
    z_cam = torch.ones(xy.shape[:-1]).to(xy.device)
    pixel_points = world_from_xy_depth(
        xy, z_cam, intrinsics=intrinsics, cam2world=cam2world
    )  # (batch, num_samples, 3)

    cam_pos = cam2world[..., :3, 3]
    ray_dirs = pixel_points - cam_pos[..., None, :]  # (batch, num_samples, 3)
    if normalize:
        ray_dirs = F.normalize(ray_dirs, dim=-1)
    return ray_dirs

from PIL import Image
def _load_16big_png_depth(depth_png) -> np.ndarray:
    with Image.open(depth_png) as depth_pil:
        # the image is stored with 16-bit depth but PIL reads it as I (32 bit).
        # we cast it to uint16, then reinterpret as float16, then cast to float32
        depth = (
            np.frombuffer(np.array(depth_pil, dtype=np.uint16), dtype=np.float16)
            .astype(np.float32)
            .reshape((depth_pil.size[1], depth_pil.size[0]))
        )
    return depth
def _load_depth(path, scale_adjustment) -> np.ndarray:
    d = _load_16big_png_depth(path) * scale_adjustment
    d[~np.isfinite(d)] = 0.0
    return d[None]  # fake feature channel

# NOTE currently using CO3D V1 because they switch to NDC cameras in 2. TODO is to make conversion code (different intrinsics), verify pointclouds, and switch. 

class Co3DNoCams(torch.utils.data.Dataset):
    """Dataset for a class of objects, where each datapoint is a SceneInstanceDataset."""

    def __init__(
        self,
        num_context=2,
        n_skip=1,
        num_trgt=1,
        low_res=(128,144),
        depth_scale=1,#1.8/5,
        val=False,
        num_cat=1000,
        overfit=False,
        category=None,
        use_mask=False,
        use_v1=True,
        # delete below, not used
        vary_context_number=False,
        query_sparsity=None,
        img_sidelength=None,
        input_img_sidelength=None,
        max_num_instances=None,
        max_observations_per_instance=None,
        specific_observation_idcs=None,
        test=False,
        test_context_idcs=None,
        context_is_last=False,
        context_is_first=False,
        cache=None,
        video=True,
    ):

        if num_cat is None: num_cat=1000

        self.n_trgt=num_trgt
        self.use_mask=use_mask
        self.depth_scale=depth_scale
        self.of=overfit
        self.val=val

        self.num_skip=n_skip
        self.low_res=low_res
        max_num_instances = None

        self.base_path=os.environ['CO3D_ROOT']
        print(self.base_path)

        # Get sequences!
        from collections import defaultdict
        sequences = defaultdict(list)
        self.total_num_data=0
        self.all_frame_names=[]
        all_cats = [ "hydrant","teddybear","apple", "ball", "bench", "cake", "donut", "plant", "suitcase", "vase","backpack", "banana", "baseballbat", "baseballglove",  "bicycle", "book", "bottle", "bowl", "broccoli",  "car", "carrot", "cellphone", "chair", "couch", "cup",  "frisbee", "hairdryer", "handbag", "hotdog", "keyboard", "kite", "laptop", "microwave", "motorcycle", "mouse", "orange", "parkingmeter", "pizza",  "remote", "sandwich", "skateboard", "stopsign",  "toaster", "toilet", "toybus", "toyplane", "toytrain", "toytruck", "tv", "umbrella",  "wineglass", ]

        for cat in (all_cats[:num_cat]) if category is None else [category]:
            print(cat)
            dataset = json.loads(gzip.GzipFile(os.path.join(self.base_path,cat,"frame_annotations.jgz"),"rb").read().decode("utf8"))
            val_amt = int(len(dataset)*.03)
            dataset = dataset[:-val_amt] if not val else dataset[-val_amt:]
            self.total_num_data+=len(dataset)
            for i,data in enumerate(dataset):
                self.all_frame_names.append((data["sequence_name"],data["frame_number"]))
                sequences[data["sequence_name"]].append(data)
    
        sorted_seq={}
        for k,v in sequences.items():
            sorted_seq[k]=sorted(sequences[k],key=lambda x:x["frame_number"])
        #for k,v in sequences.items(): sequences[k]=v[:-(max(self.num_skip) if type(self.num_skip)==list else self.num_skip)*self.n_trgt]
        self.seqs = sorted_seq

        print("done with dataloader init")

    def sparsify(self, dict, sparsity):
        new_dict = {}
        if sparsity is None:
            return dict
        else:
            # Sample upper_limit pixel idcs at random.
            rand_idcs = np.random.choice(
                self.img_sidelength ** 2, size=sparsity, replace=False
            )
            for key in ["rgb", "uv"]:
                new_dict[key] = dict[key][rand_idcs]

            for key, v in dict.items():
                if key not in ["rgb", "uv"]:
                    new_dict[key] = dict[key]

            return new_dict

    def set_img_sidelength(self, new_img_sidelength):
        """For multi-resolution training: Updates the image sidelength with which images are loaded."""
        self.img_sidelength = new_img_sidelength
        for instance in self.all_instances:
            instance.set_img_sidelength(new_img_sidelength)

    def __len__(self):
        return self.total_num_data

    def collate_fn(self, batch_list):
        keys = batch_list[0].keys()
        result = defaultdict(list)

        for entry in batch_list:
            # make them all into a new dict
            for key in keys:
                result[key].append(entry[key])

        for key in keys:
            try:
                result[key] = torch.stack(result[key], dim=0)
            except:
                continue

        return result

    def __getitem__(self, idx,seq_query=None):

        context = []
        trgt = []
        post_input = []

        n_skip = (random.choice(self.num_skip) if type(self.num_skip)==list else self.num_skip) + 1

        if seq_query is None:
            try: 
                seq_name,frame_idx=self.all_frame_names[idx]
            except: 
                print(f"Out of bounds erorr at {idx}. Investigate.")
                return self[-2*n_skip*self.n_trgt if self.val else np.random.randint(len(self))]
        
        if seq_query is not None:
            frame_idx=idx
            seq_name = list(self.seqs.keys())[seq_query]
            all_frames= self.seqs[seq_name]
        else:
            all_frames=self.seqs[seq_name] if not self.of else self.seqs[random.choice(list(self.seqs.keys())[:int(self.of)])]

        if len(all_frames)<=self.n_trgt*n_skip or frame_idx >= (len(all_frames)-self.n_trgt*n_skip):
            frame_idx=0
            if len(all_frames)<=self.n_trgt*n_skip or frame_idx >= (len(all_frames)-self.n_trgt*n_skip):
                if len(all_frames)<=self.n_trgt*n_skip:
                    print(len(all_frames) ," frames < ",self.n_trgt*n_skip," queries")
                print("returning low/high")
                return self[-2*n_skip*self.n_trgt if self.val else np.random.randint(len(self))]
        start_idx = frame_idx 

        if self.of and 1: start_idx=0

        frames = all_frames[start_idx:start_idx+self.n_trgt*n_skip:n_skip]
        if np.random.rand()<.5 and not self.of and not self.val: frames=frames[::-1]

        paths = [os.path.join(self.base_path,x["image"]["path"]) for x in frames]
        for path in paths:
            if not os.path.exists(path):
                print("path missing")
                return self[np.random.randint(len(self))]

        #masks=[torch.from_numpy(plt.imread(os.path.join(self.base_path,x["mask"]["path"]))) for x in frames]
        imgs=[torch.from_numpy(plt.imread(path)) for path in paths]

        Ks=[]
        c2ws=[]
        depths=[]
        for data in frames:

            #depths.append(torch.from_numpy(_load_depth(os.path.join(self.base_path,data["depth"]["path"]), data["depth"]["scale_adjustment"])[0])) # commenting out since slow to load; uncomment when needed

            # Below pose processing taken from co3d github issue
            p = data["viewpoint"]["principal_point"]
            f = data["viewpoint"]["focal_length"]
            h, w = data["image"]["size"]
            K = np.eye(3)
            s = (min(h, w)) / 2
            K[0, 0] = f[0] * (w) / 2
            K[1, 1] = f[1] * (h) / 2
            K[0, 2] = -p[0] * s + (w) / 2
            K[1, 2] = -p[1] * s + (h) / 2

            # Normalize intrinsics to [-1,1]
            #print(K)
            raw_K=[torch.from_numpy(K).clone(),[h,w]]
            K[:2] /= torch.tensor([w, h])[:, None]
            Ks.append(torch.from_numpy(K).float())

            R = np.asarray(data["viewpoint"]["R"]).T   # note the transpose here
            T = np.asarray(data["viewpoint"]["T"]) * self.depth_scale
            pose = np.concatenate([R,T[:,None]],1)
            pose = torch.from_numpy( np.diag([-1,-1,1]).astype(np.float32) @ pose )# flip the direction of x,y axis
            tmp=torch.eye(4)
            tmp[:3,:4]=pose
            c2ws.append(tmp.inverse())

        Ks=torch.stack(Ks)
        c2w=torch.stack(c2ws).float()

        no_mask=0
        if no_mask:
            masks=[x*0+1 for x in masks]

        low_res=self.low_res#(128,144)#(108,144)
        minx,miny=min([x.size(0) for x in imgs]),min([x.size(1) for x in imgs])

        imgs=[x[:minx,:miny].float() for x in imgs]

        if self.use_mask: # mask images and depths
            imgs = [x*y.unsqueeze(-1)+(255*(1-y).unsqueeze(-1)) for x,y in zip(imgs,masks)]
            depths = [x*y for x,y in zip(depths,masks)]

        large_scale=2
        imgs_large = F.interpolate(torch.stack([x.permute(2,0,1) for x in imgs]),(int(256*large_scale),int(288*large_scale)),antialias=True,mode="bilinear")
        imgs_med = F.interpolate(torch.stack([x.permute(2,0,1) for x in imgs]),(int(256),int(288)),antialias=True,mode="bilinear")
        imgs = F.interpolate(torch.stack([x.permute(2,0,1) for x in imgs]),low_res,antialias=True,mode="bilinear")

        if self.use_mask:
            imgs = imgs*masks[:,None]+255*(1-masks[:,None])

        imgs = imgs/255 * 2 - 1

        uv = np.mgrid[0:low_res[0], 0:low_res[1]].astype(float).transpose(1, 2, 0)
        uv = torch.from_numpy(np.flip(uv, axis=-1).copy()).long()
        uv = uv/ torch.tensor([low_res[1]-1, low_res[0]-1])  # uv in [0,1]
        uv = uv[None].expand(len(imgs),-1,-1,-1).flatten(1,2)

        model_input = {
                "trgt_rgb": imgs[1:],
                "ctxt_rgb": imgs[:-1],
                "trgt_rgb_large": imgs_large[1:],
                "ctxt_rgb_large": imgs_large[:-1],
                "trgt_rgb_med": imgs_med[1:],
                "ctxt_rgb_med": imgs_med[:-1],
                #"ctxt_depth": depths.squeeze(1)[:-1],
                #"trgt_depth": depths.squeeze(1)[1:],
                "intrinsics": Ks[1:],
                "trgt_c2w": c2w[1:],
                "ctxt_c2w": c2w[:-1],
                "x_pix": uv[1:],
                #"trgt_mask": masks[1:],
                #"ctxt_mask": masks[:-1],
                }

        gt = {
                #"paths": paths,
                #"raw_K": raw_K,
                #"seq_name": seq_name,
                "trgt_rgb": ch_sec(imgs[1:])*.5+.5,
                "ctxt_rgb": ch_sec(imgs[:-1])*.5+.5,
                #"ctxt_depth": depths.squeeze(1)[:-1].flatten(1,2).unsqueeze(-1),
                #"trgt_depth": depths.squeeze(1)[1:].flatten(1,2).unsqueeze(-1),
                "intrinsics": Ks[1:],
                "x_pix": uv[1:],
                #"seq_name": [seq_name],
                #"trgt_mask": masks[1:].flatten(1,2).unsqueeze(-1),
                #"ctxt_mask": masks[:-1].flatten(1,2).unsqueeze(-1),
                }

        return model_input,gt
