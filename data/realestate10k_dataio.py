import random
from torch.nn import functional as F
import os
import torch
import numpy as np
from glob import glob
import json
from collections import defaultdict
import os.path as osp
from imageio import imread
from torch.utils.data import Dataset
from pathlib import Path
import cv2
from tqdm import tqdm
from scipy.io import loadmat

import functools
import cv2
import numpy as np
import imageio
from glob import glob
import os
import shutil
import io

not_of=1

def load_rgb(path, sidelength=None):
    img = imageio.imread(path)[:, :, :3]
    img = skimage.img_as_float32(img)

    img = square_crop_img(img)

    if sidelength is not None:
        img = cv2.resize(img, (sidelength, sidelength), interpolation=cv2.INTER_NEAREST)

    img -= 0.5
    img *= 2.
    return img

def load_depth(path, sidelength=None):
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED).astype(np.float32)

    if sidelength is not None:
        img = cv2.resize(img, (sidelength, sidelength), interpolation=cv2.INTER_NEAREST)

    img *= 1e-4

    if len(img.shape) == 3:
        img = img[:, :, :1]
        img = img.transpose(2, 0, 1)
    else:
        img = img[None, :, :]
    return img


def load_pose(filename):
    lines = open(filename).read().splitlines()
    if len(lines) == 1:
        pose = np.zeros((4, 4), dtype=np.float32)
        for i in range(16):
            pose[i // 4, i % 4] = lines[0].split(" ")[i]
        return pose.squeeze()
    else:
        lines = [[x[0], x[1], x[2], x[3]] for x in (x.split(" ") for x in lines[:4])]
        return np.asarray(lines).astype(np.float32).squeeze()


def load_numpy_hdf5(instance_ds, key):
    rgb_ds = instance_ds['rgb']
    raw = rgb_ds[key][...]
    s = raw.tostring()
    f = io.BytesIO(s)

    img = imageio.imread(f)[:, :, :3]
    img = skimage.img_as_float32(img)

    img = square_crop_img(img)

    img -= 0.5
    img *= 2.

    return img


def load_rgb_hdf5(instance_ds, key, sidelength=None):
    rgb_ds = instance_ds['rgb']
    raw = rgb_ds[key][...]
    s = raw.tostring()
    f = io.BytesIO(s)

    img = imageio.imread(f)[:, :, :3]
    img = skimage.img_as_float32(img)

    img = square_crop_img(img)

    if sidelength is not None:
        img = cv2.resize(img, (sidelength, sidelength), interpolation=cv2.INTER_AREA)

    img -= 0.5
    img *= 2.

    return img


def load_pose_hdf5(instance_ds, key):
    pose_ds = instance_ds['pose']
    raw = pose_ds[key][...]
    ba = bytearray(raw)
    s = ba.decode('ascii')

    lines = s.splitlines()

    if len(lines) == 1:
        pose = np.zeros((4, 4), dtype=np.float32)
        for i in range(16):
            pose[i // 4, i % 4] = lines[0].split(" ")[i]
        # processed_pose = pose.squeeze()
        return pose.squeeze()
    else:
        lines = [[x[0], x[1], x[2], x[3]] for x in (x.split(" ") for x in lines[:4])]
        return np.asarray(lines).astype(np.float32).squeeze()


def cond_mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def square_crop_img(img):
    min_dim = np.amin(img.shape[:2])
    center_coord = np.array(img.shape[:2]) // 2
    img = img[center_coord[0] - min_dim // 2:center_coord[0] + min_dim // 2,
          center_coord[1] - min_dim // 2:center_coord[1] + min_dim // 2]
    return img


def glob_imgs(path):
    imgs = []
    for ext in ['*.png', '*.jpg', '*.JPEG', '*.JPG']:
        imgs.extend(glob(os.path.join(path, ext)))
    return imgs

def augment(rgb, intrinsics, c2w_mat):

    # Horizontal Flip with 50% Probability
    if np.random.uniform(0, 1) < 0.5:
        rgb = rgb[:, ::-1, :]
        tf_flip = np.array([[-1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
        c2w_mat = c2w_mat @ tf_flip

    # Crop by aspect ratio
    if np.random.uniform(0, 1) < 0.5:
        py = np.random.randint(1, 32)
        rgb = rgb[py:-py, :, :]
    else:
        py = 0

    if np.random.uniform(0, 1) < 0.5:
        px = np.random.randint(1, 32)
        rgb = rgb[:, px:-px, :]
    else:
        px = 0

    H, W, _ = rgb.shape
    rgb = cv2.resize(rgb, (256, 256))
    xscale = 256 / W
    yscale = 256 / H

    intrinsics[0, 0] = intrinsics[0, 0] * xscale
    intrinsics[1, 1] = intrinsics[1, 1] * yscale

    return rgb, intrinsics, c2w_mat

class Camera(object):
    def __init__(self, entry):
        fx, fy, cx, cy = entry[1:5]
        self.intrinsics = np.array([[fx, 0, cx, 0],
                                    [0, fy, cy, 0],
                                    [0, 0, 1, 0],
                                    [0, 0, 0, 1]])
        w2c_mat = np.array(entry[7:]).reshape(3, 4)
        w2c_mat_4x4 = np.eye(4)
        w2c_mat_4x4[:3, :] = w2c_mat
        self.w2c_mat = w2c_mat_4x4
        self.c2w_mat = np.linalg.inv(w2c_mat_4x4)


def unnormalize_intrinsics(intrinsics, h, w):
    intrinsics = intrinsics.copy()
    intrinsics[0] *= w
    intrinsics[1] *= h
    return intrinsics


def parse_pose_file(file):
    f = open(file, 'r')
    cam_params = {}
    for i, line in enumerate(f):
        if i == 0:
            continue
        entry = [float(x) for x in line.split()]
        id = int(entry[0])
        cam_params[id] = Camera(entry)
    return cam_params


def parse_pose(pose, timestep):
    timesteps = pose[:, :1]
    timesteps = np.around(timesteps)
    mask = (timesteps == timestep)[:, 0]
    pose_entry = pose[mask][0]
    camera = Camera(pose_entry)

    return camera


def get_camera_pose(scene_path, all_pose_dir, uv, views=1):
    npz_files = sorted(scene_path.glob("*.npz"))
    npz_file = npz_files[0]
    data = np.load(npz_file)
    all_pose_dir = Path(all_pose_dir)

    rgb_files = list(data.keys())

    timestamps = [int(rgb_file.split('.')[0]) for rgb_file in rgb_files]
    sorted_ids = np.argsort(timestamps)

    rgb_files = np.array(rgb_files)[sorted_ids]
    timestamps = np.array(timestamps)[sorted_ids]

    camera_file = all_pose_dir / (str(scene_path.name) + '.txt')
    cam_params = parse_pose_file(camera_file)
    # H, W, _ = data[rgb_files[0]].shape

    # Weird cropping of images
    H, W = 256, 456

    xscale = W / min(H, W)
    yscale = H / min(H, W)


    query = {}
    context = {}

    render_frame = min(128, rgb_files.shape[0])

    query_intrinsics = []
    query_c2w = []
    query_rgbs = []
    for i in range(1, render_frame):
        rgb = data[rgb_files[i]]
        timestep = timestamps[i]

        # rgb = cv2.resize(rgb, (W, H))
        intrinsics = unnormalize_intrinsics(cam_params[timestep].intrinsics, H, W)

        intrinsics[0, 2] = intrinsics[0, 2] / xscale
        intrinsics[1, 2] = intrinsics[1, 2] / yscale
        rgb = rgb.astype(np.float32) / 127.5 - 1

        query_intrinsics.append(intrinsics)
        query_c2w.append(cam_params[timestep].c2w_mat)
        query_rgbs.append(rgb)

    context_intrinsics = []
    context_c2w = []
    context_rgbs = []

    if views == 1:
        render_ids = [0]
    elif views == 2:
        render_ids = [0, min(len(rgb_files) - 1, 128)]
    else:
        assert False

    for i in render_ids:
        rgb = data[rgb_files[i]]
        timestep = timestamps[i]
        # print("render: ", i)
        # rgb = cv2.resize(rgb, (W, H))
        intrinsics = unnormalize_intrinsics(cam_params[timestep].intrinsics, H, W)
        intrinsics[0, 2] = intrinsics[0, 2] / xscale
        intrinsics[1, 2] = intrinsics[1, 2] / yscale

        rgb = rgb.astype(np.float32) / 127.5 - 1

        context_intrinsics.append(intrinsics)
        context_c2w.append(cam_params[timestep].c2w_mat)
        context_rgbs.append(rgb)

    query = {'rgb': torch.Tensor(query_rgbs)[None].float(),
             'cam2world': torch.Tensor(query_c2w)[None].float(),
             'intrinsics': torch.Tensor(query_intrinsics)[None].float(),
             'uv': uv.view(-1, 2)[None, None].expand(1, render_frame - 1, -1, -1)}
    ctxt = {'rgb': torch.Tensor(context_rgbs)[None].float(),
            'cam2world': torch.Tensor(context_c2w)[None].float(),
            'intrinsics': torch.Tensor(context_intrinsics)[None].float()}

    return {'query': query, 'context': ctxt}

class RealEstate10k():
    def __init__(self, img_root=None, pose_root=None,
                 num_ctxt_views=2, num_query_views=2, query_sparsity=None,imsl=256,
                 max_num_scenes=None, square_crop=True, augment=False, lpips=False, dual_view=False, val=False,n_skip=12):

        self.n_skip =n_skip[0] if type(n_skip)==type([]) else n_skip
        print(self.n_skip,"n_skip")
        self.val = val
        if img_root is None: img_root = os.path.join(os.environ['RE10K_IMG_ROOT'],"test" if val else "train")
        if pose_root is None: pose_root = os.path.join(os.environ['RE10K_POSE_ROOT'],"test" if val else "train")
        print("Loading RealEstate10k...")
        self.num_ctxt_views = num_ctxt_views
        self.num_query_views = num_query_views
        self.query_sparsity = query_sparsity
        self.dual_view = dual_view

        self.imsl=imsl

        all_im_dir = Path(img_root)
        #self.all_pose_dir = Path(pose_root)
        self.all_pose = loadmat(pose_root)
        self.lpips = lpips

        self.all_scenes = sorted(all_im_dir.glob('*/'))

        dummy_img_path = str(next(self.all_scenes[0].glob("*.npz")))

        if max_num_scenes:
            self.all_scenes = list(self.all_scenes)[:max_num_scenes]

        data = np.load(dummy_img_path)
        key = list(data.keys())[0]
        im = data[key]

        H, W = im.shape[:2]
        H, W = 256, 455
        self.H, self.W = H, W
        self.augment = augment

        self.square_crop = square_crop
        # Downsample to be 256 x 256 image
        # self.H, self.W = 256, 455

        xscale = W / min(H, W)
        yscale = H / min(H, W)

        dim = min(H, W)

        self.xscale = xscale
        self.yscale = yscale

        # For now the images are already square cropped
        self.H = 256
        self.W = 455

        print(f"Resolution is {H}, {W}.")

        if self.square_crop:
            i, j = torch.meshgrid(torch.arange(0, self.imsl), torch.arange(0, self.imsl))
        else:
            i, j = torch.meshgrid(torch.arange(0, W), torch.arange(0, H))

        self.uv = torch.stack([i.float(), j.float()], dim=-1).permute(1, 0, 2)

        # if self.square_crop:
        #     self.uv = data_util.square_crop_img(self.uv)

        self.uv = self.uv[None].permute(0, -1, 1, 2).permute(0, 2, 3, 1)
        self.uv = self.uv.reshape(-1, 2)

        self.scene_path_list = list(Path(img_root).glob("*/"))

    def __len__(self):
        return len(self.all_scenes)

    def __getitem__(self, idx,scene_query=None):
        idx = idx if not_of else 0
        scene_path = self.all_scenes[idx if scene_query is None else scene_query]
        npz_files = sorted(scene_path.glob("*.npz"))

        name = scene_path.name

        def get_another():
            if self.val:
                return self[idx-1 if idx >200  else idx+1]
            return self.__getitem__(random.randint(0, len(self.all_scenes) - 1))

        if name not in self.all_pose: return get_another()

        pose = self.all_pose[name]

        if len(npz_files) == 0:
            print("npz get another")
            return get_another()

        npz_file = npz_files[0]
        try:
            data = np.load(npz_file)
        except:
            print("npz load error get another")
            return get_another()

        rgb_files = list(data.keys())
        window_size = 128

        if len(rgb_files) <= 20:
            print("<20 rgbs error get another")
            return get_another()

        timestamps = [int(rgb_file.split('.')[0]) for rgb_file in rgb_files]
        sorted_ids = np.argsort(timestamps)

        rgb_files = np.array(rgb_files)[sorted_ids]
        timestamps = np.array(timestamps)[sorted_ids]

        assert (timestamps == sorted(timestamps)).all()
        num_frames = len(rgb_files)
        left_bound = 0
        right_bound = num_frames - 1
        candidate_ids = np.arange(left_bound, right_bound)

        # remove windows between frame -32 to 32
        nframe = 1
        nframe_view = 140 if self.val else 92

        id_feats = []

        n_skip=self.n_skip

        id_feat = np.array(id_feats)
        low = 0
        high = num_frames-1-n_skip*self.num_query_views

        if high <= low:
            n_skip = int(num_frames//(self.num_query_views+1))
            high = num_frames-1-n_skip*self.num_query_views
            print("high<low ",num_frames,n_skip,high)

        if self.val: base_i=0
        else: base_i=np.random.randint(low=low, high=high)

        id_render = [base_i+i*n_skip for i in range(self.num_query_views)]

        query_rgbs = []
        query_intrinsics = []
        query_c2w = []
        uvs = []

        for id in id_render:
            rgb_file = rgb_files[id]
            rgb = data[rgb_file]

            if rgb.shape[0] == 360:
                rgb = cv2.resize(rgb, (self.W, self.H))

            if self.square_crop:
                rgb = square_crop_img(rgb)

            cam_param = parse_pose(pose, timestamps[id])

            intrinsics = unnormalize_intrinsics(cam_param.intrinsics, self.H, self.W)

            if self.square_crop:
                intrinsics[0, 2] = intrinsics[0, 2] / self.xscale
                intrinsics[1, 2] = intrinsics[1, 2] / self.yscale

            if self.augment:
                rgb, intrinsics, cam_param.c2w_mat = augment(rgb, intrinsics, cam_param.c2w_mat)

            rgb = rgb.astype(np.float32) / 127.5 - 1
            img_size = rgb.shape[:2]
            rgb = rgb.reshape((-1, 3))

            mask_lpips = 0.0

            uv = self.uv
            uvs.append(uv)
            query_rgbs.append(rgb)
            query_intrinsics.append(intrinsics)
            query_c2w.append(cam_param.c2w_mat)


        uvs = torch.Tensor(np.stack(uvs, axis=0)).float()
        ctxt_rgbs = []
        ctxt_intrinsics = []
        ctxt_c2w = []

        for id in id_feat:
            rgb_file = rgb_files[id]
            rgb = data[rgb_file]

            # rgb = cv2.resize(rgb, (self.W, self.H))
            if rgb.shape[0] == 360:
                rgb = cv2.resize(rgb, (self.W, self.H))

            if self.square_crop:
                rgb = square_crop_img(rgb)

            cam_param = parse_pose(pose, timestamps[id])

            intrinsics = unnormalize_intrinsics(cam_param.intrinsics, self.H, self.W)

            if self.square_crop:
                intrinsics[0, 2] = intrinsics[0, 2] / self.xscale
                intrinsics[1, 2] = intrinsics[1, 2] / self.yscale

            if self.augment:
                rgb, intrinsics, cam_param.c2w_mat = augment(rgb, intrinsics, cam_param.c2w_mat)

            rgb = rgb.astype(np.float32) / 127.5 - 1

            ctxt_rgbs.append(rgb)
            ctxt_intrinsics.append(intrinsics)
            ctxt_c2w.append(cam_param.c2w_mat)

        query_rgbs = np.stack(query_rgbs)
        query_intrinsics = np.stack(query_intrinsics)
        query_c2w = torch.from_numpy(np.stack(query_c2w)).float()

        # make intrinsics/uv resolution indepdent
        uvs = uvs/(self.imsl-1)
        query_intrinsics[:,:2,:-1] = query_intrinsics[:,:2,:-1]/256

        imgs = torch.from_numpy(query_rgbs).permute(0,2,1).unflatten(-1,(256,256))
        imgs_large = (imgs*.5+.5) * 255
        imgs = F.interpolate(imgs,(self.imsl,self.imsl))
        imgs_flat = imgs.permute(0,2,3,1).flatten(1,2)
        Ks = torch.from_numpy(query_intrinsics)[:,:3,:3].float()

        uv = uvs.float()

        model_input = {
                "trgt_rgb": imgs[1:],
                "ctxt_rgb": imgs[:-1],
                "trgt_rgb_large": imgs_large[1:],
                "ctxt_rgb_large": imgs_large[:-1],
                "trgt_rgb_med": imgs_large[1:],
                "ctxt_rgb_med": imgs_large[:-1],
                "intrinsics": Ks[1:],
                "trgt_c2w": query_c2w[1:],
                "ctxt_c2w": query_c2w[:-1],
                "x_pix": uv[1:],
                }
        gt = {
                "trgt_rgb": imgs_flat[1:]*.5+.5,
                "ctxt_rgb": imgs_flat[:-1]*.5+.5,
                "intrinsics": Ks[1:],
                "x_pix": uv[1:],
                }
        return model_input,gt
