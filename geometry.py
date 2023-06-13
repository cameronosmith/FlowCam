"""Multi-view geometry & proejction code.."""
import torch
from einops import rearrange, repeat
from torch.nn import functional as F
import numpy as np

def d6_to_rotmat(d6):
    a1, a2 = d6[..., :3], d6[..., 3:]
    b1 = F.normalize(a1, dim=-1)
    b2 = a2 - (b1 * a2).sum(-1, keepdim=True) * b1
    b2 = F.normalize(b2, dim=-1)
    b3 = torch.cross(b1, b2, dim=-1)
    return torch.stack((b1, b2, b3), dim=-2)

def time_interp_poses(pose_inp,time_i,n_trgt,eye_pts):
    i,j = max(0,int(time_i*(n_trgt-1))-1),int(time_i*(n_trgt-1))
    pose_interp = camera_interp(*pose_inp[:,[i,j]].unbind(1),time_i)
    if i==j: pose_interp=pose_inp[:,0]
    pose_interp = repeat(pose_interp,"b x y -> b trgt x y",trgt=n_trgt)
    return pose_interp
    eye_pts = torch.cat((eye_pts,torch.ones_like(eye_pts[...,[0]])),-1)
    query_pts = torch.einsum("bcij,bcdkj->bcdki",pose_interp,eye_pts)[...,:3]
    return query_pts

def pixel_aligned_features(
    coords_3d_world, cam2world, intrinsics, img_features, interp="bilinear",padding_mode="border",
):
    # Args:
    #     coords_3d_world: shape (b, n, 3)
    #     cam2world: camera pose of shape (..., 4, 4)

    # project 3d points to 2D
    c3d_world_hom = homogenize_points(coords_3d_world)
    c3d_cam_hom = transform_world2cam(c3d_world_hom, cam2world)
    c2d_cam, depth = project(c3d_cam_hom, intrinsics.unsqueeze(1))

    # now between 0 and 1. Map to -1 and 1
    c2d_norm = (c2d_cam - 0.5) * 2
    c2d_norm = rearrange(c2d_norm, "b n ch -> b n () ch")
    c2d_norm = c2d_norm[..., :2]

    # grid_sample
    feats = F.grid_sample(
        img_features, c2d_norm, align_corners=True, padding_mode=padding_mode, mode=interp
    )
    feats = feats.squeeze(-1)  # b ch n

    feats = rearrange(feats, "b ch n -> b n ch")
    return feats, c3d_cam_hom[..., :3], c2d_cam

# https://gist.github.com/mkocabas/54ea2ff3b03260e3fedf8ad22536f427
def procrustes(S1, S2,weights=None):

    if len(S1.shape)==4:
        out = procrustes(S1.flatten(0,1),S2.flatten(0,1),weights.flatten(0,1) if weights is not None else None)
        return out[0],out[1].unflatten(0,S1.shape[:2])
    '''
    Computes a similarity transform (sR, t) that takes
    a set of 3D points S1 (BxNx3) closest to a set of 3D points, S2,
    where R is an 3x3 rotation matrix, t 3x1 translation, s scale. / mod : assuming scale is 1
    i.e. solves the orthogonal Procrutes problem.
    '''
    with torch.autocast(device_type='cuda', dtype=torch.float32):
        S1 = S1.permute(0,2,1)
        S2 = S2.permute(0,2,1)
        if weights is not None:
            weights=weights.permute(0,2,1)
        transposed = True

        if weights is None:
            weights = torch.ones_like(S1[:,:1])

        # 1. Remove mean.
        weights_norm = weights/(weights.sum(-1,keepdim=True)+1e-6)
        mu1 = (S1*weights_norm).sum(2,keepdim=True)
        mu2 = (S2*weights_norm).sum(2,keepdim=True)

        X1 = S1 - mu1
        X2 = S2 - mu2

        diags = torch.stack([torch.diag(w.squeeze(0)) for w in weights]) # does batched version exist?

        # 3. The outer product of X1 and X2.
        K = (X1@diags).bmm(X2.permute(0,2,1))

        # 4. Solution that Maximizes trace(R'K) is R=U*V', where U, V are singular vectors of K.
        U, s, V = torch.svd(K)

        # Construct Z that fixes the orientation of R to get det(R)=1.
        Z = torch.eye(U.shape[1], device=S1.device).unsqueeze(0)
        Z = Z.repeat(U.shape[0],1,1)
        Z[:,-1, -1] *= torch.sign(torch.det(U.bmm(V.permute(0,2,1))))

        # Construct R.
        R = V.bmm(Z.bmm(U.permute(0,2,1)))

        # 6. Recover translation.
        t = mu2 - ((R.bmm(mu1)))

        # 7. Error:
        S1_hat = R.bmm(S1) + t

        # Combine recovered transformation as single matrix
        R_=torch.eye(4)[None].expand(S1.size(0),-1,-1).to(S1)
        R_[:,:3,:3]=R
        T_=torch.eye(4)[None].expand(S1.size(0),-1,-1).to(S1)
        T_[:,:3,-1]=t.squeeze(-1)
        S_=torch.eye(4)[None].expand(S1.size(0),-1,-1).to(S1)
        transf = T_@S_@R_

        return (S1_hat-S2).square().mean(),transf

def symmetric_orthogonalization(x):
    # https://github.com/amakadia/svd_for_pose
    m = x.view(-1, 3, 3).type(torch.float)
    u, s, v = torch.svd(m)
    vt = torch.transpose(v, 1, 2)
    det = torch.det(torch.matmul(u, vt))
    det = det.view(-1, 1, 1)
    vt = torch.cat((vt[:, :2, :], vt[:, -1:, :] * det), 1)
    r = torch.matmul(u, vt)
    return r

def rigidity_loss(ctx_xyz,trgt_xyz):

    x_points = ctx_xyz #.view(-1, 3)
    y_points = trgt_xyz #.view(-1, 3)

    x_mean = x_points.mean(1, keepdim=True)     # x_mean and y_mean define the global translation
    y_mean = y_points.mean(1, keepdim=True)

    x_points_centered = x_points - x_mean
    y_points_centered = y_points - y_mean

    x_scale = torch.sqrt(x_points_centered.pow(2).sum(2, keepdim=True)).mean(1, keepdim=True)
    x_points_normalized = x_points_centered / x_scale          # x_scale and y_scale define the global scales

    y_scale = torch.sqrt(y_points_centered.pow(2).sum(2, keepdim=True)).mean(1, keepdim=True)
    y_points_normalized = y_points_centered / y_scale

    M = torch.einsum('b i k, b i j -> b k j', x_points_normalized, y_points_normalized)  # M is the covariance matrix
    R = symmetric_orthogonalization(M)   #this is the rotation matrix

    # Compute the transformed ctxt points
    x_points_transformed = torch.matmul(x_points_normalized, R)

    loss = (x_points_transformed - y_points_normalized).pow(2).mean()
    return loss


def homogenize_points(points: torch.Tensor):
    """Appends a "1" to the coordinates of a (batch of) points of dimension DIM.

    Args:
        points: points of shape (..., DIM)

    Returns:
        points_hom: points with appended "1" dimension.
    """
    ones = torch.ones_like(points[..., :1], device=points.device)
    return torch.cat((points, ones), dim=-1)


def homogenize_vecs(vectors: torch.Tensor):
    """Appends a "0" to the coordinates of a (batch of) vectors of dimension DIM.

    Args:
        vectors: vectors of shape (..., DIM)

    Returns:
        vectors_hom: points with appended "0" dimension.
    """
    zeros = torch.zeros_like(vectors[..., :1], device=vectors.device)
    return torch.cat((vectors, zeros), dim=-1)


def unproject(
    xy_pix: torch.Tensor, z: torch.Tensor, intrinsics: torch.Tensor
) -> torch.Tensor:
    """Unproject (lift) 2D pixel coordinates x_pix and per-pixel z coordinate
    to 3D points in camera coordinates.

    Args:
        xy_pix: 2D pixel coordinates of shape (..., 2)
        z: per-pixel depth, defined as z coordinate of shape (..., 1)
        intrinscis: camera intrinscics of shape (..., 3, 3)

    Returns:
        xyz_cam: points in 3D camera coordinates.
    """
    xy_pix_hom = homogenize_points(xy_pix)
    xyz_cam = torch.einsum("...ij,...kj->...ki", intrinsics.inverse(), xy_pix_hom)
    xyz_cam *= z
    return xyz_cam


def transform_world2cam(
    xyz_world_hom: torch.Tensor, cam2world: torch.Tensor
) -> torch.Tensor:
    """Transforms points from 3D world coordinates to 3D camera coordinates.

    Args:
        xyz_world_hom: homogenized 3D points of shape (..., 4)
        cam2world: camera pose of shape (..., 4, 4)

    Returns:
        xyz_cam: points in camera coordinates.
    """
    world2cam = torch.inverse(cam2world)
    return transform_rigid(xyz_world_hom, world2cam)


def transform_cam2world(
    xyz_cam_hom: torch.Tensor, cam2world: torch.Tensor
) -> torch.Tensor:
    """Transforms points from 3D world coordinates to 3D camera coordinates.

    Args:
        xyz_cam_hom: homogenized 3D points of shape (..., 4)
        cam2world: camera pose of shape (..., 4, 4)

    Returns:
        xyz_world: points in camera coordinates.
    """
    return transform_rigid(xyz_cam_hom, cam2world)


def transform_rigid(xyz_hom: torch.Tensor, T: torch.Tensor) -> torch.Tensor:
    """Apply a rigid-body transform to a (batch of) points / vectors.

    Args:
        xyz_hom: homogenized 3D points of shape (..., 4)
        T: rigid-body transform matrix of shape (..., 4, 4)

    Returns:
        xyz_trans: transformed points.
    """
    return torch.einsum("...ij,...kj->...ki", T, xyz_hom)


def get_unnormalized_cam_ray_directions(
    xy_pix: torch.Tensor, intrinsics: torch.Tensor
) -> torch.Tensor:
    return unproject(
        xy_pix,
        torch.ones_like(xy_pix[..., :1], device=xy_pix.device),
        intrinsics=intrinsics,
    )


def get_world_rays_(
    xy_pix: torch.Tensor,
    intrinsics: torch.Tensor,
    cam2world: torch.Tensor,
) -> torch.Tensor:

    if cam2world is None: 
        cam2world = torch.eye(4)[None].expand(xy_pix.size(0),-1,-1).to(xy_pix)

    # Get camera origin of camera 1
    cam_origin_world = cam2world[..., :3, -1]

    # Get ray directions in cam coordinates
    ray_dirs_cam = get_unnormalized_cam_ray_directions(xy_pix, intrinsics)
    ray_dirs_cam = ray_dirs_cam / ray_dirs_cam.norm(dim=-1, keepdim=True)

    # Homogenize ray directions
    rd_cam_hom = homogenize_vecs(ray_dirs_cam)

    # Transform ray directions to world coordinates
    rd_world_hom = transform_cam2world(rd_cam_hom, cam2world)

    # Tile the ray origins to have the same shape as the ray directions.
    # Currently, ray origins have shape (batch, 3), while ray directions have shape
    cam_origin_world = repeat(
        cam_origin_world, "b ch -> b num_rays ch", num_rays=ray_dirs_cam.shape[1]
    )

    # Return tuple of cam_origins, ray_world_directions
    return cam_origin_world, rd_world_hom[..., :3]

def get_world_rays(
    xy_pix: torch.Tensor,
    intrinsics: torch.Tensor,
    cam2world: torch.Tensor,
) -> torch.Tensor:
    if len(xy_pix.shape)==4:
        out = get_world_rays_(xy_pix.flatten(0,1),intrinsics.flatten(0,1),cam2world.flatten(0,1) if cam2world is not None else None)
        return [x.unflatten(0,xy_pix.shape[:2]) for x in out]
    return get_world_rays_(xy_pix,intrinsics,cam2world)




def get_opencv_pixel_coordinates(
    y_resolution: int,
    x_resolution: int,
    device='cpu'
):
    """For an image with y_resolution and x_resolution, return a tensor of pixel coordinates
    normalized to lie in [0, 1], with the origin (0, 0) in the top left corner,
    the x-axis pointing right, the y-axis pointing down, and the bottom right corner
    being at (1, 1).

    Returns:
        xy_pix: a meshgrid of values from [0, 1] of shape
                (y_resolution, x_resolution, 2)
    """
    i, j = torch.meshgrid(
        torch.linspace(0, 1, steps=x_resolution, device=device),
        torch.linspace(0, 1, steps=y_resolution, device=device),
    )

    xy_pix = torch.stack([i.float(), j.float()], dim=-1).permute(1, 0, 2)
    return xy_pix


def project(xyz_cam_hom: torch.Tensor, intrinsics: torch.Tensor) -> torch.Tensor:
    """Projects homogenized 3D points xyz_cam_hom in camera coordinates
    to pixel coordinates.

    Args:
        xyz_cam_hom: 3D points of shape (..., 4)
        intrinsics: camera intrinscics of shape (..., 3, 3)

    Returns:
        xy: homogeneous pixel coordinates of shape (..., 3) (final coordinate is 1)
    """
    if len(intrinsics.shape)==len(xyz_cam_hom.shape): intrinsics=intrinsics.unsqueeze(1)
    xyw = torch.einsum("...ij,...j->...i", intrinsics, xyz_cam_hom[..., :3])
    z = xyw[..., -1:]
    xyw = xyw / (z + 1e-5)  # z-divide
    return xyw[..., :2], z

def _sqrt_positive_part(x: torch.Tensor) -> torch.Tensor:
    # from pytorch3d
    """
    Returns torch.sqrt(torch.max(0, x))
    but with a zero subgradient where x is 0.
    """
    ret = torch.zeros_like(x)
    positive_mask = x > 0
    ret[positive_mask] = torch.sqrt(x[positive_mask])
    return ret
def matrix_to_quaternion(matrix: torch.Tensor) -> torch.Tensor:
    # from pytorch3d
    """
    Convert rotations given as rotation matrices to quaternions.
    Args:
        matrix: Rotation matrices as tensor of shape (..., 3, 3).
    Returns:
        quaternions with real part first, as tensor of shape (..., 4).
    """
    if matrix.size(-1) != 3 or matrix.size(-2) != 3:
        raise ValueError(f"Invalid rotation matrix shape {matrix.shape}.")

    batch_dim = matrix.shape[:-2]
    m00, m01, m02, m10, m11, m12, m20, m21, m22 = torch.unbind(
        matrix.reshape(batch_dim + (9,)), dim=-1
    )

    q_abs = _sqrt_positive_part(
        torch.stack(
            [
                1.0 + m00 + m11 + m22,
                1.0 + m00 - m11 - m22,
                1.0 - m00 + m11 - m22,
                1.0 - m00 - m11 + m22,
            ],
            dim=-1,
        )
    )

    # we produce the desired quaternion multiplied by each of r, i, j, k
    quat_by_rijk = torch.stack(
        [
            torch.stack([q_abs[..., 0] ** 2, m21 - m12, m02 - m20, m10 - m01], dim=-1),
            torch.stack([m21 - m12, q_abs[..., 1] ** 2, m10 + m01, m02 + m20], dim=-1),
            torch.stack([m02 - m20, m10 + m01, q_abs[..., 2] ** 2, m12 + m21], dim=-1),
            torch.stack([m10 - m01, m20 + m02, m21 + m12, q_abs[..., 3] ** 2], dim=-1),
        ],
        dim=-2,
    )

    # We floor here at 0.1 but the exact level is not important; if q_abs is small,
    # the candidate won't be picked.
    flr = torch.tensor(0.1).to(dtype=q_abs.dtype, device=q_abs.device)
    quat_candidates = quat_by_rijk / (2.0 * q_abs[..., None].max(flr))

    # if not for numerical problems, quat_candidates[i] should be same (up to a sign),
    # forall i; we pick the best-conditioned one (with the largest denominator)

    return quat_candidates[
        F.one_hot(q_abs.argmax(dim=-1), num_classes=4) > 0.5, :
    ].reshape(batch_dim + (4,))
def quaternion_to_matrix(quaternions: torch.Tensor) -> torch.Tensor:
    # from pytorch3d
    """
    Convert rotations given as quaternions to rotation matrices.
    Args:
        quaternions: quaternions with real part first,
            as tensor of shape (..., 4).
    Returns:
        Rotation matrices as tensor of shape (..., 3, 3).
    """
    r, i, j, k = torch.unbind(quaternions, -1)
    two_s = 2.0 / (quaternions * quaternions).sum(-1)

    o = torch.stack(
        (
            1 - two_s * (j * j + k * k),
            two_s * (i * j - k * r),
            two_s * (i * k + j * r),
            two_s * (i * j + k * r),
            1 - two_s * (i * i + k * k),
            two_s * (j * k - i * r),
            two_s * (i * k - j * r),
            two_s * (j * k + i * r),
            1 - two_s * (i * i + j * j),
        ),
        -1,
    )
    return o.reshape(quaternions.shape[:-1] + (3, 3))
def camera_interp(camera1, camera2, t):
    if len(camera1.shape)==3:
        return torch.stack([camera_interp(cam1,cam2,t) for cam1,cam2 in zip(camera1,camera2)])
    # Extract the rotation component from the camera matrices
    q1 = matrix_to_quaternion(camera1[:3, :3])
    q2 = matrix_to_quaternion(camera2[:3, :3])

    # todo add negative quaternion check to not go long way around

    # Interpolate the quaternions using slerp
    cos_angle = (q1 * q2).sum(dim=0)
    angle = torch.acos(cos_angle.clamp(-1, 1))
    q_interpolated = (q1 * torch.sin((1 - t) * angle) + q2 * torch.sin(t * angle)) / torch.sin(angle)
    rotation_interpolated = quaternion_to_matrix(q_interpolated)

    # Interpolate the translation component
    translation_interpolated = torch.lerp(camera1[:3,-1], camera2[:3,-1], t)

    cam_interpolated = torch.eye(4)
    cam_interpolated[:3,:3]=rotation_interpolated
    cam_interpolated[:3,-1]=translation_interpolated

    return cam_interpolated.cuda()
