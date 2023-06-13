"""Volume rendering code."""
from geometry import *
from typing import Callable, List, Optional, Tuple, Generator, Dict
import torch
from torch import nn
from torch.nn import functional as F
import numpy as np
from torch import Tensor,device


def pdf_z_values(
    bins: Tensor,
    weights: Tensor,
    samples: int,
    d: device,
    perturb: bool,
) -> Tensor:
    """Generate z-values from pdf
    Arguments:
        bins (Tensor): z-value bins (B, N - 2)
        weights (Tensor): bin weights gathered from first pass (B, N - 1)
        samples (int): number of samples N
        d (device): torch device
        perturb (bool): peturb ray query segment
    Returns:
        t (Tensor): z-values sampled from pdf (B, N)
    """
    EPS = 1e-5
    B, N = weights.size()

    weights = weights + EPS
    pdf = weights / torch.sum(weights, dim=-1, keepdim=True)
    cdf = torch.cumsum(pdf, dim=-1)
    cdf = torch.cat((torch.zeros_like(cdf[:, :1]), cdf), dim=-1)

    if perturb:
        u = torch.rand((B, samples), device=d)
        u = u.contiguous()
    else:
        u = torch.linspace(0, 1, samples, device=d)
        u = u.expand(B, samples)
        u = u.contiguous()

    idxs = torch.searchsorted(cdf, u, right=True)
    idxs_below = torch.clamp_min(idxs - 1, 0)
    idxs_above = torch.clamp_max(idxs, N)
    idxs = torch.stack((idxs_below, idxs_above), dim=-1).view(B, 2 * samples)

    cdf = torch.gather(cdf, dim=1, index=idxs).view(B, samples, 2)
    bins = torch.gather(bins, dim=1, index=idxs).view(B, samples, 2)

    den = cdf[:, :, 1] - cdf[:, :, 0]
    den[den < EPS] = 1.0

    t = (u - cdf[:, :, 0]) / den
    t = bins[:, :, 0] + t * (bins[:, :, 1] - bins[:, :, 0])

    return t


def sample_pdf(bins, weights, N_importance, det=False, eps=1e-5):
    """
    Sample @N_importance samples from @bins with distribution defined by @weights.
    Inputs:
        bins: (N_rays, N_samples_+1) where N_samples_ is "the number of coarse samples per ray - 2"
        weights: (N_rays, N_samples_)
        N_importance: the number of samples to draw from the distribution
        det: deterministic or not
        eps: a small number to prevent division by zero
    Outputs:
        samples: the sampled samples
    Source: https://github.com/kwea123/nerf_pl/blob/master/models/rendering.py
    """
    N_rays, N_samples_ = weights.shape
    weights = weights + eps  # prevent division by zero (don't do inplace op!)
    pdf = weights / torch.sum(weights, -1, keepdim=True)  # (N_rays, N_samples_)
    cdf = torch.cumsum(pdf, -1)  # (N_rays, N_samples), cumulative distribution function
    cdf = torch.cat([torch.zeros_like(cdf[:, :1]), cdf], -1)  # (N_rays, N_samples_+1)
    # padded to 0~1 inclusive

    if det:
        u = torch.linspace(0, 1, N_importance, device=bins.device)
        u = u.expand(N_rays, N_importance)
    else:
        u = torch.rand(N_rays, N_importance, device=bins.device)
    u = u.contiguous()

    inds = torch.searchsorted(cdf, u)
    below = torch.clamp_min(inds - 1, 0)
    above = torch.clamp_max(inds, N_samples_)

    inds_sampled = torch.stack([below, above], -1).view(N_rays, 2 * N_importance)
    cdf_g = torch.gather(cdf, 1, inds_sampled)
    cdf_g = cdf_g.view(N_rays, N_importance, 2)
    bins_g = torch.gather(bins, 1, inds_sampled).view(N_rays, N_importance, 2)

    denom = cdf_g[..., 1] - cdf_g[..., 0]
    denom[
        denom < eps
    ] = 1  # denom equals 0 means a bin has weight 0, in which case it will not be sampled
    # anyway, therefore any value for it is fine (set to 1 here)

    samples = bins_g[..., 0] + (u - cdf_g[..., 0]) / denom * (
        bins_g[..., 1] - bins_g[..., 0]
    )
    return samples


def pdf_rays(
    ro: Tensor,
    rd: Tensor,
    t: Tensor,
    weights: Tensor,
    samples: int,
    perturb: bool,
) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
    """Sample pdf along rays given computed weights
    Arguments:
        ro (Tensor): rays origin (B, 3)
        rd (Tensor): rays direction (B, 3)
        t (Tensor): coarse z-value (B, N)
        weights (Tensor): weights gathered from first pass (B, N)
        samples (int): number of samples along the ray
        perturb (bool): peturb ray query segment
    Returns:
        rx (Tensor): rays position queries (B, Nc + Nf, 3)
        rd (Tensor): rays direction (B, Nc + Nf, 3)
        t (Tensor): z-values from near to far (B, Nc + Nf)
        delta (Tensor): rays segment lengths (B, Nc + Nf)
    """
    B, S, N_coarse, _ = weights.shape
    weights = rearrange(weights, "b n s 1 ->  (b n) s")
    t = rearrange(t, "b n s 1 -> (b n) s")

    Nf = samples
    tm = 0.5 * (t[:, :-1] + t[:, 1:])
    t_pdf = sample_pdf(tm, weights[..., 1:-1], Nf, det=False).detach().view(B, S, Nf)
    rx = ro[..., None, :] + rd[..., None, :] * t_pdf[..., None]

    return rx, t_pdf


def sample_points_along_rays(
    near_depth: float,
    far_depth: float,
    num_samples: int,
    ray_origins: torch.Tensor,
    ray_directions: torch.Tensor,
    device: torch.device,
    logspace=False,
    perturb=False,
):
    # Compute a linspace of num_samples depth values beetween near_depth and far_depth.
    if logspace:
        z_vals = torch.logspace(np.log10(near_depth), np.log10(far_depth), num_samples, device=device)
    else:
        z_vals = torch.linspace(near_depth, far_depth, num_samples, device=device)

    if perturb:
        z_vals = z_vals + .5*(torch.rand_like(z_vals)-.5)*torch.cat([(z_vals[:-1]-z_vals[1:]).abs(),torch.tensor([0]).cuda()])

    # Using the ray_origins, ray_directions, generate 3D points along
    # the camera rays according to the z_vals.
    pts = (
        ray_origins[..., None, :] + ray_directions[..., None, :] * z_vals[..., :, None]
    )

    return pts, z_vals


def volume_integral(
    z_vals: torch.tensor, sigmas: torch.tensor, radiances: torch.tensor, white_back=False,dist_scale=True,
) -> Tuple[torch.tensor, torch.tensor, torch.tensor]:
    # Compute the deltas in depth between the points.
    dists = torch.cat(
        [
            z_vals[..., 1:] - z_vals[..., :-1],
            (z_vals[..., 1:] - z_vals[..., :-1])[..., -1:],
        ],
        -1,
    )

    # Compute the alpha values from the densities and the dists.
    # Tip: use torch.einsum for a convenient way of multiplying the correct
    # dimensions of the sigmas and the dists.
    # TODO switch to just expanding shape of dists for less code
    dist_scaling=dists if dist_scale else torch.ones_like(dists)
    if len(dists.shape)==1: alpha = 1.0 - torch.exp(-torch.einsum("brzs, z -> brzs", F.relu(sigmas), dist_scaling))
    else: alpha = 1.0 - torch.exp(-torch.einsum("brzs, brz -> brzs", F.relu(sigmas), dist_scaling.flatten(0,1)))

    alpha_shifted = torch.cat(
        [torch.ones_like(alpha[:, :, :1]), 1.0 - alpha + 1e-10], -2
    )

    # Compute the Ts from the alpha values. Use torch.cumprod.
    Ts = torch.cumprod(alpha_shifted, -2)

    # Compute the weights from the Ts and the alphas.
    weights = alpha * Ts[..., :-1, :]
    
    # Compute the pixel color as the weighted sum of the radiance values.
    rgb = torch.einsum("brzs, brzs -> brs", weights, radiances)

    # Compute the depths as the weighted sum of z_vals.
    # Tip: use torch.einsum for a convenient way of computing the weighted sum,
    # without the need to reshape the z_vals.
    if len(dists.shape)==1:
        depth_map = torch.einsum("brzs, z -> brs", weights, z_vals)
    else:
        depth_map = torch.einsum("brzs, brz -> brs", weights, z_vals.flatten(0,1))

    if white_back:
        accum = weights.sum(dim=-2)
        backgrd_color = torch.tensor([1,1,1]+[0]*(rgb.size(-1)-3)).broadcast_to(rgb.shape).to(rgb)
        #backgrd_color = torch.ones(rgb.size(-1)).broadcast_to(rgb.shape).to(rgb)
        rgb = rgb + (backgrd_color - accum)

    return rgb, depth_map, weights


class VolumeRenderer(nn.Module):
    def __init__(self, near, far, n_samples=32, backgrd_color=None):
        super().__init__()
        self.near = near
        self.far = far
        self.n_samples = n_samples

        if backgrd_color is not None:
            self.register_buffer('backgrd_color', backgrd_color)
        else:
            self.backgrd_color = None

    def forward(
        self, cam2world, intrinsics, x_pix, radiance_field: nn.Module
    ) -> Tuple[torch.tensor, torch.tensor]:
        """
        Takes as inputs ray origins and directions - samples points along the
        rays and then calculates the volume rendering integral.

        Params:
            input_dict: Dictionary with keys 'cam2world', 'intrinsics', and 'x_pix'
            radiance_field: nn.Module instance of the radiance field we want to render.

        Returns:
            Tuple of rgb, depth_map
            rgb: for each pixel coordinate x_pix, the color of the respective ray.
            depth_map: for each pixel coordinate x_pix, the depth of the respective ray.

        """
        batch_size, num_rays = x_pix.shape[0], x_pix.shape[1]

        # Compute the ray directions in world coordinates.
        # Use the function get_world_rays.
        ros, rds = get_world_rays(x_pix, intrinsics, cam2world)

        # Generate the points along rays and their depth values
        # Use the function sample_points_along_rays.
        pts, z_vals = sample_points_along_rays(
            self.near, self.far, self.n_samples, ros, rds, device=x_pix.device
        )

        # Reshape pts to (batch_size, -1, 3).
        pts = pts.reshape(batch_size, -1, 3)

        # Sample the radiance field with the points along the rays.
        sigma, feats, misc = radiance_field(pts)

        # Reshape sigma and feats back to (batch_size, num_rays, self.n_samples, -1)
        sigma = sigma.view(batch_size, num_rays, self.n_samples, 1)
        feats = feats.view(batch_size, num_rays, self.n_samples, -1)

        # Compute pixel colors, depths, and weights via the volume integral.
        rendering, depth_map, weights = volume_integral(z_vals, sigma, feats)

        if self.backgrd_color is not None:
            accum = weights.sum(dim=-2)
            backgrd_color = self.backgrd_color.broadcast_to(rendering.shape)
            rendering = rendering + (backgrd_color - accum)

        return rendering, depth_map, misc
