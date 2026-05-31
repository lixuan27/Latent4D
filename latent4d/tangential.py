"""Algorithm 3: tangential refinement via frozen visual features.

The Newton advection of Algorithm 1 moves every vertex along the surface normal,
so it recovers the on-surface constraint but not the in-surface (tangential)
component of the material motion. We recover it without training: for each vertex
we search a small grid of offsets inside the local tangent plane and pick the one
whose projected DINOv2 feature best matches the vertex's anchor-frame feature
(Eq. 11), then re-project once onto the surface.
"""
from __future__ import annotations
import torch


def _tangent_basis(normal):
    """Two orthonormal vectors spanning the plane orthogonal to `normal` [N,3]."""
    n = torch.nn.functional.normalize(normal, dim=-1)
    helper = torch.zeros_like(n)
    helper[..., 0] = 1.0
    mask = (n[..., 0].abs() > 0.9)
    helper[mask, 0] = 0.0
    helper[mask, 1] = 1.0
    u = torch.nn.functional.normalize(torch.cross(n, helper, dim=-1), dim=-1)
    v = torch.cross(n, u, dim=-1)
    return u, v


@torch.no_grad()
def refine(decoder, z_t, x, anchor_feat, image_t, feature_at, project_fn,
           rho=0.02, grid=5):
    """Refine advected positions x [N,3] in their tangent planes (Eq. 11).

    decoder      : DecoderAdapter, for normals (grad of s) and the final re-projection.
    anchor_feat  : [N, D] per-vertex features sampled at the anchor frame.
    feature_at   : callable (image, pixel_uv[N,2]) -> [N, D] features at frame t.
    project_fn   : camera projection R^3 -> pixel uv.
    rho, grid    : tangent search radius and per-axis grid resolution.
    """
    s, g = decoder.sdf_and_grad(x, z_t)
    u, v = _tangent_basis(g)
    steps = torch.linspace(-rho, rho, grid, device=x.device)
    best = x.clone()
    best_score = torch.full((x.shape[0],), -1e9, device=x.device)
    for du in steps:
        for dv in steps:
            cand = x + du * u + dv * v
            feat = feature_at(image_t, project_fn(cand))           # [N, D]
            score = (feat * anchor_feat).sum(-1)
            upd = score > best_score
            best[upd] = cand[upd]
            best_score[upd] = score[upd]
    # one normal-projection step to restore s(x) = 0
    s, g = decoder.sdf_and_grad(best, z_t)
    denom = (g * g).sum(-1, keepdim=True) + 1e-6
    best = best - (s.unsqueeze(-1) * g) / denom
    return best
