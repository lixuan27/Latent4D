"""Rendering helpers: anchor-frame vertex-identity colour, shaded mesh frames,
and multi-camera silhouettes for the mIoU metric. Uses PyTorch3D when available.

The vertex-identity colouring is what makes the cross-frame correspondence
visible in every figure and video: each vertex gets a fixed colour from its
anchor-frame position, so a stable colour pattern across frames means the
correspondence is preserved.
"""
from __future__ import annotations
import numpy as np
import colorsys


def vertex_identity_colors(anchor_vertices):
    """Map each anchor vertex to a fixed RGB colour by its spherical position."""
    v = np.asarray(anchor_vertices, dtype=np.float64)
    c = v - v.mean(0, keepdims=True)
    c = c / (np.abs(c).max() + 1e-8)
    az = np.arctan2(c[:, 2], c[:, 0]) / (2 * np.pi) + 0.5
    el = np.arctan2(c[:, 1], np.sqrt(c[:, 0] ** 2 + c[:, 2] ** 2)) / np.pi + 0.5
    hsv = np.stack([az, np.full_like(az, 0.85), 0.45 + 0.45 * el], -1)
    return np.array([colorsys.hsv_to_rgb(*row) for row in hsv])


def render_mesh(mesh, colors, image_size=384, dist=2.9, elev=12.0, azim=32.0):
    """Render one shaded, vertex-coloured mesh to an [H,W,3] float image."""
    import torch
    from pytorch3d.structures import Meshes
    from pytorch3d.renderer import (
        FoVPerspectiveCameras, PointLights, RasterizationSettings,
        MeshRenderer, MeshRasterizer, SoftPhongShader, look_at_view_transform,
        TexturesVertex)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    v = np.asarray(mesh.vertices, np.float32)
    v = v - (v.max(0) + v.min(0)) / 2.0
    v = v / (np.abs(v).max() + 1e-8)
    verts = torch.tensor(v, device=dev)
    faces = torch.tensor(np.asarray(mesh.faces, np.int64), device=dev)
    tex = TexturesVertex(torch.tensor(colors, dtype=torch.float32, device=dev)[None])
    m = Meshes([verts], [faces], tex)
    R, T = look_at_view_transform(dist, elev, azim)
    cam = FoVPerspectiveCameras(device=dev, R=R, T=T, fov=49)
    lights = PointLights(device=dev, location=[[2.0, 2.0, 2.0]])
    raster = RasterizationSettings(image_size=image_size, blur_radius=0.0, faces_per_pixel=1)
    renderer = MeshRenderer(MeshRasterizer(cameras=cam, raster_settings=raster),
                            SoftPhongShader(device=dev, cameras=cam, lights=lights))
    return renderer(m)[0, ..., :3].clamp(0, 1).cpu().numpy()
