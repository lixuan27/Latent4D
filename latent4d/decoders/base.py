"""The minimal decoder interface the read-out needs.

Any frozen single-image 3D pipeline that can (a) encode an image into a latent,
(b) answer a signed-distance query s(x; z), and (c) extract one anchor mesh, can
drive Latent-4D. Concrete adapters for TripoSG, Hunyuan3D and Step1X-3D live next
to this file; each is a thin wrapper around the official frozen model.
"""
from __future__ import annotations
import torch


class DecoderAdapter:
    """Subclass and implement the three methods below for a new decoder."""

    def encode(self, image):
        """Image -> latent code z (decoder-specific tensor)."""
        raise NotImplementedError

    def query_sdf(self, points, z):
        """points [N,3], latent z -> signed distances [N] (differentiable in points)."""
        raise NotImplementedError

    def extract_mesh(self, z):
        """latent z -> (verts [V,3], faces [F,3]); used only for the anchor frame."""
        raise NotImplementedError

    # --- provided: gradient of the SDF via autograd ----------------------------
    def sdf_and_grad(self, points, z):
        """Return (s [N], grad_x s [N,3]). Override if the decoder exposes an
        analytic gradient; the default uses autograd on :meth:`query_sdf`."""
        x = points.detach().requires_grad_(True)
        s = self.query_sdf(x, z)
        (g,) = torch.autograd.grad(s.sum(), x, create_graph=False)
        return s.detach(), g.detach()
