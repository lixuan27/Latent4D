"""Hunyuan3D adapter (decoder B).

Same pattern as :class:`TripoSGDecoder`: a frozen vector-set pipeline whose VAE
decoder answers a signed-distance query, so the spatial gradient used by the
Newton advection comes from autograd. Install Hunyuan3D from its official repo.
"""
from __future__ import annotations
import torch
from .base import DecoderAdapter


class Hunyuan3DDecoder(DecoderAdapter):
    def __init__(self, ckpt, device="cuda"):
        from hy3dgen.shapegen import Hunyuan3DDiTFlowMatchingPipeline  # official repo
        self.device = device
        self.pipe = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained(ckpt).to(device)
        for p in self.pipe.model.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def encode(self, image):
        return self.pipe(image=image, output_type="latent")

    def query_sdf(self, points, z):
        decoded = self.pipe.vae.decode(z)
        return self.pipe.vae.query(points.unsqueeze(0), decoded).squeeze(0)

    @torch.no_grad()
    def extract_mesh(self, z):
        decoded = self.pipe.vae.decode(z)
        mesh = self.pipe.vae.extract_geometry(decoded)
        v = torch.as_tensor(mesh.vertices, dtype=torch.float32, device=self.device)
        f = torch.as_tensor(mesh.faces, dtype=torch.long, device=self.device)
        return v, f
