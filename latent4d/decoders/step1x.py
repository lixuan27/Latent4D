"""Step1X-3D adapter (decoder C, released 2025).

A third, independently trained vector-set decoder, included to show that the
read-out transfers without re-tuning. Same frozen interface as the other two;
install Step1X-3D from its official repository.
"""
from __future__ import annotations
import torch
from .base import DecoderAdapter


class Step1XDecoder(DecoderAdapter):
    def __init__(self, ckpt, device="cuda"):
        from step1x3d_geometry.models.pipelines.pipeline import Step1X3DGeometryPipeline
        self.device = device
        self.pipe = Step1X3DGeometryPipeline.from_pretrained(ckpt).to(device)
        for p in self.pipe.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def encode(self, image):
        out = self.pipe(image, output_type="latent")
        return out.mesh

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
