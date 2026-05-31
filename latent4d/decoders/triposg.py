"""TripoSG adapter (decoder A).

Wraps the official frozen TripoSG vector-set pipeline. Install TripoSG from its
official repository and point ``ckpt`` at the released checkpoint; nothing here is
trained. The signed distance is read straight out of the frozen VAE decoder, so
its spatial gradient (used by the Newton advection) is available through autograd.
"""
from __future__ import annotations
import torch
from .base import DecoderAdapter


class TripoSGDecoder(DecoderAdapter):
    def __init__(self, ckpt, device="cuda"):
        from triposg.pipelines.pipeline_triposg import TripoSGPipeline  # official repo
        self.device = device
        self.pipe = TripoSGPipeline.from_pretrained(ckpt).to(device)
        self.pipe.vae.eval()
        for p in self.pipe.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def encode(self, image):
        # one forward pass of the frozen latent generator; isosurface is skipped
        out = self.pipe(image, output_type="latent", return_dict=True)
        return out.mesh                     # the vector-set latent z

    def query_sdf(self, points, z):
        decoded = self.pipe.vae.decode(z)   # decoder features for this latent
        return self.pipe.vae.query(points.unsqueeze(0), decoded).squeeze(0)

    @torch.no_grad()
    def extract_mesh(self, z):
        decoded = self.pipe.vae.decode(z)
        mesh = self.pipe.vae.extract_geometry(decoded)
        v = torch.as_tensor(mesh.vertices, dtype=torch.float32, device=self.device)
        f = torch.as_tensor(mesh.faces, dtype=torch.long, device=self.device)
        return v, f
