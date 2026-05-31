"""Algorithm 1: training-free Latent-4D read-out.

Given a frozen single-image 3D decoder that exposes a signed-distance query
s(x; z), we extract one anchor mesh from the first frame and advect its vertices
through the signed-distance field of every later frame with a few Newton steps,
following the gradient flow of Section 3 of the paper. No weight is trained; the
only inputs besides the clip are the numerical tolerances (K, eta, eps, delta).

The decoder is accessed only through the small interface in
``latent4d.decoders.base.DecoderAdapter``; any model satisfying it can be used.
"""
from __future__ import annotations
from dataclasses import dataclass
import torch


@dataclass
class ReadoutConfig:
    K: int = 20            # Newton steps per frame
    eta: float = 0.7       # step size
    eps: float = 1e-6      # gradient regulariser
    delta: float = 0.05    # per-step displacement clip (normalised units)
    alpha: float = 0.5     # latent-coherence smoothing weight (0 = off)


def smooth_latents(latents, alpha: float):
    """First-order causal IIR over the latent stream (Eq. 6).

    latents: list of tensors [z_0, ..., z_{T-1}] with identical shape.
    Returns the smoothed stream; alpha = 0 is a no-op, alpha = 1 freezes z_0.
    """
    if alpha <= 0:
        return latents
    out = [latents[0]]
    for t in range(1, len(latents)):
        out.append((1.0 - alpha) * latents[t] + alpha * out[t - 1])
    return out


@torch.no_grad()
def _clip_norm(d, delta):
    n = d.norm(dim=-1, keepdim=True)
    factor = torch.clamp(delta / (n + 1e-12), max=1.0)
    return d * factor


def advect(decoder, z, x0, cfg: ReadoutConfig):
    """Advect points x0 [N,3] onto the zero level set of s(.; z) by K Newton steps.

    Each step is x <- x - eta * s(x) * grad_s(x) / (||grad_s||^2 + eps), the
    forward-Euler discretisation of the Newton gradient flow (Eq. 4-5), with the
    displacement clipped in norm to delta. Returns the advected points.
    """
    x = x0.clone()
    for _ in range(cfg.K):
        s, g = decoder.sdf_and_grad(x, z)             # s:[N], g:[N,3]
        denom = (g * g).sum(-1, keepdim=True) + cfg.eps
        d = -cfg.eta * (s.unsqueeze(-1) * g) / denom
        x = x + _clip_norm(d, cfg.delta)
    return x


def readout(decoder, frames, cfg: ReadoutConfig | None = None):
    """Read a temporally coherent mesh sequence out of a frozen decoder.

    decoder : a DecoderAdapter (encode / sdf_and_grad / extract_mesh).
    frames  : list of input images (decoder-specific format).
    Returns (verts_seq, faces) where verts_seq is a list of [V,3] tensors that
    share the anchor connectivity `faces` and the anchor vertex indexing.
    """
    cfg = cfg or ReadoutConfig()
    z = [decoder.encode(im) for im in frames]          # per-frame latents (Eq. 2)
    z = smooth_latents(z, cfg.alpha)
    verts0, faces = decoder.extract_mesh(z[0])         # anchor mesh (Eq. 3)
    verts_seq = [verts0]
    x = verts0
    for t in range(1, len(frames)):
        x = advect(decoder, z[t], x, cfg)              # Lagrangian transport (Eq. 5)
        verts_seq.append(x)
    return verts_seq, faces
