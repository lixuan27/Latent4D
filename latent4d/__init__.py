"""Latent-4D: 4D representation is already inside your 3D decoder.

Public API:
    readout(decoder, frames, cfg)          training-free 4D mesh read-out (Alg. 1)
    ReadoutConfig                          K, eta, eps, delta, alpha
    viewpoint_similarity, attach, couple   inference-time Apex hook (Alg. 2)
    tangential.refine                      tangential refinement (Alg. 3)
    metrics                                cd_3d / cd_4d / cd_motion / mIoU / accel
    decoders.load_decoder                  TripoSG / Hunyuan3D / Step1X adapters
    rendering                              vertex-identity colour + mesh render
"""
from .readout import readout, advect, smooth_latents, ReadoutConfig
from .apex import viewpoint_similarity, couple, attach, ApexHook
from . import tangential, metrics, rendering
from .decoders import load_decoder, DecoderAdapter

__all__ = [
    "readout", "advect", "smooth_latents", "ReadoutConfig",
    "viewpoint_similarity", "couple", "attach", "ApexHook",
    "tangential", "metrics", "rendering", "load_decoder", "DecoderAdapter",
]
__version__ = "0.1.0"
