"""Frozen decoder adapters. Each wraps an official, unmodified model.

Import lazily so that the package can be used with whichever decoder you have
installed without requiring the others.
"""
from .base import DecoderAdapter


def load_decoder(name, ckpt, device="cuda"):
    name = name.lower()
    if name in ("a", "triposg"):
        from .triposg import TripoSGDecoder
        return TripoSGDecoder(ckpt, device)
    if name in ("b", "hunyuan3d", "hunyuan"):
        from .hunyuan3d import Hunyuan3DDecoder
        return Hunyuan3DDecoder(ckpt, device)
    if name in ("c", "step1x", "step1x3d"):
        from .step1x import Step1XDecoder
        return Step1XDecoder(ckpt, device)
    raise ValueError(f"unknown decoder {name}")
