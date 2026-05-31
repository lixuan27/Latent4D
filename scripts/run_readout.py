"""Read a temporally coherent 4D mesh sequence out of a frozen decoder.

Reference driver built on the latent4d package (Algorithm 1). For the exact
scripts used in the paper, see scripts/exp_triposg_readout.py and friends.

Example:
  python scripts/run_readout.py --decoder triposg \
      --ckpt $HF_HOME/triposg --clip data/actionbench/data/000-031_*/imgs \
      --out outputs/demo
"""
import argparse, glob, os
import numpy as np, trimesh
from latent4d import load_decoder, readout, ReadoutConfig


def load_clip(pattern, n=16):
    paths = sorted(glob.glob(os.path.join(pattern, "*.png")))[:n]
    if not paths:
        raise FileNotFoundError(f"no frames under {pattern}")
    return paths


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--decoder", default="triposg", choices=["triposg", "hunyuan3d", "step1x"])
    ap.add_argument("--ckpt", default=os.environ.get("LATENT4D_CKPT", ""))
    ap.add_argument("--clip", required=True, help="dir of frame PNGs (glob-expanded)")
    ap.add_argument("--out", default="outputs/demo")
    ap.add_argument("--K", type=int, default=20)
    ap.add_argument("--eta", type=float, default=0.7)
    ap.add_argument("--alpha", type=float, default=0.5)
    args = ap.parse_args()

    os.makedirs(os.path.join(args.out, "meshes"), exist_ok=True)
    clip_dir = sorted(glob.glob(args.clip))[0] if glob.glob(args.clip) else args.clip
    frames = load_clip(clip_dir)
    dec = load_decoder(args.decoder, args.ckpt)
    cfg = ReadoutConfig(K=args.K, eta=args.eta, alpha=args.alpha)
    verts_seq, faces = readout(dec, frames, cfg)

    f = faces.detach().cpu().numpy()
    for t, v in enumerate(verts_seq):
        m = trimesh.Trimesh(v.detach().cpu().numpy(), f, process=False)
        m.export(os.path.join(args.out, "meshes", f"mesh_{t:04d}.glb"))
    print(f"wrote {len(verts_seq)} frames to {args.out}/meshes")


if __name__ == "__main__":
    main()
