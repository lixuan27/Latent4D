"""Reproduce the Chamfer-family numbers (Table 1) from saved mesh sequences.

Operates on directories of per-frame meshes laid out as
    <root>/<uid>/mesh_0000.glb ... mesh_0015.glb
for both the prediction and the ground truth, and reports cd_3d / cd_4d /
cd_motion averaged over the evaluated sequences using latent4d.metrics.
"""
import argparse, glob, os
import numpy as np, torch, trimesh
from latent4d import metrics


def load_seq(root, uid, n=16, device="cpu"):
    vs = []
    for t in range(n):
        p = os.path.join(root, uid, f"mesh_{t:04d}.glb")
        if not os.path.exists(p):
            return None
        v = np.asarray(trimesh.load(p, force="mesh").vertices, np.float32)
        vs.append(torch.tensor(v, device=device))
    return vs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", required=True, help="dir of predicted <uid>/mesh_*.glb")
    ap.add_argument("--gt", required=True, help="dir of ground-truth <uid>/mesh_*.glb")
    ap.add_argument("--metric", default="chamfer", choices=["chamfer", "miou"])
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    uids = sorted(os.path.basename(d) for d in glob.glob(os.path.join(args.pred, "*")))
    c3, c4, cm, seen = [], [], [], 0
    for uid in uids:
        P = load_seq(args.pred, uid, device=args.device)
        G = load_seq(args.gt, uid, device=args.device)
        if P is None or G is None:
            continue
        with torch.no_grad():
            c3.append(metrics.cd_3d(P, G).item())
            c4.append(metrics.cd_4d(P, G).item())
            if P[0].shape == G[0].shape:
                cm.append(metrics.cd_motion(P, G).item())
        seen += 1
    print(f"evaluated {seen} sequences")
    print(f"cd_3d   = {np.mean(c3):.4f}")
    print(f"cd_4d   = {np.mean(c4):.4f}")
    if cm:
        print(f"cd_motion = {np.mean(cm):.4f}  (over {len(cm)} with shared indexing)")


if __name__ == "__main__":
    main()
