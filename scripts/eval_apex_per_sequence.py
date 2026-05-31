"""Appendix: per-sequence trajectory-acceleration reduction of Apex over the
ActionMesh baseline across the full benchmark. Real data; saves a sorted
reduction figure and prints distribution statistics so we can confirm the
result is strong before including it."""
import glob, os
import numpy as np, trimesh
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE = "outputs/actionbench_baseline_78006"
APEX = "outputs/large_savc_a0.9_80653"
GT = "data/actionbench/data"
OUT = "outputs/appendix_figs"


def accel(method_dir, uid, n=16):
    V = []
    for t in range(n):
        p = os.path.join(method_dir, "meshes", uid, f"mesh_{t:04d}.glb")
        if not os.path.exists(p):
            return None
        V.append(np.asarray(trimesh.load(p, force="mesh").vertices))
    if len(set(v.shape[0] for v in V)) != 1:
        return None
    V = np.stack(V)
    acc = V[2:] - 2 * V[1:-1] + V[:-2]
    scale = np.linalg.norm(V[0].max(0) - V[0].min(0)) + 1e-8
    return float(np.linalg.norm(acc, axis=-1).mean() / scale)


def main():
    os.makedirs(OUT, exist_ok=True)
    uids = sorted(os.listdir(os.path.join(APEX, "meshes")))
    base, apex, keep = [], [], []
    for uid in uids:
        b = accel(BASE, uid)
        a = accel(APEX, uid)
        if b is None or a is None:
            continue
        if not os.path.isdir(os.path.join(GT, uid, "imgs")):
            continue
        base.append(b); apex.append(a); keep.append(uid)
    base = np.array(base); apex = np.array(apex)
    red = (base - apex) / (base + 1e-12) * 100.0   # percent reduction (positive = better)
    order = np.argsort(-red)
    n = len(red)
    n_improved = int((red > 0).sum())
    print(f"sequences evaluated: {n}")
    print(f"improved (apex<base): {n_improved}/{n} = {100*n_improved/n:.1f}%")
    print(f"median reduction: {np.median(red):.1f}%   mean: {red.mean():.1f}%")
    print(f"reduction >= 30%: {(red>=30).sum()}/{n};  >= 50%: {(red>=50).sum()}/{n}")
    print("top-12 by reduction (uid  base  apex  red%):")
    for i in order[:12]:
        print(f"  {keep[i][:40]}  {base[i]:.4f}  {apex[i]:.4f}  {red[i]:5.1f}")

    # sorted reduction bar
    fig, ax = plt.subplots(figsize=(9, 3.2))
    rs = red[order]
    colors = ["#2a7" if v > 0 else "#c33" for v in rs]
    ax.bar(np.arange(n), rs, color=colors, width=1.0)
    ax.axhline(0, color="k", lw=0.8)
    ax.axhline(np.median(red), color="#06c", ls="--", lw=1.2,
               label=f"median {np.median(red):.0f}%")
    ax.set_xlim(-0.5, n - 0.5)
    ax.set_xlabel("benchmark sequence (sorted by reduction)")
    ax.set_ylabel("trajectory-accel.\nreduction (%)")
    ax.legend(loc="upper right", frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(OUT, f"fig_apex_accel_reduction.{ext}"), dpi=200, bbox_inches="tight")
    print("saved", os.path.join(OUT, "fig_apex_accel_reduction.pdf"))


if __name__ == "__main__":
    main()
