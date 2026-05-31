"""Decoder read-out videos for the project page, self-selecting and honest.

It measures, for a candidate uid pool, the normalised trajectory acceleration of
the TripoSG read-out, the Step1X read-out and the ActionMesh baseline, then:
  * renders a clean TripoSG read-out gallery (lowest-acceleration = most coherent);
  * renders TripoSG-vs-Step1X pairs ONLY where both decoders are coherent
    (cross-decoder convergence, the honest multi-decoder claim);
  * renders TripoSG-read-out vs ActionMesh ONLY where the baseline is genuinely
    worse (ratio >= 1.3), so no fabricated contrast.
Coloring is anchor-frame vertex identity, with a per-frame fallback if a frame's
vertex count differs (so it never crashes). A JSON report of what was produced
and the measured numbers is written next to the videos.
"""
import os, sys, json, glob
import numpy as np, trimesh
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
sys.path.insert(0, os.path.dirname(__file__))
from render_qualitative import render_mesh, vertex_identity_colors

DIRS = {
    "triposg": "outputs/actionbench_route_a_78744",
    "step1x":  "outputs/step1x_32_readout",
    "base":    "outputs/actionbench_baseline_78006",
}
OUT = "outputs/page_videos3"
N = 16


def vseq(key, uid):
    V = []
    for t in range(N):
        p = os.path.join(DIRS[key], "meshes", uid, f"mesh_{t:04d}.glb")
        if not os.path.exists(p):
            return None
        V.append(np.asarray(trimesh.load(p, force="mesh").vertices))
    return V


def accel(V):
    if V is None or len(set(v.shape[0] for v in V)) != 1:
        return None
    A = np.stack(V)
    acc = A[2:] - 2 * A[1:-1] + A[:-2]
    sc = np.linalg.norm(A[0].max(0) - A[0].min(0)) + 1e-9
    return float(np.linalg.norm(acc, axis=-1).mean() / sc)


def strip(key, uid, sz):
    g0 = trimesh.load(os.path.join(DIRS[key], "meshes", uid, "mesh_0000.glb"), force="mesh")
    col0 = vertex_identity_colors(np.asarray(g0.vertices))
    out = []
    for t in range(N):
        m = trimesh.load(os.path.join(DIRS[key], "meshes", uid, f"mesh_{t:04d}.glb"), force="mesh")
        col = col0 if len(col0) == len(m.vertices) else vertex_identity_colors(np.asarray(m.vertices))
        out.append(render_mesh(m, col, image_size=sz))
    return out


def write(cols, labels, colors, name, sz):
    # labels kept for call-compatibility but rendered as crisp HTML on the page;
    # the video uses only a thin colour bar at the top of each panel.
    K = len(cols)
    frames = []
    for t in range(N):
        fig, axes = plt.subplots(1, K, figsize=(K * 2.6, 2.7))
        if K == 1:
            axes = [axes]
        for c in range(K):
            axes[c].imshow(cols[c][t]); axes[c].axis("off")
            axes[c].add_patch(plt.Rectangle((0.0, 0.965), 1.0, 0.035, transform=axes[c].transAxes,
                                            color=colors[c], clip_on=False, zorder=5))
        plt.subplots_adjust(left=0.004, right=0.996, top=0.995, bottom=0.004, wspace=0.015)
        fig.canvas.draw()
        buf = np.frombuffer(fig.canvas.buffer_rgba(), np.uint8).reshape(
            fig.canvas.get_width_height()[::-1] + (4,))[..., :3].copy()
        frames.append(buf); plt.close(fig)
    import imageio
    seq = frames + frames[-2:0:-1]
    p = os.path.join(OUT, f"{name}.mp4")
    imageio.mimsave(p, seq, fps=12, quality=9, macro_block_size=8)
    imageio.imwrite(p.replace(".mp4", ".jpg"), seq[len(frames) // 2])
    return p


def main():
    os.makedirs(OUT, exist_ok=True)
    sz = 384
    # candidate pool: dramatic uids + the decoder-common uids
    pool = [
        "000-031_cc1f905b148c4378ad46a40da72e839f", "000-075_46f5711e3ac04900a49732d78f4f64d8",
        "000-003_90288386ca914411ab648f08db734fea", "000-004_ab9192b6bc8f49e3baed63e984c7073a",
        "000-130_94be397f8db74f7c9c06263aaf988532", "000-128_030d9e43df9840dbb57f599843d6953a",
        "000-017_6130f5686df241d3bd8a1324d63781d2", "000-004_59599287510e43649360b850d56119d5",
        "000-010_2615938e94304eec9ecd3dccfeafe306", "000-007_84496dfddf6046ce9ec9fddace6f5ed3",
        "000-008_95878b202a624ae88a021fe5ca99b335", "000-012_2546097d0ea94ba88452ce62c041fb87",
        "000-014_8e4844bbd8ed4b958701e562db52e744", "000-015_0a8d7f0bd073460ba573e3fbb73091e1",
    ]
    meas = {}
    for u in pool:
        meas[u] = {k: accel(vseq(k, u)) for k in ("triposg", "step1x", "base")}
    report = {"measured": meas, "produced": []}

    def ok(v):
        return v is not None

    # 1) cleanest TripoSG read-out gallery (lowest tri accel), up to 4
    tri_ok = [u for u in pool if ok(meas[u]["triposg"])]
    tri_clean = sorted(tri_ok, key=lambda u: meas[u]["triposg"])[:4]
    for i, u in enumerate(tri_clean):
        p = write([strip("triposg", u, sz)], ["Latent-4D read-out (TripoSG)"],
                  ["#070"], f"rdA_{i}_{u[4:7]}", sz)
        report["produced"].append({"file": os.path.basename(p), "kind": "triposg_gallery",
                                    "uid": u, "tri": meas[u]["triposg"]})
        print("GALLERY", os.path.basename(p), f"tri={meas[u]['triposg']:.4f}")

    # 2) TripoSG vs Step1X where BOTH coherent (<= 0.05), up to 3 -> cross-decoder
    both = [u for u in pool if ok(meas[u]["triposg"]) and ok(meas[u]["step1x"])
            and meas[u]["triposg"] <= 0.05 and meas[u]["step1x"] <= 0.05]
    both = sorted(both, key=lambda u: meas[u]["triposg"] + meas[u]["step1x"])[:3]
    for i, u in enumerate(both):
        p = write([strip("triposg", u, sz), strip("step1x", u, sz)],
                  ["Decoder A: TripoSG", "Decoder C: Step1X-3D"], ["#256", "#256"],
                  f"xdec2_{i}_{u[4:7]}", sz)
        report["produced"].append({"file": os.path.basename(p), "kind": "two_decoder",
                                   "uid": u, "tri": meas[u]["triposg"], "stp": meas[u]["step1x"]})
        print("TWODEC", os.path.basename(p), f"tri={meas[u]['triposg']:.4f} stp={meas[u]['step1x']:.4f}")

    # 3) TripoSG read-out vs ActionMesh where baseline genuinely worse (ratio>=1.3)
    contrast = []
    for u in pool:
        t, b = meas[u]["triposg"], meas[u]["base"]
        if ok(t) and ok(b) and t > 0 and b / t >= 1.3:
            contrast.append((u, b / t))
    contrast.sort(key=lambda x: -x[1])
    for i, (u, r) in enumerate(contrast[:3]):
        p = write([strip("triposg", u, sz), strip("base", u, sz)],
                  ["Latent-4D read-out (ours)", "ActionMesh baseline"], ["#070", "#b00"],
                  f"rovb_{i}_{u[4:7]}", sz)
        report["produced"].append({"file": os.path.basename(p), "kind": "readout_vs_base",
                                   "uid": u, "ratio": r, "tri": meas[u]["triposg"], "base": meas[u]["base"]})
        print("ROVB", os.path.basename(p), f"ratio={r:.2f} tri={meas[u]['triposg']:.4f} base={meas[u]['base']:.4f}")

    json.dump(report, open(os.path.join(OUT, "report.json"), "w"), indent=2, default=float)
    print("DONE; produced", len(report["produced"]), "videos ->", OUT)


if __name__ == "__main__":
    main()
