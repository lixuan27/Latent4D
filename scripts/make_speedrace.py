"""Generation-speed race video for the project page (2 methods, 3 clips).

A wall-clock race using the paper's real per-clip inference times (Table 3):
  ActionMesh (trained backbone)     120.0 s  -> baseline meshes
  Latent-4D + early exit (ours)       4.96 s -> Apex meshes

The clock advances over a simulated 0..125 s (playback accelerated; the labels
are the real seconds). While a method is still 'generating' its panel shows a
progress placeholder; the instant its real inference time is reached the panel
reveals and loops the actual produced 4D mesh sequence. So the Latent-4D column
shows a moving result almost immediately while ActionMesh keeps computing to 120s.

Three clips are stacked as rows; each clip is chosen so the Apex result is much
smoother than the baseline (measured trajectory-acceleration ratio >= 2x).
"""
import os, sys
import numpy as np, trimesh
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
sys.path.insert(0, os.path.dirname(__file__))
from render_qualitative import render_mesh, vertex_identity_colors

# (label, inference seconds, mesh root, colour)
METHODS = [
    ("ActionMesh", 120.0, "outputs/actionbench_baseline_78006", "#b0492f"),
    ("Latent-4D + early exit", 4.96, "outputs/large_savc_a0.9_80653", "#3f7d52"),
]
CLIPS = [
    ("000-075_46f5711e3ac04900a49732d78f4f64d8", "Leaping character"),
    ("000-130_94be397f8db74f7c9c06263aaf988532", "Creature"),
    ("000-003_90288386ca914411ab648f08db734fea", "Acrobatic figure"),
]
OUT = "outputs/page_speedrace"
N = 16
SIM_T = 125.0
VID_FRAMES = 150
SZ = 300


def render_clip(root, uid):
    g0 = trimesh.load(os.path.join(root, "meshes", uid, "mesh_0000.glb"), force="mesh")
    col0 = vertex_identity_colors(np.asarray(g0.vertices))
    frames = []
    for t in range(N):
        m = trimesh.load(os.path.join(root, "meshes", uid, f"mesh_{t:04d}.glb"), force="mesh")
        col = col0 if len(col0) == len(m.vertices) else vertex_identity_colors(np.asarray(m.vertices))
        frames.append(render_mesh(m, col, image_size=SZ))
    return frames


def main():
    os.makedirs(OUT, exist_ok=True)
    # rendered[clip][method] -> list of N frames
    rendered = [[render_clip(root, uid) for _, _, root, _ in METHODS] for uid, _ in CLIPS]
    K = len(METHODS); R = len(CLIPS)
    import imageio
    video = []
    for vf in range(VID_FRAMES):
        sim = SIM_T * vf / (VID_FRAMES - 1)
        fig, axes = plt.subplots(R, K, figsize=(K * 3.1, R * 3.0))
        for r in range(R):
            for c in range(K):
                ax = axes[r][c]; ax.axis("off"); ax.set_xlim(0, 1); ax.set_ylim(0, 1)
                name, tinf, _root, color = METHODS[c]
                if r == 0:
                    ax.text(0.5, 1.08, name, ha="center", va="bottom", fontsize=13,
                            fontweight="bold", color=color, transform=ax.transAxes)
                ax.text(0.02, 0.5, CLIPS[r][1], ha="right", va="center", fontsize=10,
                        color="#777", rotation=90, transform=ax.transAxes) if c == 0 else None
                done = sim >= tinf
                if done:
                    start_vf = int(tinf / SIM_T * (VID_FRAMES - 1))
                    idx = (vf - start_vf) % N
                    ax.imshow(rendered[r][c][idx], extent=(0.06, 0.94, 0.18, 0.93), aspect="auto", zorder=2)
                    ax.text(0.5, 0.07, f"done in {tinf:g}s", ha="center", fontsize=12,
                            fontweight="bold", color=color, transform=ax.transAxes)
                else:
                    ax.add_patch(FancyBboxPatch((0.10, 0.20), 0.80, 0.66,
                                 boxstyle="round,pad=0.01", fc="#f3f1ea", ec="#ddd", zorder=1,
                                 transform=ax.transAxes))
                    ax.text(0.5, 0.58, "generating", ha="center", fontsize=12, color="#999",
                            transform=ax.transAxes)
                    ax.text(0.5, 0.44, f"{sim:5.1f}s / {tinf:g}s", ha="center", fontsize=13,
                            fontweight="bold", color="#555", transform=ax.transAxes)
                frac = min(sim / tinf, 1.0)
                ax.add_patch(plt.Rectangle((0.10, 0.01), 0.80, 0.03, fc="#e4e0d6",
                             transform=ax.transAxes, zorder=1))
                ax.add_patch(plt.Rectangle((0.10, 0.01), 0.80 * frac, 0.03, fc=color,
                             transform=ax.transAxes, zorder=2))
        fig.suptitle(f"Wall-clock:  {sim:5.1f}s    (playback accelerated; labels are real inference seconds)",
                     fontsize=12, y=0.02, color="#333")
        plt.subplots_adjust(left=0.04, right=0.99, top=0.93, bottom=0.05, wspace=0.05, hspace=0.12)
        fig.canvas.draw()
        buf = np.frombuffer(fig.canvas.buffer_rgba(), np.uint8).reshape(
            fig.canvas.get_width_height()[::-1] + (4,))[..., :3].copy()
        video.append(buf); plt.close(fig)
        if vf % 30 == 0:
            print(f"frame {vf}/{VID_FRAMES} sim={sim:.1f}s")
    out = os.path.join(OUT, "speed_race.mp4")
    imageio.mimsave(out, video, fps=12, quality=8, macro_block_size=8)
    imageio.imwrite(out.replace(".mp4", ".jpg"), video[int(30 / SIM_T * (VID_FRAMES - 1))])
    print("saved", out, "frames", len(video))


if __name__ == "__main__":
    main()
