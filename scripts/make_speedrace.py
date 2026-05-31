"""Generation-speed race video for the project page.

A wall-clock race using the paper's real per-clip inference times (Table 3):
  ActionMesh (trained backbone)        120.0 s   -> baseline meshes
  Latent-4D (frozen substrate, ours)     9.4 s   -> TripoSG read-out meshes
  Latent-4D + early exit (ours)          4.96 s  -> Apex meshes

Each panel shows a clock and a progress bar that advance over a simulated
wall-clock of 0..125 s (playback accelerated; the labels are the real seconds).
While a method is still 'generating' its panel shows a progress placeholder;
the instant its real inference time is reached the panel reveals and then loops
the actual produced 4D mesh sequence. So the two Latent-4D panels show a moving
result almost immediately while ActionMesh keeps computing until 120 s.

Pre-renders each method's 16 mesh frames once, then composites cheaply.
"""
import os, sys, glob
import numpy as np, trimesh
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
sys.path.insert(0, os.path.dirname(__file__))
from render_qualitative import render_mesh, vertex_identity_colors

UID = "000-031_cc1f905b148c4378ad46a40da72e839f"
METHODS = [
    ("ActionMesh", 120.0, "outputs/actionbench_baseline_78006", "#b0492f", "trained 4D backbone"),
    ("Latent-4D (substrate)", 9.4, "outputs/actionbench_route_a_78744", "#3f7d52", "ours, frozen read-out"),
    ("Latent-4D + early exit", 4.96, "outputs/large_savc_a0.9_80653", "#3f7d52", "ours"),
]
OUT = "outputs/page_speedrace"
N = 16
SIM_T = 125.0          # simulated wall-clock seconds spanned by the video
VID_FRAMES = 150       # ~12.5 s at 12 fps
SZ = 360


def render_method(root):
    g0 = trimesh.load(os.path.join(root, "meshes", UID, "mesh_0000.glb"), force="mesh")
    col0 = vertex_identity_colors(np.asarray(g0.vertices))
    frames = []
    for t in range(N):
        m = trimesh.load(os.path.join(root, "meshes", UID, f"mesh_{t:04d}.glb"), force="mesh")
        col = col0 if len(col0) == len(m.vertices) else vertex_identity_colors(np.asarray(m.vertices))
        frames.append(render_mesh(m, col, image_size=SZ))
    return frames


def main():
    os.makedirs(OUT, exist_ok=True)
    rendered = [render_method(r) for _, _, r, _, _ in METHODS]
    K = len(METHODS)
    import imageio
    video = []
    for vf in range(VID_FRAMES):
        sim = SIM_T * vf / (VID_FRAMES - 1)          # current simulated second
        fig, axes = plt.subplots(1, K, figsize=(K * 3.0, 3.7))
        for c, (name, tinf, _root, color, tag) in enumerate(METHODS):
            ax = axes[c]; ax.axis("off")
            ax.set_xlim(0, 1); ax.set_ylim(0, 1)
            done = sim >= tinf
            # title
            ax.text(0.5, 1.06, name, ha="center", va="bottom", fontsize=13,
                    fontweight="bold", color=color, transform=ax.transAxes)
            ax.text(0.5, 1.00, tag, ha="center", va="bottom", fontsize=9,
                    color="#777", transform=ax.transAxes)
            # result area 0.18..0.92 vertical
            if done:
                # loop the produced 4D sequence from the moment it finished
                start_vf = int(tinf / SIM_T * (VID_FRAMES - 1))
                idx = (vf - start_vf) % N
                ax.imshow(rendered[c][idx], extent=(0.06, 0.94, 0.20, 0.94), aspect="auto", zorder=2)
                ax.text(0.5, 0.10, f"done in {tinf:g}s", ha="center", fontsize=12,
                        fontweight="bold", color=color, transform=ax.transAxes)
            else:
                ax.add_patch(FancyBboxPatch((0.10, 0.24), 0.80, 0.62,
                             boxstyle="round,pad=0.01", fc="#f3f1ea", ec="#ddd", zorder=1,
                             transform=ax.transAxes))
                ax.text(0.5, 0.58, "generating", ha="center", fontsize=12, color="#999",
                        transform=ax.transAxes)
                ax.text(0.5, 0.45, f"{sim:5.1f}s / {tinf:g}s", ha="center", fontsize=13,
                        fontweight="bold", color="#555", transform=ax.transAxes)
                ax.text(0.5, 0.10, "computing...", ha="center", fontsize=11, color="#999",
                        transform=ax.transAxes)
            # progress bar at y=0.04
            frac = min(sim / tinf, 1.0)
            ax.add_patch(plt.Rectangle((0.10, 0.02), 0.80, 0.035, fc="#e4e0d6",
                         transform=ax.transAxes, zorder=1))
            ax.add_patch(plt.Rectangle((0.10, 0.02), 0.80 * frac, 0.035, fc=color,
                         transform=ax.transAxes, zorder=2))
        fig.suptitle(f"Wall-clock:  {sim:5.1f}s   (playback accelerated; labels are real inference seconds)",
                     fontsize=12, y=0.04, color="#333")
        plt.subplots_adjust(left=0.01, right=0.99, top=0.88, bottom=0.10, wspace=0.06)
        fig.canvas.draw()
        buf = np.frombuffer(fig.canvas.buffer_rgba(), np.uint8).reshape(
            fig.canvas.get_width_height()[::-1] + (4,))[..., :3].copy()
        video.append(buf); plt.close(fig)
        if vf % 30 == 0:
            print(f"frame {vf}/{VID_FRAMES} sim={sim:.1f}s")
    # hold the final state a moment, then this loops naturally
    out = os.path.join(OUT, "speed_race.mp4")
    imageio.mimsave(out, video, fps=12, quality=8, macro_block_size=8)
    # poster: a moment where ours are done but ActionMesh still computing (sim ~ 30s)
    imageio.imwrite(out.replace(".mp4", ".jpg"), video[int(30 / SIM_T * (VID_FRAMES - 1))])
    print("saved", out, "frames", len(video))


if __name__ == "__main__":
    main()
