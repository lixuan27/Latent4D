"""Render result videos for the Latent-4D project page.

Each clip is a 16-frame mesh sequence rendered with anchor-frame vertex-identity
colour. Panels are composed mesh-only with a thin colour bar at the top of each
column (red = baseline, green = ours, teal = frozen read-out) so the columns stay
unambiguous after video compression; the textual labels live in crisp HTML on the
page rather than burnt into the (downscaled, blurry) video. Seamless ping-pong loop.

Manifest lines:  mode|uid|name
  single   one method (Apex output)
  compare  ActionMesh | +Apex
  triple   read-out | +Apex | ActionMesh
"""
import argparse, os, sys
import numpy as np, trimesh
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
sys.path.insert(0, os.path.dirname(__file__))
from render_qualitative import render_mesh, vertex_identity_colors

DIRS = {
    "base": "outputs/actionbench_baseline_78006",
    "apex": "outputs/large_savc_a0.9_80653",
    "triposg": "outputs/actionbench_route_a_78744",
}
RED, GREEN, TEAL = "#b0492f", "#3f7d52", "#2f6db5"
# (key, colour) per column
MODES = {
    "single":  [("apex", GREEN)],
    "compare": [("base", RED), ("apex", GREEN)],
    "triple":  [("triposg", TEAL), ("apex", GREEN), ("base", RED)],
}


def have(uid, keys, n):
    for k in keys:
        for t in range(n):
            if not os.path.exists(os.path.join(DIRS[k], "meshes", uid, f"mesh_{t:04d}.glb")):
                return False
    return True


def strip(key, uid, n, sz):
    g0 = trimesh.load(os.path.join(DIRS[key], "meshes", uid, "mesh_0000.glb"), force="mesh")
    col = vertex_identity_colors(np.asarray(g0.vertices))
    out = []
    for t in range(n):
        m = trimesh.load(os.path.join(DIRS[key], "meshes", uid, f"mesh_{t:04d}.glb"), force="mesh")
        out.append(render_mesh(m, col, image_size=sz))
    return out


def compose_frames(cols, colors, n):
    frames = []
    K = len(cols)
    for t in range(n):
        fig, axes = plt.subplots(1, K, figsize=(K * 2.6, 2.7))
        if K == 1:
            axes = [axes]
        for c in range(K):
            axes[c].imshow(cols[c][t]); axes[c].axis("off")
            # crisp solid colour bar at the very top of the panel
            axes[c].add_patch(plt.Rectangle((0.0, 0.965), 1.0, 0.035, transform=axes[c].transAxes,
                                            color=colors[c], clip_on=False, zorder=5))
        plt.subplots_adjust(left=0.004, right=0.996, top=0.995, bottom=0.004, wspace=0.015)
        fig.canvas.draw()
        buf = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
        buf = buf.reshape(fig.canvas.get_width_height()[::-1] + (4,))[..., :3].copy()
        frames.append(buf)
        plt.close(fig)
    return frames


def write_video(frames, out_path, fps=12):
    import imageio
    seq = frames + frames[-2:0:-1]
    imageio.mimsave(out_path, seq, fps=fps, quality=9, macro_block_size=8)
    imageio.imwrite(out_path.replace(".mp4", ".jpg"), seq[len(frames) // 2])
    return out_path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--out", default="outputs/page_videos")
    ap.add_argument("--size", type=int, default=384)
    ap.add_argument("--frames", type=int, default=16)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    for line in open(args.manifest):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        mode, uid, name = line.split("|")[:3]
        spec = MODES[mode]
        keys = [k for k, _ in spec]
        if not have(uid, keys, args.frames):
            print(f"[skip] {name} ({uid[:20]}): missing meshes")
            continue
        print(f"[{mode}] {name} ({uid[:20]})")
        cols = [strip(k, uid, args.frames, args.size) for k in keys]
        frames = compose_frames(cols, [col for _, col in spec], args.frames)
        print("  saved", write_video(frames, os.path.join(args.out, f"{name}.mp4")))
    print("DONE", args.out)


if __name__ == "__main__":
    main()
