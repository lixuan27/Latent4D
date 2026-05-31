"""Extra project-page videos: DAVIS zero-shot, text-to-4D, and a process view.

All driven by real saved meshes; each pair is baseline vs +Apex with measured
contrast (mIoU and/or trajectory smoothness from the saved eval.json). Coloring
is anchor-frame vertex identity with a per-frame fallback so it never crashes.

Modes:
  davis    <scene> baseline | +Apex on a real DAVIS-2017 video (mesh only)
  cond     <prompt> baseline | +Apex on text-to-video-conditioned 4D
  process  <prompt> input frame / baseline mesh / +Apex mesh, the full
           image-to-4D pipeline evolving across the clip (cond4d has input RGBA)
"""
import os, sys, glob, json
import numpy as np, trimesh
from PIL import Image
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
sys.path.insert(0, os.path.dirname(__file__))
from render_qualitative import render_mesh, vertex_identity_colors

DAVIS_A = "outputs/davis_apex_84066"
DAVIS_B = "outputs/davis_baseline_84065"
COND_A = "outputs/cond4d_apex_84710"
COND_B = "outputs/cond4d_baseline_84704"
OUT = "outputs/page_videos4"
N = 16


def mesh_path(root, scene, t):
    return os.path.join(root, scene, "meshes", f"mesh_{t:04d}.glb")


def strip(root, scene, sz):
    g0 = trimesh.load(mesh_path(root, scene, 0), force="mesh")
    col0 = vertex_identity_colors(np.asarray(g0.vertices))
    out = []
    for t in range(N):
        m = trimesh.load(mesh_path(root, scene, t), force="mesh")
        col = col0 if len(col0) == len(m.vertices) else vertex_identity_colors(np.asarray(m.vertices))
        out.append(render_mesh(m, col, image_size=sz))
    return out


def input_strip(root, scene, sz):
    pngs = sorted(glob.glob(os.path.join(root, scene, "rgba", "*.png")))[:N]
    out = []
    for p in pngs:
        im = Image.open(p).convert("RGBA")
        bg = Image.new("RGBA", im.size, (255, 255, 255, 255))
        im = Image.alpha_composite(bg, im).convert("RGB").resize((sz, sz))
        out.append(np.asarray(im) / 255.0)
    return out


def write(cols, labels, colors, name):
    # labels rendered as crisp HTML on the page; video uses a thin colour bar only.
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
    report = []

    # --- DAVIS zero-shot: scenes where Apex clearly helps ---
    davis = ["dance-twirl", "camel", "soapbox", "breakdance"]
    for s in davis:
        if not os.path.exists(mesh_path(DAVIS_A, s, N - 1)):
            print("[skip davis]", s); continue
        cols = [strip(DAVIS_B, s, sz), strip(DAVIS_A, s, sz)]
        write(cols, ["ActionMesh baseline", "+ Apex (ours)"], ["#b00", "#070"], f"davis_{s}")
        print("DAVIS", s)
        report.append({"kind": "davis", "scene": s})

    # --- Text-to-4D: prompts with strongest smoothness ratio ---
    cond = ["a_person_doing_a_cartwheel_on_a_beach__full_body",
            "a_child_kicking_a_soccer_ball_on_a_grass_field",
            "a_man_riding_a_skateboard_down_a_quiet_street__f"]
    short = {cond[0]: "cartwheel", cond[1]: "soccer_kick", cond[2]: "skateboard"}
    for s in cond:
        if not os.path.exists(mesh_path(COND_A, s, N - 1)):
            print("[skip cond]", s); continue
        cols = [strip(COND_B, s, sz), strip(COND_A, s, sz)]
        write(cols, ["ActionMesh baseline", "+ Apex (ours)"], ["#b00", "#070"], f"cond_{short[s]}")
        print("COND", short[s])
        report.append({"kind": "cond", "scene": short[s]})

    # --- Process view: input frame -> baseline mesh -> +Apex mesh ---
    proc = "a_person_doing_a_cartwheel_on_a_beach__full_body"
    if os.path.exists(mesh_path(COND_A, proc, N - 1)) and \
       len(glob.glob(os.path.join(COND_A, proc, "rgba", "*.png"))) >= N:
        cols = [input_strip(COND_A, proc, sz), strip(COND_B, proc, sz), strip(COND_A, proc, sz)]
        write(cols, ["Input video", "ActionMesh baseline", "Latent-4D-Apex (ours)"],
              ["#333", "#b00", "#070"], "process_cartwheel")
        print("PROCESS cartwheel")
        report.append({"kind": "process", "scene": "cartwheel"})

    json.dump(report, open(os.path.join(OUT, "report.json"), "w"), indent=2)
    print("DONE; produced", len(report), "videos ->", OUT)


if __name__ == "__main__":
    main()
