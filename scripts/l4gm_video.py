"""L4GM + Apex result videos for the project page (reuses the PROVEN pipeline).

This imports the exact, working input/model/hook helpers from l4gm_apex_eval.py
(the script that produced the 42.3% trajectory-acceleration result), then renders
the per-frame Gaussian centres as a 3D point cloud for the baseline L4GM and for
the same model under the frozen-weight Apex hook, side by side. gsplat
rasterization does not JIT-compile here, so we visualise the Gaussian centres
directly (exactly the quantity behind the appendix metric). Point colour encodes
anchor-frame identity, so a stable colour field across frames means coherent
motion. No weight is updated; only the inference-time hook differs.
"""
import os, sys, glob
import numpy as np, torch
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
# proven helpers (these also insert the L4GM repo on sys.path on import)
from l4gm_apex_eval import (config_defaults, LGM, TempAttention, load_file,
                            load_dino, dino_sim, build_input, proj_mat)

OUT = "/public/home/lixuan/lixuan/Latent4D/docs/static/videos"
BETA = 0.3
NUM_FRAMES = 8
N_POINTS = 5000

PICKS = {
    "l4gm_running_humanoid": "000-031_cc1f905b148c4378ad46a40da72e839f",
    "l4gm_quadruped":        "000-004_ab9192b6bc8f49e3baed63e984c7073a",
    "l4gm_character":        "000-128_030d9e43df9840dbb57f599843d6953a",
}


def identity_colors(xyz0):
    import colorsys
    c = xyz0 - xyz0.mean(0, keepdims=True)
    c = c / (np.abs(c).max() + 1e-8)
    az = np.arctan2(c[:, 2], c[:, 0]) / (2 * np.pi) + 0.5
    el = np.arctan2(c[:, 1], np.sqrt(c[:, 0] ** 2 + c[:, 2] ** 2)) / np.pi + 0.5
    return np.array([colorsys.hsv_to_rgb(a, 0.85, 0.45 + 0.45 * e) for a, e in zip(az, el)])


def scatter_frame(base_xyz, apex_xyz, colors, lim, elev=14, azim=40, sz=4):
    fig = plt.figure(figsize=(7.2, 3.7))
    for k, (xyz, lab, col) in enumerate([(base_xyz, "L4GM", "#b00"),
                                         (apex_xyz, "L4GM + Apex", "#070")]):
        ax = fig.add_subplot(1, 2, k + 1, projection="3d")
        ax.scatter(xyz[:, 0], xyz[:, 2], xyz[:, 1], c=colors, s=sz, linewidths=0, alpha=0.9)
        ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim); ax.set_zlim(-lim, lim)
        ax.set_axis_off(); ax.view_init(elev=elev, azim=azim)
        ax.set_title(lab, fontsize=13, color=col, pad=0)
    plt.subplots_adjust(left=0, right=1, top=0.93, bottom=0, wspace=0)
    fig.canvas.draw()
    buf = np.frombuffer(fig.canvas.buffer_rgba(), np.uint8).reshape(
        fig.canvas.get_width_height()[::-1] + (4,))[..., :3].copy()
    plt.close(fig)
    return buf


def make_clip(base_xyz, apex_xyz, name):
    # base_xyz, apex_xyz: [T, N, 3]
    idx = np.random.RandomState(0).choice(base_xyz.shape[1],
                                          min(N_POINTS, base_xyz.shape[1]), replace=False)
    base, apex = base_xyz[:, idx], apex_xyz[:, idx]
    colors = identity_colors(base[0])
    c = base[0].mean(0)
    base = base - c; apex = apex - c
    lim = float(np.abs(base[0]).max()) * 1.3 + 1e-6
    frames = [scatter_frame(base[t], apex[t], colors, lim) for t in range(base.shape[0])]
    import imageio
    seq = frames + frames[-2:0:-1]
    out = os.path.join(OUT, f"{name}.mp4")
    imageio.mimsave(out, seq, fps=10, quality=8, macro_block_size=8)
    imageio.imwrite(out.replace(".mp4", ".jpg"), seq[len(frames) // 2])
    print("  saved", out)


def main():
    os.makedirs(OUT, exist_ok=True)
    device = "cuda"
    opt = config_defaults["big"]; opt.num_frames = NUM_FRAMES
    recon = glob.glob(os.path.expanduser(
        "~/.cache/huggingface/hub/models--jiawei011--L4GM/snapshots/*/recon.safetensors"))[0]
    model = LGM(opt); model.load_state_dict(load_file(recon, device="cpu"), strict=False)
    model = model.half().to(device).eval()
    rays = torch.cat([model.prepare_default_rays(device) for _ in range(opt.num_frames)])
    dino, proc = load_dino(device)

    state = {"A": None, "beta": BETA, "on": False}

    def hook(module, inp, out):
        if not state["on"] or state["A"] is None:
            return out
        BTV, C, H, W = out.shape
        T, V = module.num_frames, module.num_views
        B = BTV // (T * V)
        x = out.reshape(B, T, V, C, H, W)
        A = state["A"].to(out.dtype)
        x = x + state["beta"] * (torch.einsum("ts,bsvchw->btvchw", A, x) - x)
        return x.reshape(BTV, C, H, W)

    nh = 0
    for m in model.modules():
        if isinstance(m, TempAttention):
            m.register_forward_hook(hook); nh += 1
    print(f"registered Apex hooks on {nh} TempAttention modules, beta={BETA}")

    def gauss_xyz(inp):
        g = model.forward_gaussians(inp)
        g = g.reshape(1, -1, *g.shape[1:])      # [1, T, N, 14]
        return g[0, :, :, :3].float().cpu().numpy()   # [T, N, 3]

    for name, uid in PICKS.items():
        seq = os.path.join("data/actionbench/data", uid)
        if not os.path.isdir(os.path.join(seq, "imgs")):
            print(f"[skip] {name}: no imgs for {uid[:20]}")
            continue
        print(f"[l4gm] {name} ({uid[:20]})")
        inp, rgbs, _ = build_input(seq, NUM_FRAMES, opt.num_input_views, opt, device, rays)
        state["A"] = dino_sim(dino, proc, rgbs, device)
        with torch.inference_mode():
            state["on"] = False; base = gauss_xyz(inp)
            state["on"] = True;  apex = gauss_xyz(inp)
        make_clip(base, apex, name)
    print("DONE", OUT)


if __name__ == "__main__":
    main()
