"""L4GM + Apex result videos for the project page.

L4GM outputs per-frame Gaussians; gsplat rasterization does not JIT-compile in
this environment, so we visualise the Gaussian *centres* directly as a 3D point
cloud (this is exactly the quantity behind the trajectory-acceleration metric in
the appendix). For each clip we render the baseline L4GM and the same model under
the frozen-weight Apex hook side by side, with each point coloured by its
anchor-frame identity so that a stable colour field across frames means coherent
motion. No weight is updated; only the inference-time hook differs.
"""
import os, sys, glob
import numpy as np
import torch
from PIL import Image
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, "third_party/L4GM-official")
CKPT = "pretrained-model/l4gm/recon.safetensors"
IMG_ROOT = "data/actionbench/data"
OUT = "/public/home/lixuan/lixuan/Latent4D/docs/static/videos"
BETA = 0.3
N_FRAMES = 16
N_POINTS = 5000          # subsample for a clean, fast scatter
DINO = "facebook/dinov2-base"


def load_model(device):
    from core.options import config_defaults
    from core.models import LGM
    from safetensors.torch import load_file
    cfg = config_defaults["big"]
    cfg.resume = CKPT
    cfg.workspace = "/tmp/l4gm_ws"
    model = LGM(cfg)
    model.load_state_dict(load_file(CKPT), strict=False)
    return model.half().to(device).eval(), cfg


def dino_sim(image_paths, device):
    import torch.nn.functional as F
    from transformers import AutoImageProcessor, AutoModel
    proc = AutoImageProcessor.from_pretrained(DINO)
    dm = AutoModel.from_pretrained(DINO).to(device).eval()
    feats = []
    for p in image_paths:
        inp = proc(images=Image.open(p).convert("RGB"), return_tensors="pt").to(device)
        with torch.no_grad():
            feats.append(F.normalize(dm(**inp).last_hidden_state.mean(1), dim=-1))
    Fm = torch.cat(feats, 0)
    return torch.softmax(Fm @ Fm.t(), dim=-1)


def apex_hook(A, beta):
    def hook(module, inp, out):
        x = out[0] if isinstance(out, tuple) else out
        shape = x.shape
        if x.dim() == 4:
            xm = x
        elif x.dim() == 3 and x.shape[0] % N_FRAMES == 0:
            B = x.shape[0] // N_FRAMES
            xm = x.view(B, N_FRAMES, x.shape[1], x.shape[2])
        else:
            return out
        Ad = A.to(xm.dtype)
        xm = xm + beta * (torch.einsum("ts,bsvc->btvc", Ad, xm) - xm)
        x2 = xm.view(shape)
        return (x2,) + out[1:] if isinstance(out, tuple) else x2
    return hook


def gaussians_xyz(model, image_paths, device, A=None, beta=0.0):
    handles = []
    if A is not None:
        for m in model.modules():
            if type(m).__name__ == "TempAttention":
                handles.append(m.register_forward_hook(apex_hook(A, beta)))
    imgs = [np.asarray(Image.open(p).convert("RGB").resize((256, 256))) / 255.0
            for p in image_paths]
    x = torch.tensor(np.stack(imgs), dtype=torch.float32).permute(0, 3, 1, 2)
    x = x.unsqueeze(0).half().to(device)
    with torch.no_grad():
        g = model.forward_gaussians(x)[0].float().cpu().numpy()
    for h in handles:
        h.remove()
    N = g.shape[0] // N_FRAMES
    return g[:, :3].reshape(N_FRAMES, N, 3)


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


def make_clip(base, apex, name):
    idx = np.random.RandomState(0).choice(base.shape[1], min(N_POINTS, base.shape[1]), replace=False)
    base, apex = base[:, idx], apex[:, idx]
    colors = identity_colors(base[0])
    c = base[0].mean(0)
    base = base - c; apex = apex - c
    lim = float(np.abs(base[0]).max()) * 1.3 + 1e-6
    frames = [scatter_frame(base[t], apex[t], colors, lim) for t in range(N_FRAMES)]
    import imageio
    seq = frames + frames[-2:0:-1]
    out = os.path.join(OUT, f"{name}.mp4")
    imageio.mimsave(out, seq, fps=12, quality=8, macro_block_size=8)
    imageio.imwrite(out.replace(".mp4", ".jpg"), seq[len(frames) // 2])
    print("  saved", out)


def main():
    os.makedirs(OUT, exist_ok=True)
    device = "cuda"
    model, _ = load_model(device)
    picks = {
        "l4gm_running_humanoid": "000-031_cc1f905b148c4378ad46a40da72e839f",
        "l4gm_quadruped":        "000-004_ab9192b6bc8f49e3baed63e984c7073a",
        "l4gm_character":        "000-128_030d9e43df9840dbb57f599843d6953a",
    }
    for name, uid in picks.items():
        sd = os.path.join(IMG_ROOT, uid, "imgs")
        imgs = sorted(glob.glob(os.path.join(sd, "*.png")))[:N_FRAMES]
        if len(imgs) < N_FRAMES:
            print(f"[skip] {name}: only {len(imgs)} frames")
            continue
        print(f"[l4gm] {name} ({uid[:20]})")
        A = dino_sim(imgs, device)
        base = gaussians_xyz(model, imgs, device, A=None)
        apex = gaussians_xyz(model, imgs, device, A=A, beta=BETA)
        make_clip(base, apex, name)
    print("DONE", OUT)


if __name__ == "__main__":
    main()
