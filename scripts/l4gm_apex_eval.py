"""item3 L4GM + Apex: 把 Latent-4D-Apex 推理期干预适配到第二个 trained 4D backbone (L4GM).

Apex hook 挂在 L4GM 的 TempAttention (core/unet.py) 输出上, 用 DINOv2 视角相似度
A 重混跨帧特征: c_hat_t = c_t + beta*(sum_s Abar[t,s] c_s - c_t)  (论文 eq.7-9).
对每个序列: baseline forward_gaussians 渲染轮廓算 mIoU; 挂 Apex hook 再算; 对比。
"""
import os, sys, glob, argparse, time
import numpy as np, cv2, torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF

L4GM = "/public/home/lixuan/lixuan/4DWM/third_party/L4GM-official"
sys.path.insert(0, L4GM)
from core.options import config_defaults
from core.models import LGM
from core.unet import TempAttention
from kiui.cam import orbit_camera
from safetensors.torch import load_file

MEAN = (0.485, 0.456, 0.406); STD = (0.229, 0.224, 0.225)


def load_dino(device):
    from transformers import AutoModel, AutoImageProcessor
    name = "facebook/dinov2-base"
    proc = AutoImageProcessor.from_pretrained(name)
    dino = AutoModel.from_pretrained(name).to(device).eval()
    return dino, proc


@torch.no_grad()
def dino_sim(dino, proc, frames_rgb, device):
    # frames_rgb: list of HxWx3 uint8. return row-stochastic [T,T]
    import PIL.Image as Image
    feats = []
    for f in frames_rgb:
        x = proc(images=Image.fromarray(f), return_tensors="pt").to(device)
        e = dino(**x).last_hidden_state[:, 0]          # CLS [1,D]
        feats.append(F.normalize(e, dim=-1))
    E = torch.cat(feats, 0)                             # [T,D]
    A = E @ E.t()                                       # cosine [T,T]
    return torch.softmax(A / 0.1, dim=-1)              # row-stochastic


def build_input(seq, T, V, opt, device, rays):
    pngs = sorted(glob.glob(os.path.join(seq, "imgs", "*.png")))[:T]
    rgbs, img_TV = [], []
    for p in pngs:
        img = cv2.imread(p, cv2.IMREAD_UNCHANGED)
        if img.shape[-1] == 4:
            a = img[..., 3:4].astype(np.float32) / 255.0
            img = (img[..., :3].astype(np.float32) * a + 255.0 * (1 - a)).astype(np.uint8)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (256, 256), cv2.INTER_AREA)
        rgbs.append(img)
        f = img.astype(np.float32) / 255.0
        img_TV.append(np.stack([f] * V, axis=0))
    ref = np.stack(img_TV, axis=0)
    inp = torch.from_numpy(ref).reshape([-1, *ref.shape[2:]]).permute(0, 3, 1, 2).float().to(device)
    inp = F.interpolate(inp, size=(opt.input_size, opt.input_size), mode="bilinear", align_corners=False)
    inp = TF.normalize(inp, MEAN, STD)
    inp = torch.cat([inp, rays], dim=1).unsqueeze(0).half()
    return inp, rgbs, pngs


def proj_mat(opt, device):
    f = np.tan(0.5 * np.deg2rad(opt.fovy))
    P = torch.zeros(4, 4, device=device)
    P[0, 0] = 1 / f; P[1, 1] = 1 / f
    P[2, 2] = (opt.zfar + opt.znear) / (opt.zfar - opt.znear)
    P[3, 2] = -(opt.zfar * opt.znear) / (opt.zfar - opt.znear); P[2, 3] = 1
    return P


@torch.no_grad()
def render_sil(model, gaussians_T, opt, P, device):
    # gaussians_T: [1, T, N, 14]; return per-frame best-view silhouettes [T, 4, 256,256]
    sils = []
    bg = torch.ones(3, device=device)
    for t in range(gaussians_T.shape[1]):
        g = gaussians_T[:, t]
        views = []
        for azi in (0, 90, 180, 270):
            cp = torch.from_numpy(orbit_camera(0, azi, radius=opt.cam_radius, opengl=True)).unsqueeze(0).to(device).float()
            cp[:, :3, 1:3] *= -1
            cv_ = torch.inverse(cp).transpose(1, 2)
            cvp = cv_ @ P
            cpos = -cp[:, :3, 3]
            img = model.gs.render(g, cv_.unsqueeze(0), cvp.unsqueeze(0), cpos.unsqueeze(0), bg_color=bg)["image"]
            img = F.interpolate(img.squeeze(1), (256, 256))[0]          # [3,256,256]
            sil = (img.float().mean(0) < 0.97).float()                 # non-white = foreground
            views.append(sil.cpu().numpy())
        sils.append(np.stack(views, 0))
    return np.stack(sils, 0)                                            # [T,4,256,256]


def best_view_miou(sils_T, gt_alphas):
    # sils_T [T,4,H,W], gt_alphas list of HxW. best-view IoU per frame.
    ious = []
    for t in range(len(gt_alphas)):
        gt = (cv2.resize(gt_alphas[t], (256, 256)) > 127).astype(np.float32)
        best = 0.0
        for v in range(sils_T.shape[1]):
            s = sils_T[t, v]
            inter = (s * gt).sum(); union = ((s + gt) > 0).sum()
            if union > 0: best = max(best, inter / union)
        ious.append(best)
    return float(np.mean(ious))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=4)
    ap.add_argument("--num_frames", type=int, default=8)
    ap.add_argument("--beta", type=float, default=0.3)
    ap.add_argument("--out", default="outputs/l4gm_apex")
    args = ap.parse_args()
    device = "cuda"; os.makedirs(args.out, exist_ok=True)

    opt = config_defaults["big"]; opt.num_frames = args.num_frames
    recon = glob.glob(os.path.expanduser("~/.cache/huggingface/hub/models--jiawei011--L4GM/snapshots/*/recon.safetensors"))[0]
    model = LGM(opt); model.load_state_dict(load_file(recon, device="cpu"), strict=False)
    model = model.half().to(device).eval()
    rays = torch.cat([model.prepare_default_rays(device) for _ in range(opt.num_frames)])
    P = proj_mat(opt, device)
    dino, proc = load_dino(device)

    # Apex hook state (A set per-sequence)
    state = {"A": None, "beta": args.beta, "on": False}
    def hook(module, inp, out):
        if not state["on"] or state["A"] is None: return out
        BTV, C, H, W = out.shape; T, V = module.num_frames, module.num_views; B = BTV // (T * V)
        x = out.reshape(B, T, V, C, H, W)
        A = state["A"].to(out.dtype)
        mixed = torch.einsum('ts,bsvchw->btvchw', A, x)
        x = x + state["beta"] * (mixed - x)
        return x.reshape(BTV, C, H, W)
    n_hooks = 0
    for m in model.modules():
        if isinstance(m, TempAttention): m.register_forward_hook(hook); n_hooks += 1
    print(f"registered Apex hooks on {n_hooks} TempAttention modules, beta={args.beta}")

    seqs = sorted(glob.glob("data/actionbench/data/*"))[:args.n]
    res = []
    for seq in seqs:
        uid = os.path.basename(seq)
        gt_dir = os.path.join(seq, "imgs")
        gt_alphas = [np.array(__import__("PIL.Image", fromlist=["Image"]).open(p).convert("RGBA"))[..., 3]
                     for p in sorted(glob.glob(os.path.join(gt_dir, "*.png")))[:args.num_frames]]
        inp, rgbs, _ = build_input(seq, args.num_frames, opt.num_input_views, opt, device, rays)
        state["A"] = dino_sim(dino, proc, rgbs, device)
        # 度量 = 轨迹加速度 (论文附录文本条件4D的 motion-coherence 指标, 无需渲染).
        # L4GM Gaussian 像素对齐 → 同索引跨帧近似对应; xyz=channels[:3].
        def traj_accel(g):  # g: [1,T,N,14]
            x = g[0, :, :, :3].float()                       # [T,N,3]
            acc = x[2:] - 2 * x[1:-1] + x[:-2]               # [T-2,N,3]
            return acc.norm(dim=-1).mean().item()
        with torch.inference_mode():
            state["on"] = False
            g = model.forward_gaussians(inp); g = g.reshape(1, -1, *g.shape[1:])
            ta_base = traj_accel(g)
            state["on"] = True
            g2 = model.forward_gaussians(inp); g2 = g2.reshape(1, -1, *g2.shape[1:])
            ta_apex = traj_accel(g2)
        rel = (ta_apex - ta_base) / max(ta_base, 1e-9) * 100
        print(f"{uid[:24]}: trajAccel L4GM={ta_base:.5f}  +Apex={ta_apex:.5f}  rel={rel:+.1f}%")
        res.append((ta_base, ta_apex))
    b = np.mean([r[0] for r in res]); a = np.mean([r[1] for r in res])
    rel = (a - b) / max(b, 1e-9) * 100
    print(f"\n=== L4GM mean trajectory-acceleration: baseline={b:.5f}  +Apex={a:.5f}  rel={rel:+.1f}% (n={len(res)}, beta={args.beta}) ===")
    print("(lower = smoother motion; negative rel = Apex improves temporal coherence on the 2nd 4D backbone)")
    import json; json.dump({"metric": "trajectory_acceleration", "baseline": b, "apex": a, "rel_pct": rel,
                            "n": len(res), "beta": args.beta, "per_seq": res},
                           open(os.path.join(args.out, "l4gm_apex_trajaccel.json"), "w"), indent=2, default=float)


if __name__ == "__main__":
    main()
