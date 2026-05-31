"""Final qualitative comparison figures for Path A (DAVIS hard scenes)
and Path C (text-conditioned 4D, hard prompts).

Layout (6 columns per row):

    input @ t=0  | ActionMesh @ 0  | + Apex @ 0  ||  input @ t0  | ActionMesh @ t0 | + Apex @ t0

The right-half time index t0 is the timestamp at which the baseline
visibly drifts and the Apex variant most clearly preserves shape and
vertex identity (selected per scene from the preview job). Vertices
are coloured by the anchor-frame spherical coordinate; identical
colour patterns across t signify preserved identity.
"""
from __future__ import annotations
import argparse, glob, os, sys
from pathlib import Path

import numpy as np
import torch
import trimesh
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import hsv_to_rgb

from pytorch3d.structures import Meshes
from pytorch3d.renderer import (
    FoVPerspectiveCameras, PointLights, RasterizationSettings,
    MeshRenderer, MeshRasterizer, SoftPhongShader, TexturesVertex,
    look_at_view_transform,
)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def vertex_identity_colors(v0: np.ndarray) -> np.ndarray:
    v = v0 - v0.mean(0)
    v = v / (np.abs(v).max() + 1e-8)
    az = np.arctan2(v[:, 2], v[:, 0]) / (2 * np.pi) + 0.5
    el = np.arctan2(v[:, 1], np.sqrt(v[:, 0]**2 + v[:, 2]**2)) / np.pi + 0.5
    hsv = np.stack([az, np.full_like(az, 0.85), 0.45 + 0.45 * el], -1)
    return hsv_to_rgb(hsv).astype(np.float32)


def render_mesh(mesh: trimesh.Trimesh, colors: np.ndarray,
                image_size: int = 360,
                dist: float = 2.9, elev: float = 12.0, azim: float = 32.0) -> np.ndarray:
    # Normalize mesh to fit inside unit half-extent so dist/fov below
    # always frames the entire object with ~15% margin.
    v = np.asarray(mesh.vertices, dtype=np.float32)
    center = (v.max(0) + v.min(0)) / 2.0
    v = v - center
    half = np.abs(v).max() + 1e-8
    v = v / half
    f = np.asarray(mesh.faces, dtype=np.int64)
    verts = torch.from_numpy(v).to(DEVICE)
    faces = torch.from_numpy(f).to(DEVICE)
    col = torch.from_numpy(colors).to(DEVICE)
    tex = TexturesVertex(verts_features=col.unsqueeze(0))
    pm = Meshes(verts=[verts], faces=[faces], textures=tex)
    R, T = look_at_view_transform(dist=dist, elev=elev, azim=azim)
    # FoV chosen so half-height at z=0 plane is ~1.32 (15% margin on
    # the worst-case unit-half-extent object): tan(fov/2) = 1.32 / dist.
    cam = FoVPerspectiveCameras(device=DEVICE, R=R, T=T, fov=49)
    raster = RasterizationSettings(image_size=image_size, faces_per_pixel=1,
                                   blur_radius=0.0, max_faces_per_bin=50000)
    rast = MeshRasterizer(cameras=cam, raster_settings=raster)
    lights = PointLights(device=DEVICE,
                         location=[[1.6, 1.6, 2.0]],
                         ambient_color=[[0.55, 0.55, 0.55]],
                         diffuse_color=[[0.55, 0.55, 0.55]],
                         specular_color=[[0.10, 0.10, 0.10]])
    shader = SoftPhongShader(device=DEVICE, cameras=cam, lights=lights)
    renderer = MeshRenderer(rasterizer=rast, shader=shader)
    out = renderer(pm)[0, ..., :3].clamp(0, 1).cpu().numpy()
    sil = (rast(pm).pix_to_face[0, ..., 0] >= 0).cpu().numpy()
    bg = np.ones_like(out)
    bg[sil] = out[sil]
    return (bg * 255).astype(np.uint8)


def load_input_square(path: str, size: int = 360) -> np.ndarray:
    img = Image.open(path).convert("RGB")
    w, h = img.size
    s = min(w, h)
    img = img.crop(((w - s) // 2, (h - s) // 2, (w + s) // 2, (h + s) // 2))
    return np.asarray(img.resize((size, size), Image.LANCZOS))


def render_mesh_pair(base_dir: Path, apex_dir: Path, t_idx: int, image_size: int = 360):
    base_files = sorted(glob.glob(str(base_dir / "mesh_*.glb")))
    apex_files = sorted(glob.glob(str(apex_dir / "mesh_*.glb")))
    bm = trimesh.load(base_files[t_idx], force="mesh")
    am = trimesh.load(apex_files[t_idx], force="mesh")
    # Anchor each variant's vertex-identity colour on its own t=0 mesh
    bm0 = trimesh.load(base_files[0], force="mesh")
    am0 = trimesh.load(apex_files[0], force="mesh")
    cb = vertex_identity_colors(np.asarray(bm0.vertices))
    ca = vertex_identity_colors(np.asarray(am0.vertices))
    return render_mesh(bm, cb, image_size), render_mesh(am, ca, image_size)


def davis_input_at(davis_root: str, scene: str, t_idx_in_16: int,
                   size: int = 360) -> np.ndarray:
    jpgs = sorted(glob.glob(f"{davis_root}/JPEGImages/480p/{scene}/*.jpg"))
    sample = np.linspace(0, len(jpgs) - 1, num=16).astype(int)
    return load_input_square(jpgs[sample[t_idx_in_16]], size)


def cond4d_input_at(cond4d_base: str, slug: str, t_idx_in_16: int,
                    size: int = 360) -> np.ndarray:
    rgba_files = sorted(glob.glob(f"{cond4d_base}/{slug}/rgba/*.png"))
    arr = np.asarray(Image.open(rgba_files[t_idx_in_16]).convert("RGBA"))
    a = arr[..., 3:4] / 255.0
    bg = np.full_like(arr[..., :3], 255)
    flat = (arr[..., :3] * a + bg * (1 - a)).astype(np.uint8)
    h, w = flat.shape[:2]; s = min(h, w)
    flat = flat[(h-s)//2:(h+s)//2, (w-s)//2:(w+s)//2]
    return np.asarray(Image.fromarray(flat).resize((size, size), Image.LANCZOS))


def compose_grid(rows: list[dict], out_path: Path, tile: float = 1.45):
    n_rows = len(rows)
    fig = plt.figure(figsize=(6 * tile + 0.35, n_rows * tile + 0.55))
    gs = gridspec.GridSpec(n_rows, 6, fig, wspace=0.04, hspace=0.06,
                           left=0.06, right=0.995, top=0.93, bottom=0.02)
    col_titles = [
        "input, $t{=}0$",
        "ActionMesh, $t{=}0$",
        "+ Apex (ours), $t{=}0$",
        "input, $t{=}t_0$",
        "ActionMesh, $t{=}t_0$",
        "+ Apex (ours), $t{=}t_0$",
    ]
    for i, row in enumerate(rows):
        tiles = [row["in0"], row["base0"], row["apex0"],
                 row["inT"], row["baseT"], row["apexT"]]
        for j, im in enumerate(tiles):
            ax = fig.add_subplot(gs[i, j])
            ax.imshow(im); ax.set_xticks([]); ax.set_yticks([])
            for sp in ax.spines.values():
                sp.set_visible(False)
            if i == 0:
                title = col_titles[j]
                weight = "bold" if j == 2 or j == 5 else "normal"
                ax.set_title(title, fontsize=9.5, pad=4, weight=weight)
        # Row label with t0 annotation
        ax_first = fig.axes[i * 6]
        label = row["label"]
        if "t0_str" in row:
            label = f"{label}\n($t_0{{=}}{row['t0_str']}$)"
        ax_first.text(-0.09, 0.5, label, transform=ax_first.transAxes,
                       rotation=90, va="center", ha="right",
                       fontsize=10, weight="bold")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    fig.savefig(str(out_path).replace(".pdf", ".png"), dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fig_root", default="outputs/figure_meshes",
                     help="Directory of bg-stripped re-runs from figure_meshes.py. "
                          "Each scene/<name>/{baseline,apex}/meshes/*.glb and "
                          "<name>/input/*.png.")
    ap.add_argument("--out_dir", default="outputs/qualitative")
    args = ap.parse_args()
    fig_root = Path(args.fig_root)
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)

    # Final picks made from the preview job (84800). Each tuple is
    # (scene_key, display label, t0). The four DAVIS scenes were
    # chosen as the cases where the ActionMesh baseline visibly loses
    # geometric coherence in the second half of the clip while the
    # Apex variant preserves both silhouette and vertex-identity
    # colouring; the three Wan2.1 prompts were chosen analogously.
    # All seven scenes come from outputs/figure_meshes, which re-runs
    # ActionMesh + Apex on background-stripped (white-background) RGBA
    # inputs so the encoder is not distracted by background clutter.
    davis_picks = [
        ("camel",           "camel",           15),
        ("horsejump-high",  "horsejump-high",  15),
        ("pigs",            "pigs",            15),
        ("parkour",         "parkour",         15),
    ]
    rows = []
    for scene, label, t0 in davis_picks:
        in_dir = fig_root / scene / "input"
        in0 = load_input_square(str(sorted(in_dir.glob("*.png"))[0]))
        inT = load_input_square(str(sorted(in_dir.glob("*.png"))[t0]))
        base0, apex0 = render_mesh_pair(
            fig_root / scene / "baseline" / "meshes",
            fig_root / scene / "apex" / "meshes",
            0)
        baseT, apexT = render_mesh_pair(
            fig_root / scene / "baseline" / "meshes",
            fig_root / scene / "apex" / "meshes",
            t0)
        rows.append(dict(label=label, t0_str=str(t0),
                          in0=in0, base0=base0, apex0=apex0,
                          inT=inT, baseT=baseT, apexT=apexT))
        print(f"  davis: {scene}")
    compose_grid(rows, out_dir / "fig_davis_qualitative.pdf")

    cond4d_picks = [("polar-bear", 15), ("panda", 15), ("elephant", 15)]
    rows = []
    for label, t0 in cond4d_picks:
        in_dir = fig_root / label / "input"
        in0 = load_input_square(str(sorted(in_dir.glob("*.png"))[0]))
        inT = load_input_square(str(sorted(in_dir.glob("*.png"))[t0]))
        base0, apex0 = render_mesh_pair(
            fig_root / label / "baseline" / "meshes",
            fig_root / label / "apex" / "meshes",
            0)
        baseT, apexT = render_mesh_pair(
            fig_root / label / "baseline" / "meshes",
            fig_root / label / "apex" / "meshes",
            t0)
        rows.append(dict(label=label, t0_str=str(t0),
                          in0=in0, base0=base0, apex0=apex0,
                          inT=inT, baseT=baseT, apexT=apexT))
        print(f"  cond4d: {label}")
    compose_grid(rows, out_dir / "fig_cond4d_qualitative.pdf")


if __name__ == "__main__":
    main()
