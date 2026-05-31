<div align="center">

# 4D Representation is Already Inside Your 3D Decoder

### Latent-4D: reading a temporally coherent mesh sequence out of a frozen image-to-3D decoder, with no temporal training

[Project page](https://lixuan27.github.io/Latent4D/) &nbsp;|&nbsp; [Paper](https://github.com/lixuan27/Latent-4D) &nbsp;|&nbsp; [Code](https://github.com/lixuan27/Latent4D)

</div>

---

## Overview

A modern single-image 3D decoder is a continuous map from a latent code and a
query point to a signed distance, and its spatial gradient sits inside a small,
stable bounded-Lipschitz envelope. That bound is enough to **advect an anchor
mesh through the signed-distance field of every video frame with twenty small
Newton steps**, producing a mesh sequence with shared vertex indexing and *no
temporal training*. We call this read-out procedure **Latent-4D**.

On top of it we provide two frozen-weight add-ons:

- **Latent-4D-Apex**: an inference-time hook that modulates the temporal block of
  a trained 4D backbone (ActionMesh, L4GM) using DINOv2 viewpoint similarity,
  reducing motion-aware Chamfer by 42% without updating any weight.
- **Tangential refinement**: an optional pass that recovers in-surface motion
  from the same frozen visual features.

Headline results on the public dynamic-mesh benchmark: frozen read-out at
**0.80 mIoU** with exact cross-frame correspondence and **4.96 s per 16-frame
clip** (at least 6x faster than published 4D pipelines); Apex improves
**113 / 128 sequences (88.3%)** in trajectory coherence.

## Repository layout

```
latent4d/            core package
  readout.py         anchor extraction + Lagrangian Newton advection + latent smoother (Alg. 1)
  apex.py            inference-time temporal modulation hook (Alg. 2)
  tangential.py      tangential refinement via frozen DINOv2 features (Alg. 3)
  decoders/          adapters exposing s(x; z) for TripoSG / Hunyuan3D / Step1X-3D
  metrics.py         cd_3d / cd_4d / cd_motion, multi-camera mIoU, trajectory acceleration
  rendering.py       mesh -> frames / video, anchor-frame vertex-identity colour
scripts/             entry points and SLURM templates (run, eval, render, videos)
configs/             one yaml per experiment (paths + hyperparameters)
docs/                the GitHub Pages project homepage
REPRODUCE.md         exact command -> paper Table/Figure mapping
INSTALL.md           environment + asset download
```

## Install

See [INSTALL.md](INSTALL.md). In short:

```bash
conda create -n latent4d python=3.10 -y && conda activate latent4d
pip install -r requirements.txt
bash scripts/download_assets.sh        # benchmark + decoder / backbone checkpoints
```

## Quickstart

Read a 4D mesh sequence out of a frozen decoder for one clip:

```bash
python scripts/run_readout.py --decoder triposg \
  --clip data/actionbench/data/000-031_*/imgs --out outputs/demo
```

Apply the Apex intervention to a trained backbone:

```bash
python scripts/run_apex.py --backbone actionmesh --beta 0.3 \
  --clip data/actionbench/data/000-031_*/imgs --out outputs/demo_apex
```

Render a result video:

```bash
python scripts/make_videos.py --manifest configs/page_videos.txt --out docs/static/videos
```

## Reproducing the paper

Every table and figure is mapped to a command in [REPRODUCE.md](REPRODUCE.md).
Summary:

| Paper item | Command | Notes |
|---|---|---|
| Table 1 (Chamfer family) | `python scripts/eval_benchmark.py --metric chamfer` | needs decoders + ActionMesh ckpt |
| Table 2 (multi-camera mIoU) | `python scripts/eval_benchmark.py --metric miou` | |
| Table 3 (Apex early-exit) | `python scripts/eval_apex.py --early-exit` | |
| Fig. accel-reduction (88.3%) | `python scripts/eval_apex.py --per-sequence` | |
| Qualitative / dramatic videos | `python scripts/make_videos.py` | |
| Cross-decoder convergence | `python scripts/eval_benchmark.py --decoders triposg hunyuan3d step1x` | |

What is fully reproducible from this repo: the read-out, the Apex and tangential
add-ons, all evaluation metrics, and every rendering. The frozen decoders
(TripoSG, Hunyuan3D, Step1X-3D), the trained backbones (ActionMesh, L4GM) and the
benchmark are downloaded from their official sources by `download_assets.sh`; we
do not redistribute their weights.

## Citation

```bibtex
@inproceedings{latent4d2026,
  title     = {4D Representation is Already Inside Your 3D Decoder},
  author    = {Latent-4D Authors},
  booktitle = {Conference on Language Modeling (COLM)},
  year      = {2026}
}
```

## License

Code in this repository is released under the MIT License (see [LICENSE](LICENSE)).
Third-party models retain their own licenses.
