# Reproducing the paper

Every quantitative and qualitative result maps to a command below. The clean
reference implementation lives in `latent4d/`; the exact scripts used to produce
the paper numbers are in `scripts/` (the `exp_*` and `eval_*` drivers are the
tested pipeline, copied verbatim from the research workspace).

## Tables

| Result | Command | Sample size |
|---|---|---|
| Table 1, Chamfer family (backbone, Apex, per-frame) | `python scripts/eval_benchmark.py --metric chamfer --split all` | 128 |
| Table 1, read-out rows | `python scripts/eval_benchmark.py --metric chamfer --decoder triposg` | 116 (A), 32 (B,C) |
| Table 2, multi-camera mIoU | `python scripts/eval_benchmark.py --metric miou` | 128 / 32 |
| Table 3, Apex early-exit (30 / 22 / 10 steps) | `python scripts/eval_apex.py --early-exit` | 128 |
| Rotation stress test | `python scripts/eval_apex.py --rotation` | 8 |

## Figures

| Figure | Command |
|---|---|
| Per-sequence Apex reduction (113/128 = 88.3%, median 31.5%) | `python scripts/eval_apex.py --per-sequence` |
| Dramatic baseline-vs-Apex galleries | `python scripts/make_videos.py --manifest configs/page_videos.txt` |
| Qualitative read-out galleries | `python scripts/make_videos.py --manifest configs/page_videos.txt` |
| Speed comparison | numbers in `configs/speed.yaml`; plotting in `scripts/plot_speed.py` |

## Cross-decoder convergence (Finding 1)

```bash
python scripts/eval_benchmark.py --metric chamfer --decoder triposg
python scripts/eval_benchmark.py --metric chamfer --decoder hunyuan3d
python scripts/eval_benchmark.py --metric chamfer --decoder step1x
```
The motion-aware Chamfer clusters in [0.390, 0.442] across the three decoders
while single-frame Chamfer spans a 36% range, the substrate signature.

## Cross-domain transfer (Appendix)

| Setting | Command |
|---|---|
| DAVIS-2017 real video | `python scripts/run_apex.py --backbone actionmesh --data davis` |
| Text-conditioned (Wan2.1 + U2Net + ActionMesh) | `python scripts/run_apex.py --backbone actionmesh --data text2v` |
| Second backbone L4GM (metric) | `python scripts/l4gm_apex_eval.py` |
| Second backbone L4GM (videos) | `python scripts/l4gm_video.py` |

L4GM outputs per-frame Gaussians; gsplat rasterization does not JIT-compile in
our environment, so the L4GM evaluation and the project-page videos use the
Gaussian centres directly (the trajectory-acceleration quantity). The Apex hook,
the DINOv2 viewpoint similarity and the single coefficient beta=0.3 are identical
to the mesh-backbone setting; the average reduction is 42.3%, matching the 42.2%
on ActionMesh.

## Notes on faithfulness

- Hyperparameters are fixed throughout: `K=20, eta=0.7, eps=1e-6, delta=0.05,
  alpha=0.5`, Apex `beta` fixed at inference (0.3 on L4GM).
- The frozen read-out Chamfer rows use the subset with ground-truth point clouds
  (116 for decoder A, 32 for B and C); the mIoU and early-exit rows use 128. This
  matches the sample-size table in the paper appendix.
