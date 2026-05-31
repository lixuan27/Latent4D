# Installation

## 1. Environment

```bash
conda create -n latent4d python=3.10 -y
conda activate latent4d
pip install -r requirements.txt
```

PyTorch3D is used for the qualitative renders and the project-page videos. Install
the build that matches your CUDA / PyTorch version following the official PyTorch3D
instructions. The core read-out and the metrics do not require PyTorch3D.

## 2. Frozen models

Latent-4D trains nothing; it reads out of frozen, officially released models.
Download the ones you need and set their paths in `configs/`.

| Role | Model | Source |
|---|---|---|
| Decoder A | TripoSG | official TripoSG release on Hugging Face |
| Decoder B | Hunyuan3D | official Hunyuan3D release on Hugging Face |
| Decoder C | Step1X-3D (2025) | official Step1X-3D release on Hugging Face |
| Trained 4D backbone | ActionMesh | official ActionMesh release |
| Second 4D backbone | L4GM | official L4GM release |
| Visual features | DINOv2 (base) | `facebook/dinov2-base` |

```bash
bash scripts/download_assets.sh        # fetches the benchmark and the above checkpoints
```

The script never stores credentials in the repository. Export `HF_TOKEN` in your
shell if a checkpoint requires authentication.

## 3. Benchmark

The public dynamic-mesh benchmark (128 sequences, 16 frames each, 512x512 RGBA
with 100k ground-truth surface points) is downloaded by `download_assets.sh` into
`data/actionbench/`. It is git-ignored and never committed.

## 4. Sanity check

```bash
python -c "import latent4d, torch; print('latent4d', latent4d.__version__, 'cuda', torch.cuda.is_available())"
python scripts/run_readout.py --decoder triposg --clip data/actionbench/data/000-031_*/imgs --out outputs/demo
```
