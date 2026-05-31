#!/bin/bash
# Download the benchmark and the frozen checkpoints. No credential is ever stored
# in the repository; export HF_TOKEN in your shell if a model needs authentication.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
mkdir -p "$ROOT/data" "$ROOT/pretrained-model"

echo "[1/2] Dynamic-mesh benchmark -> data/actionbench/"
# Replace <BENCHMARK_URL> with the official benchmark release used in the paper
# (jiang2026mesh4d / yenphraphai2025shapegen4d). Example:
#   huggingface-cli download <org/benchmark> --repo-type dataset --local-dir data/actionbench
echo "    See INSTALL.md for the official benchmark source."

echo "[2/2] Frozen models -> pretrained-model/"
fetch () { echo "    huggingface-cli download $1 --local-dir pretrained-model/$2"; }
fetch "<org>/TripoSG"            triposg
fetch "<org>/Hunyuan3D-2"        hunyuan3d
fetch "<org>/Step1X-3D"          step1x3d
fetch "<org>/ActionMesh"         actionmesh
fetch "<org>/L4GM"               l4gm
fetch "facebook/dinov2-base"     dinov2-base

echo "Edit the <org> placeholders above with the official repo ids, then re-run."
echo "Done. Set decoder paths in configs/."
