#!/bin/bash
# End-to-end reproduction entry point. Assumes scripts/download_assets.sh has
# populated data/ and pretrained-model/, and that read-out / Apex mesh outputs
# have been produced (run_readout.py / run_apex.py, or the cluster sbatch in slurm/).
set -euo pipefail
GT=${GT:-data/actionbench/gt}            # per-frame GT meshes per sequence

echo "== Table 1: Chamfer family =="
python scripts/eval_benchmark.py --pred outputs/apex          --gt "$GT" --metric chamfer
python scripts/eval_benchmark.py --pred outputs/readout_triposg --gt "$GT" --metric chamfer
python scripts/eval_benchmark.py --pred outputs/readout_step1x  --gt "$GT" --metric chamfer

echo "== Per-sequence Apex reduction (Fig.) =="
python scripts/eval_apex_per_sequence.py

echo "== Project-page videos =="
python scripts/make_videos.py --manifest configs/page_videos.txt --out docs/static/videos

echo "All reproduction steps issued. See REPRODUCE.md for the table/figure mapping."
