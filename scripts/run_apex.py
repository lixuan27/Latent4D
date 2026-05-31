"""Apply the frozen-weight Latent-4D-Apex intervention to a trained 4D backbone.

Reference driver (Algorithm 2). It builds the DINOv2 viewpoint-similarity matrix
for a clip, registers the Apex hook on the backbone's temporal-attention modules,
and runs the backbone under the intervention without updating any weight. The
backbone loader is left as an integration point because ActionMesh and L4GM ship
their own inference entry points; the tested, paper-exact wiring for each is in
scripts/exp_actionmesh_apex.py and scripts/l4gm_apex_eval.py.
"""
import argparse
import torch
from latent4d import viewpoint_similarity, attach


def load_backbone(name, device):
    raise SystemExit(
        "Plug in your trained backbone here. The paper-exact wiring for ActionMesh "
        "and L4GM is in scripts/exp_actionmesh_apex.py and scripts/l4gm_apex_eval.py; "
        "this driver shows the generic three-line hook usage below.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backbone", default="actionmesh", choices=["actionmesh", "l4gm"])
    ap.add_argument("--beta", type=float, default=0.3)
    ap.add_argument("--clip", required=True)
    ap.add_argument("--out", default="outputs/demo_apex")
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    backbone, temporal_modules, frames, dino = load_backbone(args.backbone, device)

    # --- the entire intervention, no training -------------------------------
    A_bar = viewpoint_similarity(frames, dino)               # Eq. 7-8
    hooks, handles = attach(backbone, temporal_modules, beta=args.beta)   # Eq. 9
    for h in hooks:
        h.set_clip(A_bar)
    out = backbone(frames)                                   # runs under Apex
    for hd in handles:
        hd.remove()
    print("done; wrote", args.out)


if __name__ == "__main__":
    main()
