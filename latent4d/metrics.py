"""Evaluation metrics used in the paper.

cd_3d   : per-frame symmetric Chamfer distance, averaged over the clip.
cd_4d   : Chamfer after a single rigid (Kabsch) alignment of the whole clip,
          so it penalises drift that a per-frame metric absorbs.
cd_motion : Chamfer between the predicted and ground-truth per-vertex
          displacement sets; undefined without shared vertex indexing.
trajectory_acceleration : intrinsic temporal-coherence proxy used for the
          DAVIS / text-conditioned / L4GM settings and the per-sequence figure.
best_view_miou : multi-camera best-view silhouette IoU (masks rendered elsewhere).
"""
from __future__ import annotations
import torch


def _nn_dist(a, b, chunk=4096):
    """Mean over a of min distance to b (a:[N,3], b:[M,3])."""
    out = []
    for i in range(0, a.shape[0], chunk):
        d = torch.cdist(a[i:i + chunk], b)            # [c, M]
        out.append(d.min(dim=1).values)
    return torch.cat(out)


def chamfer(a, b):
    return 0.5 * (_nn_dist(a, b).mean() + _nn_dist(b, a).mean())


def cd_3d(pred_seq, gt_seq):
    return torch.stack([chamfer(p, g) for p, g in zip(pred_seq, gt_seq)]).mean()


def _kabsch(P, Q):
    """Rigid transform aligning P to Q (both [N,3]); returns R, t."""
    pc, qc = P.mean(0), Q.mean(0)
    H = (P - pc).t() @ (Q - qc)
    U, _, Vt = torch.linalg.svd(H)
    d = torch.sign(torch.det(Vt.t() @ U.t()))
    D = torch.diag(torch.tensor([1.0, 1.0, d], device=P.device, dtype=P.dtype))
    R = Vt.t() @ D @ U.t()
    t = qc - R @ pc
    return R, t


def cd_4d(pred_seq, gt_seq):
    P = torch.cat(pred_seq, 0)
    Q = torch.cat(gt_seq, 0)
    n = min(P.shape[0], Q.shape[0])
    R, t = _kabsch(P[:n], Q[:n])
    aligned = [(R @ p.t()).t() + t for p in pred_seq]
    return cd_3d(aligned, gt_seq)


def cd_motion(pred_seq, gt_seq):
    """Chamfer between displacement sets {V_t - V_0}. Requires shared indexing."""
    pd = torch.cat([pred_seq[t] - pred_seq[0] for t in range(1, len(pred_seq))], 0)
    gd = torch.cat([gt_seq[t] - gt_seq[0] for t in range(1, len(gt_seq))], 0)
    return chamfer(pd, gd)


def trajectory_acceleration(verts_seq):
    """Scale-normalised mean second difference of vertex trajectories.

    Lower means smoother, more temporally coherent motion. This is the metric
    behind the per-sequence reduction figure (Apex improves 113/128 sequences).
    """
    V = torch.stack(verts_seq)                         # [T, N, 3]
    acc = V[2:] - 2 * V[1:-1] + V[:-2]
    scale = (V[0].max(0).values - V[0].min(0).values).norm() + 1e-9
    return (acc.norm(dim=-1).mean() / scale)


def iou(mask_a, mask_b):
    a, b = mask_a.bool(), mask_b.bool()
    inter = (a & b).sum().float()
    union = (a | b).sum().float().clamp(min=1)
    return inter / union


def best_view_miou(pred_masks, gt_masks):
    """Best-view silhouette IoU.

    pred_masks, gt_masks : [V_cams, T, H, W] boolean silhouettes rendered from
    several cameras (see latent4d.rendering.render_silhouettes). For each frame we
    take the camera with the highest IoU and average over frames.
    """
    per_frame = []
    for t in range(pred_masks.shape[1]):
        best = max(iou(pred_masks[c, t], gt_masks[c, t]) for c in range(pred_masks.shape[0]))
        per_frame.append(best)
    return torch.stack(per_frame).mean()
