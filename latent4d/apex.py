"""Algorithm 2: Latent-4D-Apex inference-time temporal modulation.

We do not train anything. Given a frozen trained 4D backbone whose temporal
block carries a per-frame conditioning tensor, we replace that conditioning with
a viewpoint-aligned mixture (Eq. 7-9): build a frame-by-frame similarity matrix A
from frozen DINOv2 features, row-normalise it to A_bar, and inside the temporal
block set

    c_hat_t = c_t + beta * ( sum_{t'} A_bar[t, t'] c_{t'} - c_t ).

This is a strict generalisation of the latent smoother of Algorithm 1 and is
backbone-agnostic: it has been applied unchanged (same beta, same DINOv2) to both
ActionMesh and L4GM. The only knob is the scalar beta.
"""
from __future__ import annotations
import torch


@torch.no_grad()
def viewpoint_similarity(images, feature_extractor):
    """Row-stochastic viewpoint similarity A_bar [T, T] from frozen features.

    images            : list/iterable of T input frames.
    feature_extractor : callable image -> feature vector (e.g. frozen DINOv2).
    """
    e = torch.stack([feature_extractor(im).flatten() for im in images])   # [T, D]
    e = torch.nn.functional.normalize(e, dim=-1)
    A = e @ e.t()                                                         # cosine sim, Eq. 7
    A = A.softmax(dim=-1)                                                 # row-stochastic, Eq. 8
    return A


def couple(c, A_bar, beta, time_dim):
    """Apply c_hat = c + beta (A_bar @ c - c) along `time_dim` (Eq. 9).

    c        : conditioning tensor with a frame axis of length T at `time_dim`.
    A_bar    : [T, T] row-stochastic matrix.
    """
    c_t = c.movedim(time_dim, 0)                       # [T, ...]
    T = c_t.shape[0]
    mixed = torch.tensordot(A_bar.to(c_t), c_t.reshape(T, -1), dims=([1], [0]))
    mixed = mixed.reshape(c_t.shape)
    out = c_t + beta * (mixed - c_t)
    return out.movedim(0, time_dim)


class ApexHook:
    """Forward hook that rewrites a temporal block's conditioning in place.

    Attach to every temporal-attention module of the frozen backbone. Set
    ``A_bar`` once per clip (from :func:`viewpoint_similarity`); the hook then
    couples the module output across frames at inference time. ``output_index``
    and ``time_dim`` adapt to the backbone's tensor layout.
    """

    def __init__(self, beta=0.3, time_dim=1, output_index=None):
        self.beta = beta
        self.time_dim = time_dim
        self.output_index = output_index
        self.A_bar = None

    def set_clip(self, A_bar):
        self.A_bar = A_bar
        return self

    def __call__(self, module, inputs, output):
        if self.A_bar is None:
            return output
        if self.output_index is None:
            return couple(output, self.A_bar, self.beta, self.time_dim)
        out = list(output)
        out[self.output_index] = couple(out[self.output_index], self.A_bar,
                                        self.beta, self.time_dim)
        return type(output)(out)


def attach(backbone, modules, beta=0.3, time_dim=1, output_index=None):
    """Register an ApexHook on each module in `modules`; returns (hooks, handles).

    Typical use::

        hooks, handles = attach(backbone, temporal_blocks, beta=0.3)
        A_bar = viewpoint_similarity(frames, dino)
        for h in hooks: h.set_clip(A_bar)
        out = backbone(frames)                 # runs under the intervention
        for hd in handles: hd.remove()
    """
    hooks, handles = [], []
    for m in modules:
        h = ApexHook(beta=beta, time_dim=time_dim, output_index=output_index)
        handles.append(m.register_forward_hook(h))
        hooks.append(h)
    return hooks, handles
