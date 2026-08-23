"""Tensor-only n-gram RMSProp update kernels.

The functions are kept separate from ``train.py`` so their state semantics can
be tested on CPU without importing the CUDA-only training entry point.  The
training script wraps each function in a static ``torch.compile`` graph.
"""

from __future__ import annotations

import torch


def rmsprop_global_bias_step_(
    p,
    grad,
    exp_avg_sq,
    step_t,
    lr_t,
    beta2_t,
    eps_t,
    wd_t,
):
    """Original dense RMSProp update with one global bias-correction step."""
    p.mul_(1 - lr_t * wd_t)
    grad_for_state = grad.to(dtype=exp_avg_sq.dtype)
    exp_avg_sq.lerp_(grad_for_state.square(), 1 - beta2_t)
    bias2 = 1 - beta2_t**step_t
    denom = (exp_avg_sq / bias2).sqrt() + eps_t
    p.add_((grad_for_state / denom).to(dtype=p.dtype), alpha=-lr_t)


def rmsprop_global_no_bias_step_(
    p,
    grad,
    exp_avg_sq,
    lr_t,
    beta2_t,
    eps_t,
    wd_t,
):
    """Dense RMSProp update with no bias correction."""
    p.mul_(1 - lr_t * wd_t)
    grad_for_state = grad.to(dtype=exp_avg_sq.dtype)
    exp_avg_sq.lerp_(grad_for_state.square(), 1 - beta2_t)
    denom = exp_avg_sq.sqrt() + eps_t
    p.add_((grad_for_state / denom).to(dtype=p.dtype), alpha=-lr_t)


def rmsprop_rowwise_bias_step_(
    p,
    grad,
    exp_avg_sq,
    row_touch_count,
    lr_t,
    beta2_t,
    eps_t,
    wd_t,
):
    """Lazy row-wise RMSProp using a bias-correction clock per touched row.

    ``grad`` remains a dense tensor.  A fixed-shape mask selects rows with at
    least one non-zero gradient element; no dynamic indexing or data-dependent
    output shape is used.  Untouched parameter rows, second moments, and touch
    counts are bitwise unchanged.
    """
    if grad.ndim < 2:
        raise ValueError("rowwise_bias requires an embedding-like tensor with ndim >= 2")
    if row_touch_count.shape != (grad.shape[0],) + (1,) * (grad.ndim - 1):
        raise ValueError("row_touch_count must have shape [rows, 1, ...]")

    reduce_dims = tuple(range(1, grad.ndim))
    active_rows = grad.ne(0).any(dim=reduce_dims, keepdim=True)
    grad_for_state = grad.to(dtype=exp_avg_sq.dtype)

    candidate_v = exp_avg_sq * beta2_t + grad_for_state.square() * (1 - beta2_t)
    next_v = torch.where(active_rows, candidate_v, exp_avg_sq)
    next_touch_count = row_touch_count + active_rows.to(dtype=row_touch_count.dtype)

    bias2 = 1 - beta2_t**next_touch_count
    # Counts are zero on never-touched rows.  They are masked from the update,
    # but using a denominator of one also prevents an unused 0/0 from entering
    # the compiled graph.
    safe_bias2 = torch.where(active_rows, bias2, torch.ones_like(bias2))
    denom = (next_v / safe_bias2).sqrt() + eps_t
    adaptive_update = (grad_for_state / denom).to(dtype=p.dtype)
    candidate_p = p * (1 - lr_t * wd_t) - lr_t * adaptive_update

    exp_avg_sq.copy_(next_v)
    row_touch_count.copy_(next_touch_count)
    p.copy_(torch.where(active_rows, candidate_p, p))
