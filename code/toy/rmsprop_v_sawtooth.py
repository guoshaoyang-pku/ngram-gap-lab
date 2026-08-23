"""Controlled test: does RMSprop dense v-decay cause the epoch sawtooth?

Reproduces the table-optimizer dynamics of the sample285 n-gram runs
(ngram_rms.py kernels, copied verbatim) on a minimal memorization task:

  - N contexts with the real per-epoch frequency distribution (72% hapax)
  - table row h_i (RMSprop, lr=0.2, beta2=0.999, wd=0.0 -- the real config)
  - hash collisions: ~4 contexts share one bucket row, like the real runs
    (22.7M contexts -> 5M buckets); a stale row gets overwritten by its
    colliding contexts, which is the table-side forgetting channel
  - shared readout W (AdamW lr=0.004, betas=(0.8,0.95), wd=0.1) whose drift
    adds the trunk-mediated forgetting channel
  - 285 steps/epoch, fresh reshuffle each epoch, pre-update loss logging

Arms:
  A global_bias   dense v decay every step (the real runs' mode)
  B rowwise_bias  lazy v: untouched rows keep v bitwise (no compression)
  C sgd           no v at all, same lr
  D global_fixed  global_bias but identical batch order every epoch
  E global_frozen global_bias but W frozen (no readout drift)
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "nanogpt_gap_onset_source"))
from ngram_rms import rmsprop_global_bias_step_, rmsprop_rowwise_bias_step_  # noqa: E402

# ---------------- config ----------------
N_CONTEXTS = 120_000
N_BUCKETS = 30_000      # ~4 contexts/bucket, as in the real 22.7M -> 5M hash
D_MODEL = 128
V_OUT = 512
STEPS_PER_EPOCH = 285
N_EPOCHS = 7
TABLE_LR = 0.2          # embedding_lr * dmodel_lr_scale (d=768 -> scale 1)
BETA2 = 0.999
EPS = 1e-10
TABLE_WD = 0.0          # real config: tables have weight_decay=0
W_LR = 0.004
W_BETAS = (0.8, 0.95)
W_WD = 0.1
SEED = 0
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"

FREQ_JSON = REPO / "remote_training_runs/ngram5_baseline_sample285_v3_final/exact_frequency_distribution.json"


def build_contexts() -> torch.Tensor:
    """Per-context instance counts matching the real frequency histogram."""
    dist = json.load(open(FREQ_JSON))["rows"]
    frac = {r["frequency"]: r["fraction"] for r in dist if r["frequency"] <= 50}
    total = sum(frac.values())
    counts = []
    for f, fr in frac.items():
        counts += [f] * max(1, round(N_CONTEXTS * fr / total))
    return torch.tensor(counts, dtype=torch.long)


def run_arm(mode: str, fixed_order: bool = False, freeze_w: bool = False):
    torch.manual_seed(SEED)
    freq = build_contexts()
    n = freq.numel()
    instances = torch.repeat_interleave(torch.arange(n), freq)  # context id per instance
    M = instances.numel()
    batch_size = M // STEPS_PER_EPOCH
    print(f"[{mode}] contexts={n} instances/epoch={M} batch={batch_size}", flush=True)

    g = torch.Generator().manual_seed(SEED)
    targets = torch.randint(0, V_OUT, (n,), generator=g)
    bucket_of = torch.randint(0, N_BUCKETS, (n,), generator=g)  # fixed random hash

    h = torch.zeros(N_BUCKETS, D_MODEL, device=DEVICE)
    W = torch.randn(V_OUT, D_MODEL, generator=g).to(DEVICE) * (D_MODEL ** -0.5)
    v = torch.zeros(N_BUCKETS, D_MODEL, device=DEVICE)
    touch_count = torch.zeros(N_BUCKETS, 1, device=DEVICE)
    # AdamW state for W
    mw = torch.zeros_like(W)
    vw = torch.zeros_like(W)
    w_step = 0

    targets_d = targets.to(DEVICE)
    instances_d = instances.to(DEVICE)
    bucket_d = bucket_of.to(DEVICE)
    last_touch = torch.full((N_BUCKETS,), -10**9, device=DEVICE)
    age_stat = {}  # age_bin -> [sum_v, sum_eff, count]

    order_rng = torch.Generator().manual_seed(SEED + 1)
    fixed_perm = torch.randperm(M, generator=order_rng)

    step_t = torch.tensor(0.0, device=DEVICE)
    lr_t = torch.tensor(TABLE_LR, device=DEVICE)
    beta2_t = torch.tensor(BETA2, device=DEVICE)
    eps_t = torch.tensor(EPS, device=DEVICE)
    wd_t = torch.tensor(TABLE_WD, device=DEVICE)

    log = {"mode": mode, "steps": [], "probe": []}
    global_step = 0
    t0 = time.time()
    for epoch in range(1, N_EPOCHS + 1):
        perm = fixed_perm if fixed_order else torch.randperm(M, generator=order_rng)
        for b in range(STEPS_PER_EPOCH):
            global_step += 1
            idx = instances_d[perm[b * batch_size:(b + 1) * batch_size].to(DEVICE)]
            buckets = bucket_d[idx]
            y = targets_d[idx]

            # ---- forward (pre-update loss, like the real trainer) ----
            rows = h[buckets]                  # index_select
            logits = rows @ W.T
            logp = torch.log_softmax(logits.float(), dim=-1)
            loss = -logp[torch.arange(idx.numel(), device=DEVICE), y].mean()
            log["steps"].append({"step": global_step, "epoch": epoch, "loss": loss.item()})

            # ---- analytic grads ----
            p = logp.exp()
            p[torch.arange(idx.numel(), device=DEVICE), y] -= 1.0
            p /= idx.numel()
            grad_rows = p.to(h.dtype) @ W          # [B, d]
            grad_h = torch.zeros_like(h)
            grad_h.index_add_(0, buckets, grad_rows)
            grad_W = p.to(W.dtype).T @ rows        # [V, d]

            # ---- probe: v at re-touch and effective step vs age ----
            touched = torch.nonzero(grad_h.abs().sum(dim=1) > 0).flatten()
            ages = (global_step - last_touch[touched] - 1).clamp(min=0)
            if mode == "rowwise_bias":
                bias2_r = 1 - BETA2 ** touch_count[touched].clamp(min=1)
            else:
                bias2_r = torch.full((touched.numel(), 1), 1 - BETA2 ** max(global_step, 1),
                                     device=DEVICE)
            denom = (v[touched] / bias2_r).sqrt() + EPS
            amplif = (grad_h[touched].abs() / denom).mean(dim=1)  # |g|/denom: step per unit lr
            v_mean = v[touched].mean(dim=1)
            for a, vv, ee in zip(ages.tolist(), v_mean.tolist(), amplif.tolist()):
                ab = min(int(a) // 20 * 20, 560)
                s = age_stat.setdefault(ab, [0.0, 0.0, 0])
                s[0] += vv; s[1] += ee; s[2] += 1
            last_touch[touched] = global_step

            # ---- table update (real kernels) ----
            step_t.fill_(float(global_step))
            if mode == "global_bias":
                rmsprop_global_bias_step_(h, grad_h, v, step_t, lr_t, beta2_t, eps_t, wd_t)
            elif mode == "rowwise_bias":
                rmsprop_rowwise_bias_step_(h, grad_h, v, touch_count, lr_t, beta2_t, eps_t, wd_t)
            elif mode == "sgd":
                h.mul_(1 - TABLE_LR * TABLE_WD)
                h.add_(grad_h, alpha=-TABLE_LR)
            else:
                raise ValueError(mode)

            # ---- readout AdamW update ----
            if not freeze_w:
                w_step += 1
                mw.lerp_(grad_W, 1 - W_BETAS[0])
                vw.lerp_(grad_W.square(), 1 - W_BETAS[1])
                mhat = mw / (1 - W_BETAS[0] ** w_step)
                vhat = vw / (1 - W_BETAS[1] ** w_step)
                W.mul_(1 - W_LR * W_WD)
                W.add_(mhat / (vhat.sqrt() + 1e-10), alpha=-W_LR)

        print(f"[{mode}] epoch {epoch} done ({time.time()-t0:.0f}s)", flush=True)

    # ---- final per-frequency memorization ----
    with torch.no_grad():
        final = {}
        for f_lo, f_hi, name in [(1, 1, "freq1"), (2, 2, "freq2"), (5, 50, "freq5+")]:
            sel = torch.nonzero((freq >= f_lo) & (freq <= f_hi)).flatten()[:20000].to(DEVICE)
            lg = (h[bucket_d[sel]] @ W.T).float()
            lp = torch.log_softmax(lg, dim=-1)
            final[name] = (-lp[torch.arange(sel.numel(), device=DEVICE), targets_d[sel]].mean().item())
        # held-out contexts: rows never trained -> loss ~ ln(V)
        final["lnV"] = float(torch.log(torch.tensor(float(V_OUT))))
    log["final_train_loss_by_freq"] = final
    log["age_stat"] = {str(k): [s[0] / s[2], s[1] / s[2], s[2]] for k, s in sorted(age_stat.items())}
    print(f"[{mode}] final train loss by freq: {final}", flush=True)
    return log


def summarize(log):
    losses = {s["step"]: s["loss"] for s in log["steps"]}
    out = []
    for e in range(2, N_EPOCHS + 1):
        start = (e - 1) * STEPS_PER_EPOCH + 1
        end = e * STEPS_PER_EPOCH
        pre = sum(losses[s] for s in range(start - 3, start)) / 3
        post = sum(losses[s] for s in range(start, start + 3)) / 3
        early = sum(losses[s] for s in range(start, start + 8)) / 8
        late = sum(losses[s] for s in range(end - 2, end + 1)) / 3
        out.append({"epoch": e, "boundary_drop": pre - post, "within_epoch_rise": late - early})
    return out


if __name__ == "__main__":
    arms = [
        ("global_bias", {}),
        ("rowwise_bias", {}),
        ("sgd", {}),
        ("global_bias", {"fixed_order": True}),
        ("global_bias", {"freeze_w": True}),
    ]
    only = sys.argv[1] if len(sys.argv) > 1 else None
    results = {}
    for mode, kw in arms:
        name = mode + ("_fixedorder" if kw.get("fixed_order") else "_frozenW" if kw.get("freeze_w") else "")
        if only and only != name:
            continue
        log = run_arm(mode, **kw)
        log["name"] = name
        results[name] = log
        summ = summarize(log)
        for s in summ:
            print(f"  {name} epoch {s['epoch']}: boundary_drop={s['boundary_drop']:+.4f} "
                  f"within_epoch_rise={s['within_epoch_rise']:+.4f}", flush=True)
        out = REPO / "toy" / "results" / f"rmsprop_v_sawtooth_{name}.json"
        out.parent.mkdir(exist_ok=True)
        out.write_text(json.dumps(log))
        print(f"  saved {out}", flush=True)
