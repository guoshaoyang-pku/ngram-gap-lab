"""ngram-gap-lab · code/train.py

Minimal clean reproduction of n-gram-value-memory-induced replay-specific
train/val gap on vanilla nanoGPT.

Only three things are kept from the full OPHIS codebase:
  1. vanilla nanoGPT (NanoGPTOriginal) — Karpathy-style transformer.
  2. n-gram value table (bigram + trigram, hash + embedding + gate).
  3. Three injection points (v / y / input) and RMSProp(table)/AdamW(backbone)
     optimizer grouping.

Removed (proven unnecessary for gap):
  current shell, Muon, RoPE, RMSNorm, untied embedding, split QKV,
  x0 residual, layer pool, head gate, softcap, full theory-obs system.

Default config = baseline_input standard setting (see docs/plan.md §3.1a).

Outputs JSONL logs under data/runs/<run_id>/:
  - train_log.jsonl   : one line per step {step, train_loss, val_loss, epoch, ...}
  - summary.json      : final gap + config snapshot
  - table_norm.jsonl  : periodic table param RMS (every TABLE_NORM_INTERVAL steps)
  - online_loss.jsonl : actual writer-batch loss at every optimizer step
  - online_frequency_gap_contribution.jsonl : moving train/val bucket contribution
  - fixed_probe_frequency_gap_contribution.jsonl : fixed train/val probe contribution
  - fixed_gram_frequency_gap_contribution.jsonl : fixed bucket-stratified occurrence gap

Usage:
  python train.py --run_id myrun --injection_position input --steps 1000
  (or via env vars matching the old launcher conventions; see cluster/run_injpos.sh)
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
import random
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# 0. Config
# ---------------------------------------------------------------------------


@dataclass
class Config:
    # model
    vocab_size: int = 8192
    n_layer: int = 8
    n_head: int = 6
    n_embd: int = 768
    sequence_len: int = 2048
    dropout: float = 0.0
    bias: bool = True
    # n-gram value table
    enable_nanogpt_ngram_ve: bool = True
    enable_unigram_ve: bool = False
    enable_bigram_ve: bool = True
    enable_trigram_ve: bool = True
    enable_fourgram_ve: bool = False
    nanogpt_ngram_injection_position: str = "input"  # v | y | input
    # optimizer
    nanogpt_adam_lr: float = 0.004
    ngram_table_betas: tuple = (0.0, 0.999)
    adam_betas: tuple = (0.8, 0.95)
    weight_decay: float = 0.1
    # training
    seed: int = 42
    max_steps: int = 1000
    device_batch_size: int = 72
    total_batch_size: int = 147456
    val_interval_steps: int = 50
    val_batches: int = 4
    table_norm_interval_steps: int = 10
    warmdown_ratio: float = 0.65  # last 65% of steps decays LR
    # data (paths)
    data_dir: str = ""          # directory with train.bin / val.bin
    train_shards: list = field(default_factory=list)  # list of shard indices for train
    val_shards: list = field(default_factory=list)    # list of shard indices for val
    data_mode: str = "fixed"    # fixed = deterministic epoch replay
    data_seed: int = 42
    # output
    out_dir: str = ""


def has_ve(layer_idx: int, n_layer: int) -> bool:
    """Alternating VE layers (matches OPHIS convention)."""
    return layer_idx % 2 == (n_layer - 1) % 2


# ---------------------------------------------------------------------------
# 1. Model: vanilla nanoGPT + n-gram value table
# ---------------------------------------------------------------------------

# Hash prime families (decorrelated across layers). First 4 are historical;
# deeper models deterministically extend via expand_bigram_hash_primes.
_BASE_BIGRAM_PRIMES = [
    [(2654435761, 2246822519), (1013904223, 6291469)],
    [(374761393, 668265263), (3266489917, 104729)],
    [(1640531527, 97531), (48271, 40503)],
    [(16777619, 2166136261), (3432918353, 461845907)],
]
_BASE_TRIGRAM_PRIMES = [
    (16777619, 2166136261, 3432918353, 461845907, 2654435769, 1540483477),
    (3405403843, 2654435761, 2246822519, 1013904223, 6291469, 374761393),
    (668265263, 3266489917, 104729, 1640531527, 97531, 48271),
]


def _next_odd_hash_constant(seed, used):
    value = (seed & 0xFFFFFFFF) | 1
    while value in used:
        value = ((value + 0x9E3779B2) & 0xFFFFFFFF) | 1
    used.add(value)
    return value


def expand_bigram_hash_primes(base, count):
    if count <= len(base):
        return base[:count]
    expanded = list(base)
    used = {c for fam in expanded for pair in fam for c in pair}
    salts = (0x9E3779B1, 0x85EBCA77, 0xC2B2AE3D, 0x27D4EB2F)
    while len(expanded) < count:
        fi = len(expanded)
        b = (fi + 1) * 0xA24BAED5
        expanded.append([
            (_next_odd_hash_constant(b ^ salts[0], used),
             _next_odd_hash_constant((b + 0x632BE59B) ^ salts[1], used)),
            (_next_odd_hash_constant((b + 0x85157AF5) ^ salts[2], used),
             _next_odd_hash_constant((b + 0xC2B2AE35) ^ salts[3], used)),
        ])
    return expanded


class LayerNorm(nn.Module):
    def __init__(self, ndim, bias=True):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None

    def forward(self, x):
        return F.layer_norm(x, self.weight.shape, self.weight, self.bias, 1e-5)


class CausalSelfAttention(nn.Module):
    """nanoGPT fused-c_attn attention with optional n-gram value injection.

    Injection positions:
      v     : add gated n-gram value to V before attention  (ResFormer-style)
      y     : add gated n-gram value to attention output y  (post-attn residual)
      input : handled in GPT.forward (over-encoding to wte); gates not used here
    """

    def __init__(self, config: Config, layer_idx: int):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.head_dim = config.n_embd // config.n_head
        self.layer_idx = layer_idx
        self.ngram_injection_position = config.nanogpt_ngram_injection_position
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        self.attn_dropout_p = config.dropout
        self.resid_dropout = nn.Dropout(config.dropout)
        self.ve_gate_channels = 32
        use_ngram = config.enable_nanogpt_ngram_ve
        use_layer_ve = has_ve(layer_idx, config.n_layer)
        ve_layers = sorted(i for i in range(config.n_layer) if has_ve(i, config.n_layer))
        trigram_layers = (
            {ve_layers[0], ve_layers[-2], ve_layers[-1]}
            if len(ve_layers) >= 2 else {ve_layers[-1]}
        )
        # gates only needed when injection is v or y (input injection has no gate)
        need_gate = use_ngram and config.nanogpt_ngram_injection_position in {"v", "y"}
        self.ve_gate = (
            nn.Linear(self.ve_gate_channels, self.n_head, bias=False)
            if need_gate and config.enable_unigram_ve and use_layer_ve else None
        )
        self.bigram_gate = (
            nn.Linear(self.ve_gate_channels, self.n_head, bias=False)
            if need_gate and config.enable_bigram_ve and use_layer_ve else None
        )
        self.trigram_gate = (
            nn.Linear(self.ve_gate_channels, self.n_head, bias=False)
            if need_gate and config.enable_trigram_ve and layer_idx in trigram_layers else None
        )
        self.fourgram_gate = None  # fourgram not supported in minimal version

    def _add_value_residual(self, branch, v_heads, residual, gate, x, ch_start, heads):
        residual = residual.view(x.size(0), x.size(1), heads, self.head_dim)
        gate_input = x[..., ch_start: ch_start + self.ve_gate_channels]
        value_gate = 2 * torch.sigmoid(gate(gate_input)).to(residual.dtype)
        gated = value_gate.unsqueeze(-1) * residual
        return v_heads + gated

    def _add_ngram_residuals(self, v_heads, x, ve, bigram_ve, trigram_ve, heads):
        if ve is not None:
            v_heads = self._add_value_residual("unigram", v_heads, ve, self.ve_gate, x, 0, heads)
        if bigram_ve is not None:
            v_heads = self._add_value_residual(
                "bigram", v_heads, bigram_ve, self.bigram_gate, x, self.ve_gate_channels, heads)
        if trigram_ve is not None:
            v_heads = self._add_value_residual(
                "trigram", v_heads, trigram_ve, self.trigram_gate, x, 2 * self.ve_gate_channels, heads)
        return v_heads

    def _compute_ngram_residual_flat(self, x, ve, bigram_ve, trigram_ve, heads):
        """Post-attention (y) injection: sum gated residuals as (B,T,C)."""
        r = torch.zeros_like(x)
        if ve is not None:
            v = ve.view(x.size(0), x.size(1), heads, self.head_dim)
            g = 2 * torch.sigmoid(self.ve_gate(x[..., :self.ve_gate_channels])).to(v.dtype)
            r = r + (g.unsqueeze(-1) * v).view_as(x)
        if bigram_ve is not None:
            v = bigram_ve.view(x.size(0), x.size(1), heads, self.head_dim)
            g = 2 * torch.sigmoid(self.bigram_gate(
                x[..., self.ve_gate_channels:2 * self.ve_gate_channels])).to(v.dtype)
            r = r + (g.unsqueeze(-1) * v).view_as(x)
        if trigram_ve is not None:
            v = trigram_ve.view(x.size(0), x.size(1), heads, self.head_dim)
            g = 2 * torch.sigmoid(self.trigram_gate(
                x[..., 2 * self.ve_gate_channels:3 * self.ve_gate_channels])).to(v.dtype)
            r = r + (g.unsqueeze(-1) * v).view_as(x)
        return r

    def forward(self, x, ve=None, bigram_ve=None, trigram_ve=None):
        B, T, C = x.size()
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        v_heads = v.view(B, T, self.n_head, self.head_dim)
        if self.ngram_injection_position == "v":
            v_heads = self._add_ngram_residuals(v_heads, x, ve, bigram_ve, trigram_ve, self.n_head)
        v = v_heads.reshape(B, T, C)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        y = F.scaled_dot_product_attention(
            q, k, v, attn_mask=None,
            dropout_p=self.attn_dropout_p if self.training else 0.0, is_causal=True)
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        if self.ngram_injection_position == "y":
            ngram_res = self._compute_ngram_residual_flat(x, ve, bigram_ve, trigram_ve, self.n_head)
            y = y + ngram_res
        y = self.resid_dropout(self.c_proj(y))
        return y


class MLP(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
        self.gelu = nn.GELU()
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        return self.dropout(self.c_proj(self.gelu(self.c_fc(x))))


class Block(nn.Module):
    def __init__(self, config: Config, layer_idx: int):
        super().__init__()
        self.ln_1 = LayerNorm(config.n_embd, bias=config.bias)
        self.attn = CausalSelfAttention(config, layer_idx)
        self.ln_2 = LayerNorm(config.n_embd, bias=config.bias)
        self.mlp = MLP(config)

    def forward(self, x, ve=None, bigram_ve=None, trigram_ve=None):
        x = x + self.attn(self.ln_1(x), ve=ve, bigram_ve=bigram_ve, trigram_ve=trigram_ve)
        x = x + self.mlp(self.ln_2(x))
        return x


class NanoGPT(nn.Module):
    """vanilla nanoGPT + n-gram value table (bigram/trigram)."""

    def __init__(self, config: Config):
        super().__init__()
        self.config = config
        self.transformer = nn.ModuleDict({
            "wte": nn.Embedding(config.vocab_size, config.n_embd),
            "drop": nn.Dropout(config.dropout),
            "h": nn.ModuleList([Block(config, i) for i in range(config.n_layer)]),
        })
        self.transformer["wpe"] = nn.Embedding(config.sequence_len, config.n_embd)
        self.transformer["ln_f"] = LayerNorm(config.n_embd, bias=config.bias)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        # tied embeddings (matches baseline_input setting)
        self.lm_head.weight = self.transformer.wte.weight

        ngram_layers = sorted(i for i in range(config.n_layer) if has_ve(i, config.n_layer))
        self.value_embeds = nn.ModuleDict({
            str(i): nn.Embedding(config.vocab_size, config.n_embd)
            for i in ngram_layers
            if config.enable_nanogpt_ngram_ve and config.enable_unigram_ve
        })
        self.bigram_ve_layers = (
            set(ngram_layers) if config.enable_nanogpt_ngram_ve and config.enable_bigram_ve else set()
        )
        self.bigram_table_size = config.vocab_size * 64
        self.bigram_K = 2
        half_dim = config.n_embd // 2
        _bp = expand_bigram_hash_primes(_BASE_BIGRAM_PRIMES, len(ngram_layers))
        self.bigram_hash_primes_per_layer = {}
        self.bigram_ves = nn.ModuleDict()
        for j, li in enumerate(sorted(self.bigram_ve_layers)):
            self.bigram_ves[str(li)] = nn.ModuleList([
                nn.Embedding(self.bigram_table_size, half_dim),
                nn.Embedding(self.bigram_table_size, config.n_embd - half_dim),
            ])
            self.bigram_hash_primes_per_layer[li] = _bp[j]
        self.trigram_ve_layers = (
            ({ngram_layers[0], ngram_layers[-2], ngram_layers[-1]}
             if len(ngram_layers) >= 2 else {ngram_layers[-1]})
            if config.enable_nanogpt_ngram_ve and config.enable_trigram_ve and ngram_layers
            else set()
        )
        self.trigram_table_size = config.vocab_size * 64
        _tp = _BASE_TRIGRAM_PRIMES[:max(1, len(self.trigram_ve_layers))]
        while len(_tp) < len(self.trigram_ve_layers):
            _tp.append(_tp[len(_tp) % len(_BASE_TRIGRAM_PRIMES)])
        self.trigram_hash_primes_per_layer = {}
        self.trigram_ves = nn.ModuleDict()
        for j, li in enumerate(sorted(self.trigram_ve_layers)):
            self.trigram_ves[str(li)] = nn.ModuleList([
                nn.Embedding(self.trigram_table_size, half_dim),
                nn.Embedding(self.trigram_table_size, config.n_embd - half_dim),
            ])
            self.trigram_hash_primes_per_layer[li] = _tp[j]

    @torch.no_grad()
    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                torch.nn.init.normal_(m.weight, mean=0.0, std=0.02)
                if m.bias is not None:
                    torch.nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Embedding):
                torch.nn.init.normal_(m.weight, mean=0.0, std=0.02)
        for name, p in self.named_parameters():
            if name.endswith("c_proj.weight"):
                torch.nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * self.config.n_layer))
        s = 3 ** 0.5 * self.config.n_embd ** -0.5
        for ve in self.value_embeds.values():
            torch.nn.init.uniform_(ve.weight, -s, s)
        for lvs in self.bigram_ves.values():
            for bve in lvs:
                torch.nn.init.uniform_(bve.weight, -s, s)
        for lvs in self.trigram_ves.values():
            for tve in lvs:
                torch.nn.init.uniform_(tve.weight, -s, s)
        for blk in self.transformer.h:
            for gname in ("ve_gate", "bigram_gate", "trigram_gate"):
                g = getattr(blk.attn, gname, None)
                if g is not None:
                    torch.nn.init.zeros_(g.weight)

    def _compute_input_ngram_residual(self, idx):
        """Over-encoding: sum all enabled layers' n-gram values, no gate."""
        _B, T = idx.size()
        residual = None
        prev_idx = torch.cat([idx[:, :1], idx[:, :-1]], dim=1)
        prev2_idx = torch.cat([idx[:, :2], idx[:, :-2]], dim=1)
        if self.config.enable_unigram_ve:
            for li in sorted(self.value_embeds.keys()):
                ve = self.value_embeds[li](idx)
                residual = ve if residual is None else residual + ve
        if self.config.enable_bigram_ve:
            for li in sorted(self.bigram_ve_layers):
                lvs = self.bigram_ves[str(li)]
                primes = self.bigram_hash_primes_per_layer[li]
                idxs = [((prev_idx * p1) ^ (idx * p2)) % self.bigram_table_size for p1, p2 in primes]
                bgve = torch.cat([lvs[k](idxs[k]) for k in range(self.bigram_K)], dim=-1)
                residual = bgve if residual is None else residual + bgve
        if self.config.enable_trigram_ve:
            for li in sorted(self.trigram_ve_layers):
                lp = self.trigram_hash_primes_per_layer[li]
                ti = (
                    ((prev2_idx * lp[0]) ^ (prev_idx * lp[1]) ^ (idx * lp[2])) % self.trigram_table_size,
                    ((prev2_idx * lp[3]) ^ (prev_idx * lp[4]) ^ (idx * lp[5])) % self.trigram_table_size,
                )
                lvs = self.trigram_ves[str(li)]
                tgve = torch.cat([lvs[0](ti[0]), lvs[1](ti[1])], dim=-1)
                residual = tgve if residual is None else residual + tgve
        if residual is None:
            residual = torch.zeros(idx.size(0), T, self.config.n_embd,
                                   device=idx.device, dtype=self.transformer.wte.weight.dtype)
        return residual

    def forward(self, idx, targets=None, reduction: str = "mean"):
        B, T = idx.size()
        pos = torch.arange(0, T, dtype=torch.long, device=idx.device).unsqueeze(0)
        x = self.transformer.wte(idx)
        x = x + self.transformer.wpe(pos)
        if self.config.nanogpt_ngram_injection_position == "input":
            x = x + self._compute_input_ngram_residual(idx)
        x = self.transformer.drop(x)
        prev_idx = torch.cat([idx[:, :1], idx[:, :-1]], dim=1)
        prev2_idx = torch.cat([idx[:, :2], idx[:, :-2]], dim=1)
        bigram_indices = {}
        if self.config.enable_bigram_ve:
            for li in self.bigram_ve_layers:
                bp = self.bigram_hash_primes_per_layer[li]
                bigram_indices[li] = [
                    ((prev_idx * p1) ^ (idx * p2)) % self.bigram_table_size for p1, p2 in bp
                ]
        trigram_indices = {}
        if self.config.enable_trigram_ve:
            for li in self.trigram_ve_layers:
                lp = self.trigram_hash_primes_per_layer[li]
                trigram_indices[li] = (
                    ((prev2_idx * lp[0]) ^ (prev_idx * lp[1]) ^ (idx * lp[2])) % self.trigram_table_size,
                    ((prev2_idx * lp[3]) ^ (prev_idx * lp[4]) ^ (idx * lp[5])) % self.trigram_table_size,
                )
        for i, block in enumerate(self.transformer.h):
            ve = (self.value_embeds[str(i)](idx)
                  if self.config.enable_unigram_ve and str(i) in self.value_embeds else None)
            bgve = None
            if self.config.enable_bigram_ve and i in self.bigram_ve_layers:
                lvs = self.bigram_ves[str(i)]
                bi = bigram_indices[i]
                bgve = torch.cat([lvs[k](bi[k]) for k in range(self.bigram_K)], dim=-1)
            tgve = None
            if self.config.enable_trigram_ve and i in self.trigram_ve_layers:
                ti = trigram_indices[i]
                lvs = self.trigram_ves[str(i)]
                tgve = torch.cat([lvs[0](ti[0]), lvs[1](ti[1])], dim=-1)
            x = block(x, ve=ve, bigram_ve=bgve, trigram_ve=tgve)
        x = self.transformer.ln_f(x)
        logits = self.lm_head(x)
        if targets is not None:
            loss = F.cross_entropy(
                logits.float().view(-1, logits.size(-1)),
                targets.view(-1), ignore_index=-1, reduction=reduction)
            if reduction == "none":
                loss = loss.view_as(targets)
            return loss
        return logits.float()


# ---------------------------------------------------------------------------
# 2. Optimizer: AdamW (backbone) + RMSProp (n-gram table), mixed grouping
# ---------------------------------------------------------------------------


class MixedOptimizer:
    """Minimal AdamW + RMSProp optimizer (no Muon).

    n-gram table params (value_embeds / bigram_ves / trigram_ves / *_gate)
    -> RMSProp with betas=(0.0, 0.999), no momentum, bias-corrected.
    Everything else -> AdamW with betas=(0.8, 0.95), weight_decay=0.1 (matrices only).
    """

    def __init__(self, model: nn.Module, lr: float, ngram_betas, adam_betas,
                 weight_decay: float):
        self.model = model
        self.lr = lr
        self.ngram_beta2 = ngram_betas[1]
        self.adam_betas = adam_betas
        self.weight_decay = weight_decay
        self.adam_steps = {}
        self.adam_exp_avg = {}
        self.adam_exp_avg_sq = {}
        self.rms_steps = {}
        self.rms_exp_avg_sq = {}
        ngram_markers = ("value_embeds", "bigram_ves", "trigram_ves",
                         "ve_gate", "bigram_gate", "trigram_gate")
        self.ngram_params = []
        self.adam_params = []
        for name, p in model.named_parameters():
            if not p.requires_grad:
                continue
            if any(m in name for m in ngram_markers):
                self.ngram_params.append((name, p))
            else:
                self.adam_params.append((name, p))

    def zero_grad(self, set_to_none=True):
        for _, p in self.adam_params + self.ngram_params:
            if p.grad is not None:
                if set_to_none:
                    p.grad = None
                else:
                    p.grad.zero_()

    def _adamw_step(self, name, p, lr_t):
        g = p.grad
        if g is None:
            return
        if name not in self.adam_steps:
            self.adam_steps[name] = 0
            self.adam_exp_avg[name] = torch.zeros_like(p)
            self.adam_exp_avg_sq[name] = torch.zeros_like(p)
        self.adam_steps[name] += 1
        step = self.adam_steps[name]
        b1, b2 = self.adam_betas
        exp_avg = self.adam_exp_avg[name]
        exp_avg_sq = self.adam_exp_avg_sq[name]
        # weight decay on 2D matrix params only
        is_matrix = p.dim() >= 2 and not name.endswith("wte.weight") \
            and not name.endswith("wpe.weight") and not name.endswith("lm_head.weight")
        wd = self.weight_decay if is_matrix else 0.0
        p.mul_(1 - lr_t * wd)
        exp_avg.lerp_(g, 1 - b1)
        exp_avg_sq.lerp_(g.square(), 1 - b2)
        bias1 = 1 - b1 ** step
        bias2 = 1 - b2 ** step
        denom = (exp_avg_sq / bias2).sqrt() + 1e-8
        step_size = lr_t / bias1
        p.add_(exp_avg / denom, alpha=-step_size)

    def _rmsprop_step(self, name, p, lr_t):
        g = p.grad
        if g is None:
            return
        if name not in self.rms_steps:
            self.rms_steps[name] = 0
            self.rms_exp_avg_sq[name] = torch.zeros_like(p)
        self.rms_steps[name] += 1
        step = self.rms_steps[name]
        b2 = self.ngram_beta2
        exp_avg_sq = self.rms_exp_avg_sq[name]
        # no weight decay on n-gram tables
        exp_avg_sq.lerp_(g.square(), 1 - b2)
        bias2 = 1 - b2 ** step
        denom = (exp_avg_sq / bias2).sqrt() + 1e-10
        p.add_(g / denom, alpha=-lr_t)

    def step(self, lr_mult: float = 1.0):
        lr_t = self.lr * lr_mult
        with torch.no_grad():
            for name, p in self.adam_params:
                self._adamw_step(name, p, lr_t)
            for name, p in self.ngram_params:
                self._rmsprop_step(name, p, lr_t)

    def state_dict(self):
        return {
            "adam_steps": dict(self.adam_steps),
            "rms_steps": dict(self.rms_steps),
        }


def get_lr_multiplier(progress: float, warmdown_ratio: float = 0.65) -> float:
    """Linear warmup (0->1 over first 1-warmdown), then linear decay to 0.05."""
    if progress < 1.0 - warmdown_ratio:
        # warmup phase: linear from 0.1 to 1.0
        w = progress / max(1e-6, 1.0 - warmdown_ratio)
        return 0.1 + 0.9 * w
    else:
        # warmdown: linear from 1.0 to 0.05
        w = (progress - (1.0 - warmdown_ratio)) / max(1e-6, warmdown_ratio)
        return 1.0 - 0.95 * w


# ---------------------------------------------------------------------------
# 3. Data: tokenized shard loader (fixed-order epoch replay)
# ---------------------------------------------------------------------------


class TokenizedShardDataset:
    """Loads tokenized shards as packed uint16 sequences.

    Expects data_dir to contain shard_<id>.bin files (raw uint16 token ids,
    packed sequentially, BOS-aligned chunks of sequence_len+1 each).
    Fixed mode: iterates shards in order, yielding non-overlapping chunks;
    each epoch restarts from shard 0. This gives deterministic replay.
    """

    def __init__(self, data_dir: str, shard_ids: list, sequence_len: int,
                 device_batch_size: int, seed: int = 42):
        self.data_dir = data_dir
        self.shard_ids = list(shard_ids)
        self.sequence_len = sequence_len
        self.chunk_size = sequence_len + 1
        self.device_batch_size = device_batch_size
        self.seed = seed
        self._buffers = {}
        self._epoch = 0
        # total tokens per shard (lazy)
        self._shard_lens = {}

    def _load_shard(self, sid):
        if sid not in self._buffers:
            path = os.path.join(self.data_dir, f"shard_{sid:05d}.bin")
            if not os.path.exists(path):
                raise FileNotFoundError(f"Missing shard: {path}")
            buf = np.memmap(path, dtype=np.uint16, mode="r")
            self._buffers[sid] = buf
        return self._buffers[sid]

    def shard_chunks(self, sid):
        buf = self._load_shard(sid)
        return len(buf) // self.chunk_size

    def total_chunks(self):
        return sum(self.shard_chunks(s) for s in self.shard_ids)

    def _iter_chunks(self):
        """Yield (input, target) tensors for each chunk in fixed order."""
        for sid in self.shard_ids:
            buf = self._load_shard(sid)
            n = self.shard_chunks(sid)
            for i in range(n):
                start = i * self.chunk_size
                chunk = np.array(buf[start:start + self.chunk_size], dtype=np.int64)
                inp = torch.from_numpy(chunk[:-1])
                tgt = torch.from_numpy(chunk[1:])
                yield inp, tgt

    def iter_batches(self, device: torch.device):
        """Yield (inputs, targets) batches of shape (B, T) in fixed order, looping forever."""
        batch_inp, batch_tgt = [], []
        while True:
            for inp, tgt in self._iter_chunks():
                batch_inp.append(inp)
                batch_tgt.append(tgt)
                if len(batch_inp) == self.device_batch_size:
                    yield (torch.stack(batch_inp).to(device),
                           torch.stack(batch_tgt).to(device))
                    batch_inp, batch_tgt = [], []
            self._epoch += 1
            # fixed mode: no shuffle, deterministic replay


# ---------------------------------------------------------------------------
# 4. Table-norm observable
# ---------------------------------------------------------------------------


def table_param_rms(model: NanoGPT) -> dict:
    """Compute RMS of n-gram table params (bigram/trigram layer_1 table_0)."""
    out = {}
    for li in sorted(model.bigram_ve_layers):
        lvs = model.bigram_ves[str(li)]
        rms = lvs[0].weight.detach().float().pow(2).mean().sqrt().item()
        out[f"bigram.layer_{li:02d}.table_0.rms"] = rms
    for li in sorted(model.trigram_ve_layers):
        lvs = model.trigram_ves[str(li)]
        rms = lvs[0].weight.detach().float().pow(2).mean().sqrt().item()
        out[f"trigram.layer_{li:02d}.table_0.rms"] = rms
    return out


# ---------------------------------------------------------------------------
# 5. Training loop
# ---------------------------------------------------------------------------


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def evaluate_val(model: NanoGPT, val_loader, val_batches: int) -> float:
    model.eval()
    losses = []
    with torch.no_grad():
        for _ in range(val_batches):
            inp, tgt = next(val_loader)
            loss = model(inp, targets=tgt)
            losses.append(loss.item())
    model.train()
    return float(np.mean(losses)) if losses else float("nan")


def evaluate_freq_bins(model: NanoGPT, loader, freq_index, n_batches: int,
                       vocab_size: int) -> dict:
    """Run per-frequency-bin loss accumulation on n_batches from loader.

    Returns dict with 'bigram' and 'trigram' keys, each mapping to
    {bucket_label: {frac, mean_loss, total_contrib, token_count}}.
    """
    from ngram_freq import FreqBinLossAccumulator, compute_per_token_loss
    model.eval()
    accs = {
        "bigram": FreqBinLossAccumulator(freq_index, vocab_size, "bigram"),
        "trigram": FreqBinLossAccumulator(freq_index, vocab_size, "trigram"),
    }
    with torch.no_grad():
        for _ in range(n_batches):
            inp, tgt = next(loader)
            ptl = compute_per_token_loss(model, inp, tgt)
            for branch in accs:
                accs[branch].update(inp, ptl)
    model.train()
    return {branch: acc.summary() for branch, acc in accs.items()}


def frequency_summary_from_batches(model: NanoGPT, batches, freq_index,
                                   vocab_size: int) -> tuple[float, dict]:
    """Evaluate a supplied batch collection without advancing its source iterator."""
    from ngram_freq import FreqBinLossAccumulator, compute_per_token_loss
    was_training = model.training
    model.eval()
    accs = {
        "bigram": FreqBinLossAccumulator(freq_index, vocab_size, "bigram"),
        "trigram": FreqBinLossAccumulator(freq_index, vocab_size, "trigram"),
    }
    loss_sum = 0.0
    token_count = 0
    with torch.no_grad():
        for inp, tgt in batches:
            ptl = compute_per_token_loss(model, inp, tgt)
            loss_sum += float(ptl.sum().item())
            token_count += ptl.numel()
            for acc in accs.values():
                acc.update(inp, ptl)
    if was_training:
        model.train()
    return loss_sum / max(1, token_count), {
        branch: acc.summary() for branch, acc in accs.items()
    }


def contribution_gap(train_frequency: dict, val_frequency: dict) -> dict:
    """Per-bucket mean token-loss gap, with total contribution kept as a diagnostic."""
    out = {}
    for branch, train_buckets in train_frequency.items():
        out[branch] = {}
        val_buckets = val_frequency[branch]
        for bucket, train_stats in train_buckets.items():
            val_stats = val_buckets[bucket]
            out[branch][bucket] = {
                "contribution": val_stats["total_contrib"] - train_stats["total_contrib"],
                "mean_loss_gap": val_stats["mean_loss"] - train_stats["mean_loss"],
                "train_total_contrib": train_stats["total_contrib"],
                "val_total_contrib": val_stats["total_contrib"],
                "train_frac": train_stats["frac"],
                "val_frac": val_stats["frac"],
            }
    return out


def collect_fixed_probe(dataset: TokenizedShardDataset, device: torch.device,
                        n_batches: int, offset_batches: int = 0
                        ) -> list[tuple[torch.Tensor, torch.Tensor]]:
    """Select deterministic batches once without touching the writer iterator.

    ``offset_batches`` is zero-based within the replay epoch.  It lets a
    contiguous fixed train probe be placed in the middle of an epoch instead
    of always selecting the first batches after the replay boundary.
    """
    if offset_batches < 0:
        raise ValueError("fixed probe offset must be non-negative")
    iterator = dataset.iter_batches(device)
    for _ in range(offset_batches):
        next(iterator)
    return [next(iterator) for _ in range(n_batches)]


def fixed_probe_center_steps(steps_per_epoch: int, max_steps: int,
                             offset_optimizer_steps: int) -> list[int]:
    """Return 1-based writer steps where the fixed train probe begins."""
    if steps_per_epoch <= 0 or offset_optimizer_steps < 0:
        return []
    return [epoch_start + offset_optimizer_steps
            for epoch_start in range(1, max_steps + 1, steps_per_epoch)
            if epoch_start + offset_optimizer_steps <= max_steps]


def frequency_sample_reason(step: int, steps_per_epoch: int, interval: int,
                            epoch_window: int, epoch_dense_interval: int,
                            max_steps: int, probe_centers: list[int] | None = None,
                            probe_window: int = 0,
                            probe_dense_interval: int = 1) -> Optional[str]:
    """Return why a moving/fixed frequency checkpoint is taken at this 1-based step."""
    if step == max_steps or step % interval == 0:
        return "interval"
    reasons = []
    if steps_per_epoch > 0:
        for boundary in range(steps_per_epoch, max_steps, steps_per_epoch):
            if (abs(step - boundary) <= epoch_window
                    and (step - boundary) % epoch_dense_interval == 0):
                reasons.append("epoch_dense")
                break
    for center in probe_centers or []:
        if (abs(step - center) <= probe_window
                and (step - center) % probe_dense_interval == 0):
            reasons.append("probe_dense")
            break
    return "+".join(reasons) if reasons else None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_id", default="run")
    parser.add_argument("--injection_position", default="input",
                        choices=["v", "y", "input"])
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--data_dir", default=os.environ.get("NGLAB_DATA_DIR", ""))
    parser.add_argument("--out_dir", default=os.environ.get("NGLAB_OUT_DIR", "data/runs"))
    parser.add_argument("--train_shards", default=os.environ.get("NGLAB_TRAIN_SHARDS", "1"),
                        help="comma-separated shard ids for train")
    parser.add_argument("--val_shards", default=os.environ.get("NGLAB_VAL_SHARDS", "2,3,4,5,6,7,8,9,10,6542"),
                        help="comma-separated shard ids for val")
    parser.add_argument("--device_batch_size", type=int, default=72)
    parser.add_argument("--total_batch_size", type=int, default=147456)
    parser.add_argument("--val_interval", type=int, default=50)
    parser.add_argument("--val_batches", type=int, default=4)
    parser.add_argument("--table_norm_interval", type=int, default=10)
    parser.add_argument("--lr", type=float, default=0.004)
    parser.add_argument("--enable_unigram", type=int, default=0)
    parser.add_argument("--enable_bigram", type=int, default=1)
    parser.add_argument("--enable_trigram", type=int, default=1)
    parser.add_argument("--n_layer", type=int, default=8)
    parser.add_argument("--n_head", type=int, default=6)
    parser.add_argument("--n_embd", type=int, default=768)
    parser.add_argument("--vocab_size", type=int, default=8192)
    parser.add_argument("--sequence_len", type=int, default=2048)
    parser.add_argument("--freq_index", default="",
                        help="path to freq_index.npz; if set, enables freq-bin eval")
    parser.add_argument("--freq_eval_interval", type=int, default=50)
    parser.add_argument("--freq_eval_batches", type=int, default=4)
    parser.add_argument("--legacy_freq_eval", action="store_true",
                        help="enable legacy independent diagnostic frequency evaluation")
    parser.add_argument("--fixed_gram_samples_per_bucket", type=int, default=100)
    parser.add_argument("--fixed_gram_seed", type=int, default=None,
                        help="seed for fixed gram occurrence sampling; defaults to --seed")
    parser.add_argument("--online_frequency_interval", type=int, default=50,
                        help="base interval for online/fixed contribution checkpoints")
    parser.add_argument("--online_frequency_epoch_window", type=int, default=25,
                        help="sample around each replay epoch boundary within this many steps")
    parser.add_argument("--online_frequency_dense_interval", type=int, default=1,
                        help="step spacing for epoch-boundary dense sampling")
    parser.add_argument("--online_frequency_probe_window", type=int, default=0,
                        help="sample around each fixed train-probe position within this many steps")
    parser.add_argument("--online_frequency_probe_dense_interval", type=int, default=1,
                        help="step spacing for fixed train-probe dense sampling")
    parser.add_argument("--online_frequency_val_batches", type=int, default=1,
                        help="fresh moving validation batches per online checkpoint")
    parser.add_argument("--fixed_probe_batches", type=int, default=4,
                        help="deterministic train and validation batches kept for fixed-probe checks")
    parser.add_argument("--fixed_probe_train_offset_steps", type=int, default=0,
                        help="zero-based writer-step offset of the fixed train probe in each replay epoch")
    args = parser.parse_args()

    cfg = Config(
        vocab_size=args.vocab_size,
        n_layer=args.n_layer,
        n_head=args.n_head,
        n_embd=args.n_embd,
        sequence_len=args.sequence_len,
        nanogpt_ngram_injection_position=args.injection_position,
        enable_unigram_ve=bool(args.enable_unigram),
        enable_bigram_ve=bool(args.enable_bigram),
        enable_trigram_ve=bool(args.enable_trigram),
        seed=args.seed,
        max_steps=args.steps,
        device_batch_size=args.device_batch_size,
        total_batch_size=args.total_batch_size,
        val_interval_steps=args.val_interval,
        val_batches=args.val_batches,
        table_norm_interval_steps=args.table_norm_interval,
        nanogpt_adam_lr=args.lr,
        data_dir=args.data_dir,
        train_shards=[int(x) for x in args.train_shards.split(",") if x.strip()],
        val_shards=[int(x) for x in args.val_shards.split(",") if x.strip()],
        out_dir=os.path.join(args.out_dir, args.run_id),
    )

    set_seed(cfg.seed)
    os.makedirs(cfg.out_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[nglab] device={device} injection={cfg.nanogpt_ngram_injection_position} steps={cfg.max_steps}")

    # data
    train_ds = TokenizedShardDataset(cfg.data_dir, cfg.train_shards, cfg.sequence_len,
                                     cfg.device_batch_size, cfg.data_seed)
    val_ds = TokenizedShardDataset(cfg.data_dir, cfg.val_shards, cfg.sequence_len,
                                   cfg.device_batch_size, cfg.data_seed)
    train_iter = train_ds.iter_batches(device)
    val_iter = val_ds.iter_batches(device)
    online_val_ds = TokenizedShardDataset(cfg.data_dir, cfg.val_shards, cfg.sequence_len,
                                          cfg.device_batch_size, cfg.data_seed)
    online_val_iter = online_val_ds.iter_batches(device)
    grad_accum = max(1, cfg.total_batch_size // (cfg.device_batch_size * cfg.sequence_len))
    full_train_batches_per_epoch = train_ds.total_chunks() // cfg.device_batch_size
    steps_per_epoch = full_train_batches_per_epoch // grad_accum
    fixed_probe_offset_batches = args.fixed_probe_train_offset_steps * grad_accum
    if (fixed_probe_offset_batches + args.fixed_probe_batches
            > full_train_batches_per_epoch):
        raise ValueError(
            "fixed train probe exceeds one replay epoch: "
            f"offset={fixed_probe_offset_batches}, batches={args.fixed_probe_batches}, "
            f"available={full_train_batches_per_epoch}")
    probe_center_steps = fixed_probe_center_steps(
        steps_per_epoch, cfg.max_steps, args.fixed_probe_train_offset_steps)
    print(f"[nglab] grad_accum={grad_accum} train_chunks={train_ds.total_chunks()} "
          f"val_chunks={val_ds.total_chunks()} estimated_epoch_steps={steps_per_epoch}")

    # model
    model = NanoGPT(cfg).to(device)
    model.init_weights()
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[nglab] model params: {n_params/1e6:.1f}M")
    optimizer = MixedOptimizer(model, lr=cfg.nanogpt_adam_lr,
                               ngram_betas=cfg.ngram_table_betas,
                               adam_betas=cfg.adam_betas,
                               weight_decay=cfg.weight_decay)

    # logs
    train_log = open(os.path.join(cfg.out_dir, "train_log.jsonl"), "w")
    table_log = open(os.path.join(cfg.out_dir, "table_norm.jsonl"), "w")
    online_loss_log = open(os.path.join(cfg.out_dir, "online_loss.jsonl"), "w")
    online_frequency_log = None
    fixed_probe_frequency_log = None
    fixed_gram_frequency_log = None
    freq_bin_log = None
    freq_index_obj = None
    fixed_gram_probe = None
    freq_diag_train_iter = None
    freq_diag_val_iter = None
    fixed_train_probe = []
    fixed_val_probe = []
    if args.freq_index and os.path.exists(args.freq_index):
        from ngram_freq import (
            FixedGramProbe,
            GlobalFrequencyIndex,
            build_fixed_gram_manifest,
            fixed_gram_gap_summary,
            fixed_gram_overall_loss,
            fixed_gram_manifest_matches,
            load_fixed_gram_manifest,
            save_fixed_gram_manifest,
        )
        freq_index_obj = GlobalFrequencyIndex.load(args.freq_index)
        online_frequency_log = open(
            os.path.join(cfg.out_dir, "online_frequency_gap_contribution.jsonl"), "w")
        fixed_probe_frequency_log = open(
            os.path.join(cfg.out_dir, "fixed_probe_frequency_gap_contribution.jsonl"), "w")
        fixed_gram_frequency_log = open(
            os.path.join(cfg.out_dir, "fixed_gram_frequency_gap_contribution.jsonl"), "w")
        fixed_gram_seed = cfg.seed if args.fixed_gram_seed is None else args.fixed_gram_seed
        manifest_path = os.path.join(cfg.out_dir, "fixed_gram_probe_manifest.json")
        manifest = None
        if os.path.exists(manifest_path):
            candidate_manifest = load_fixed_gram_manifest(manifest_path)
            if fixed_gram_manifest_matches(
                    candidate_manifest, cfg.train_shards, cfg.val_shards,
                    cfg.vocab_size, cfg.sequence_len,
                    args.fixed_gram_samples_per_bucket, fixed_gram_seed):
                manifest = candidate_manifest
                print(f"[nglab] reusing fixed gram manifest: {manifest_path}")
        if manifest is None:
            manifest = build_fixed_gram_manifest(
                cfg.data_dir, cfg.train_shards, cfg.val_shards, freq_index_obj,
                cfg.vocab_size, cfg.sequence_len,
                args.fixed_gram_samples_per_bucket, fixed_gram_seed)
            save_fixed_gram_manifest(manifest, manifest_path)
            print(f"[nglab] wrote fixed gram manifest: {manifest_path}")
        fixed_gram_probe = FixedGramProbe(
            manifest, cfg.data_dir, cfg.sequence_len, device, cfg.device_batch_size)

        # Keep the old contiguous-batch probe and legacy diagnostic on their
        # own dataset instances so neither can advance the writer iterator.
        fixed_train_ds = TokenizedShardDataset(
            cfg.data_dir, cfg.train_shards, cfg.sequence_len,
            cfg.device_batch_size, cfg.data_seed)
        fixed_val_ds = TokenizedShardDataset(
            cfg.data_dir, cfg.val_shards, cfg.sequence_len,
            cfg.device_batch_size, cfg.data_seed)
        fixed_train_probe = collect_fixed_probe(
            fixed_train_ds, device, args.fixed_probe_batches, fixed_probe_offset_batches)
        fixed_val_probe = collect_fixed_probe(
            fixed_val_ds, device, args.fixed_probe_batches)
        if args.legacy_freq_eval:
            freq_diag_train_ds = TokenizedShardDataset(
                cfg.data_dir, cfg.train_shards, cfg.sequence_len,
                cfg.device_batch_size, cfg.data_seed)
            freq_diag_val_ds = TokenizedShardDataset(
                cfg.data_dir, cfg.val_shards, cfg.sequence_len,
                cfg.device_batch_size, cfg.data_seed)
            freq_diag_train_iter = freq_diag_train_ds.iter_batches(device)
            freq_diag_val_iter = freq_diag_val_ds.iter_batches(device)
            freq_bin_log = open(os.path.join(cfg.out_dir, "freq_bin_loss.jsonl"), "w")
        print(f"[nglab] fixed gram/online frequency eval enabled (index: {args.freq_index})")
        if args.legacy_freq_eval:
            print("[nglab] legacy frequency eval enabled with independent diagnostic iterators")
    else:
        print("[nglab] online/fixed frequency contribution disabled (no --freq_index)")
    measurement_meta = {
        "run_id": args.run_id,
        "parameter_states": {
            "online": "pre_optimizer_step",
            "fixed_probe": "post_optimizer_step",
            "fixed_gram": "post_optimizer_step",
        },
        "online_train": "actual writer microbatches composing this optimizer step",
        "online_val": "fresh moving validation batches from an independent iterator",
        "fixed_probe": {
            "train_batches": args.fixed_probe_batches,
            "val_batches": args.fixed_probe_batches,
            "train_offset_batches": fixed_probe_offset_batches,
            "train_offset_optimizer_steps": args.fixed_probe_train_offset_steps,
            "selection": "deterministic batches from independent fixed-order iterators",
        },
        "fixed_gram_probe": ({
            "samples_per_bucket": args.fixed_gram_samples_per_bucket,
            "seed": cfg.seed if args.fixed_gram_seed is None else args.fixed_gram_seed,
            "selection": "fixed train/val token occurrences sampled independently per train-frequency bucket",
            "timing": "post_optimizer_step",
            "manifest": "fixed_gram_probe_manifest.json",
            "stats": fixed_gram_probe.manifest_stats() if fixed_gram_probe is not None else None,
        } if freq_index_obj is not None else None),
        "legacy_freq_eval": bool(args.legacy_freq_eval and freq_index_obj is not None),
        "sampling": {
            "base_interval": args.online_frequency_interval,
            "epoch_window": args.online_frequency_epoch_window,
            "epoch_dense_interval": args.online_frequency_dense_interval,
            "probe_window": args.online_frequency_probe_window,
            "probe_dense_interval": args.online_frequency_probe_dense_interval,
            "probe_center_steps": probe_center_steps,
            "estimated_steps_per_epoch": steps_per_epoch,
        },
        "geometry": {
            "device_batch_size": cfg.device_batch_size,
            "sequence_len": cfg.sequence_len,
            "grad_accum": grad_accum,
        },
    }
    with open(os.path.join(cfg.out_dir, "frequency_measurement_meta.json"), "w") as f:
        json.dump(measurement_meta, f, indent=2)
    last_val_loss = float("nan")
    last_train_loss = float("nan")

    model.train()
    t0 = time.time()
    for step in range(cfg.max_steps):
        step_1based = step + 1
        sample_reason = frequency_sample_reason(
            step_1based, steps_per_epoch, args.online_frequency_interval,
            args.online_frequency_epoch_window, args.online_frequency_dense_interval,
            cfg.max_steps, probe_center_steps,
            args.online_frequency_probe_window,
            args.online_frequency_probe_dense_interval)
        # gradient accumulation
        optimizer.zero_grad()
        accum_loss = 0.0
        online_train_frequency = None
        online_train_loss_sum = 0.0
        online_train_token_count = 0
        if sample_reason is not None and freq_index_obj is not None:
            from ngram_freq import FreqBinLossAccumulator
            online_accs = {
                "bigram": FreqBinLossAccumulator(freq_index_obj, cfg.vocab_size, "bigram"),
                "trigram": FreqBinLossAccumulator(freq_index_obj, cfg.vocab_size, "trigram"),
            }
        for _ in range(grad_accum):
            try:
                inp, tgt = next(train_iter)
            except StopIteration:
                train_iter = train_ds.iter_batches(device)
                inp, tgt = next(train_iter)
            token_loss = model(inp, targets=tgt, reduction="none")
            loss = token_loss.mean() / grad_accum
            loss.backward()
            accum_loss += loss.item()
            if sample_reason is not None and freq_index_obj is not None:
                online_train_loss_sum += float(token_loss.detach().sum().item())
                online_train_token_count += token_loss.numel()
                for acc in online_accs.values():
                    acc.update(inp, token_loss.detach())
        train_loss = accum_loss
        online_loss_entry = {
            "step": step_1based,
            "epoch": train_ds._epoch + 1,
            "train_writer_loss": train_loss,
            "elapsed_s": time.time() - t0,
        }
        online_loss_log.write(json.dumps(online_loss_entry) + "\n")
        online_loss_log.flush()

        if sample_reason is not None and freq_index_obj is not None:
            online_train_frequency = {
                branch: acc.summary() for branch, acc in online_accs.items()
            }
            online_train_loss = online_train_loss_sum / max(1, online_train_token_count)
            online_val_batches = [
                next(online_val_iter) for _ in range(args.online_frequency_val_batches)
            ]
            online_val_loss, online_val_frequency = frequency_summary_from_batches(
                model, online_val_batches, freq_index_obj, cfg.vocab_size)
            online_frequency_entry = {
                "step": step_1based,
                "epoch": train_ds._epoch + 1,
                "reason": sample_reason,
                "train_writer_loss": online_train_loss,
                "online_val_loss": online_val_loss,
                "train_writer": online_train_frequency,
                "online_val": online_val_frequency,
                "gap_contribution": contribution_gap(
                    online_train_frequency, online_val_frequency),
            }
            online_frequency_log.write(json.dumps(online_frequency_entry) + "\n")
            online_frequency_log.flush()

        progress = step_1based / cfg.max_steps
        lr_mult = get_lr_multiplier(progress, cfg.warmdown_ratio)
        optimizer.step(lr_mult=lr_mult)

        if sample_reason is not None and freq_index_obj is not None:
            fixed_train_loss, fixed_train_frequency = frequency_summary_from_batches(
                model, fixed_train_probe, freq_index_obj, cfg.vocab_size)
            fixed_val_loss, fixed_val_frequency = frequency_summary_from_batches(
                model, fixed_val_probe, freq_index_obj, cfg.vocab_size)
            fixed_probe_entry = {
                "step": step_1based,
                "epoch": train_ds._epoch + 1,
                "reason": sample_reason,
                "fixed_train_loss": fixed_train_loss,
                "fixed_val_loss": fixed_val_loss,
                "train_probe": fixed_train_frequency,
                "val_probe": fixed_val_frequency,
                "gap_contribution": contribution_gap(
                    fixed_train_frequency, fixed_val_frequency),
            }
            fixed_probe_frequency_log.write(json.dumps(fixed_probe_entry) + "\n")
            fixed_probe_frequency_log.flush()

            fixed_gram_evaluation = fixed_gram_probe.evaluate(model)
            fixed_gram_entry = {
                "step": step_1based,
                "epoch": train_ds._epoch + 1,
                "reason": sample_reason,
                "train_loss": fixed_gram_overall_loss(
                    fixed_gram_evaluation.get("train", {})),
                "val_loss": fixed_gram_overall_loss(
                    fixed_gram_evaluation.get("val", {})),
                "branches": fixed_gram_gap_summary(fixed_gram_evaluation),
                "unique_chunks": fixed_gram_probe.manifest_stats()["unique_chunks"],
            }
            fixed_gram_frequency_log.write(json.dumps(fixed_gram_entry) + "\n")
            fixed_gram_frequency_log.flush()

        # periodic val
        if step_1based % cfg.val_interval_steps == 0 or step == cfg.max_steps - 1:
            last_val_loss = evaluate_val(model, val_iter, cfg.val_batches)
            last_train_loss = train_loss
            entry = {
                "step": step_1based,
                "train_loss": train_loss,
                "val_loss": last_val_loss,
                "gap": last_val_loss - train_loss,
                "lr_mult": lr_mult,
                "epoch": train_ds._epoch + 1,
                "elapsed_s": time.time() - t0,
            }
            train_log.write(json.dumps(entry) + "\n")
            train_log.flush()
            print(f"[nglab] step {step_1based:4d} | train {train_loss:.4f} | val {last_val_loss:.4f} "
                  f"| gap {last_val_loss-train_loss:+.4f} | epoch {train_ds._epoch+1} | "
                  f"lr_m {lr_mult:.2f} | {(time.time()-t0):.0f}s")

        # Optional legacy diagnostic. It uses independent iterators and is
        # intentionally separate from the fixed gram statistics above.
        if freq_bin_log is not None and (step_1based % args.freq_eval_interval == 0 or step == cfg.max_steps - 1):
            train_freq = evaluate_freq_bins(model, freq_diag_train_iter, freq_index_obj,
                                            args.freq_eval_batches, cfg.vocab_size)
            val_freq = evaluate_freq_bins(model, freq_diag_val_iter, freq_index_obj,
                                          args.freq_eval_batches, cfg.vocab_size)
            fb_entry = {
                "step": step_1based,
                "epoch": train_ds._epoch + 1,
                "train": train_freq,
                "val": val_freq,
            }
            freq_bin_log.write(json.dumps(fb_entry) + "\n")
            freq_bin_log.flush()

        # periodic table norm
        if step_1based % cfg.table_norm_interval_steps == 0:
            tn = table_param_rms(model)
            tn_entry = {"step": step_1based, **tn}
            table_log.write(json.dumps(tn_entry) + "\n")
            table_log.flush()

    train_log.close()
    table_log.close()
    online_loss_log.close()
    if freq_bin_log is not None:
        freq_bin_log.close()
    if online_frequency_log is not None:
        online_frequency_log.close()
    if fixed_probe_frequency_log is not None:
        fixed_probe_frequency_log.close()
    if fixed_gram_frequency_log is not None:
        fixed_gram_frequency_log.close()

    # summary
    summary = {
        "run_id": args.run_id,
        "injection_position": cfg.nanogpt_ngram_injection_position,
        "steps": cfg.max_steps,
        "seed": cfg.seed,
        "final_train_loss": last_train_loss,
        "final_val_loss": last_val_loss,
        "final_gap": last_val_loss - last_train_loss,
        "n_params": n_params,
        "config": cfg.__dict__,
    }
    with open(os.path.join(cfg.out_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"[nglab] DONE. final gap = {last_val_loss - last_train_loss:+.4f}")
    print(f"[nglab] output: {cfg.out_dir}")


if __name__ == "__main__":
    main()
