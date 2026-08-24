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

Usage:
  python train.py --run_id myrun --injection_position input --steps 1000
  (or via env vars matching the old launcher conventions; see cluster/run_injpos.sh)
"""

from __future__ import annotations

import argparse
import hashlib
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
    ngram_table_betas: tuple = (0.0, 0.99)
    table_optimizer: str = "rmsprop"  # rmsprop | adamw | sgd
    table_lr_scale: float = 2.0       # multiplier on the n-gram table LR
    table_betas: tuple = (0.0, 0.99)  # (beta1, beta2) for adamw; beta1 = momentum for sgd
    table_mult: int = 64              # n-gram table size = vocab_size * table_mult
    intervention: str = "none"        # none | reset_table | mask_readout | freeze_table | freeze_backbone
    intervention_epoch: int = 1       # fire when 0-indexed epoch reaches this value (1 = start of epoch 2)
    adam_betas: tuple = (0.8, 0.95)
    weight_decay: float = 0.1
    # training
    seed: int = 42
    max_steps: int = 1000
    device_batch_size: int = 72
    total_batch_size: int = 147456
    val_interval_steps: int = 10
    val_batches: int = 4
    table_norm_interval_steps: int = 10
    warmdown_ratio: float = 0.65  # last 65% of steps decays LR
    lr_schedule_epochs: int = 0    # >0: anchor LR schedule to epoch count (ignores max_steps)
    # data (paths)
    data_dir: str = ""          # directory with train.bin / val.bin
    train_shards: list = field(default_factory=list)  # list of shard indices for train
    val_shards: list = field(default_factory=list)    # list of shard indices for val
    data_mode: str = "fixed"    # fixed = deterministic epoch replay
    data_seed: int = 42
    epoch_batches: int = 0      # >0: fix one epoch to exactly this many device batches
                                # (nested-prefix length control); 0 = use full shard length
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
        self.bigram_table_size = config.vocab_size * config.table_mult
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
        self.trigram_table_size = config.vocab_size * config.table_mult
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
        self._ngram_init_cpu_rng_state = torch.get_rng_state()
        self._ngram_init_cuda_rng_state = (
            torch.cuda.get_rng_state(torch.cuda.current_device())
            if torch.cuda.is_available() else None
        )
        self._initialize_ngram_tables()

    @torch.no_grad()
    def _initialize_ngram_tables(self):
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

    @torch.no_grad()
    def reset_ngram_tables(self):
        """Reset all n-gram table rows to their init distribution (P1: erase
        accumulated historical row state, keeping backbone intact)."""
        if not hasattr(self, "_ngram_init_cpu_rng_state"):
            raise RuntimeError("init_weights() must run before reset_ngram_tables()")
        device = next(self.parameters()).device
        devices = [torch.cuda.current_device()] if device.type == "cuda" else []
        with torch.random.fork_rng(devices=devices):
            torch.set_rng_state(self._ngram_init_cpu_rng_state)
            if device.type == "cuda":
                torch.cuda.set_rng_state(self._ngram_init_cuda_rng_state, device)
            self._initialize_ngram_tables()

    def apply_intervention(self, epoch0: int):
        """Fire intervention when 0-indexed epoch reaches `intervention_epoch`."""
        if self.config.intervention == "none":
            return
        if epoch0 != self.config.intervention_epoch:
            return
        if self.config.intervention == "reset_table":
            self.reset_ngram_tables()
        elif self.config.intervention == "mask_readout":
            self.config.enable_bigram_ve = False
            self.config.enable_trigram_ve = False
        elif self.config.intervention == "freeze_table":
            for p in self._ngram_params():
                p.requires_grad_(False)
        elif self.config.intervention == "freeze_backbone":
            for p in self._backbone_params():
                p.requires_grad_(False)

    def _ngram_params(self):
        markers = ("value_embeds", "bigram_ves", "trigram_ves",
                   "ve_gate", "bigram_gate", "trigram_gate")
        for name, p in self.named_parameters():
            if any(m in name for m in markers):
                yield p

    def _backbone_params(self):
        markers = ("value_embeds", "bigram_ves", "trigram_ves",
                   "ve_gate", "bigram_gate", "trigram_gate")
        for name, p in self.named_parameters():
            if not any(m in name for m in markers):
                yield p

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

    def forward(self, idx, targets=None, return_token_losses=False,
                return_injection_stats=False):
        B, T = idx.size()
        pos = torch.arange(0, T, dtype=torch.long, device=idx.device).unsqueeze(0)
        x = self.transformer.wte(idx)
        x = x + self.transformer.wpe(pos)
        injection_stats = {
            "bigram_rms": {"sequence_rms": [], "batch_rms": 0.0},
            "trigram_rms": {"sequence_rms": [], "batch_rms": 0.0},
            "total_rms": {"sequence_rms": [], "batch_rms": 0.0},
            "fourgram_rms": {"sequence_rms": [], "batch_rms": 0.0},
        }
        branch_residuals = {
            "bigram": [],
            "trigram": [],
        }
        prev_idx = torch.cat([idx[:, :1], idx[:, :-1]], dim=1)
        prev2_idx = torch.cat([idx[:, :2], idx[:, :-2]], dim=1)
        if self.config.nanogpt_ngram_injection_position == "input":
            ngram_residual = self._compute_input_ngram_residual(idx)
            x = x + ngram_residual
            if return_injection_stats:
                branch_residuals["bigram"] = []
                branch_residuals["trigram"] = []
                for li in sorted(self.bigram_ve_layers):
                    lvs = self.bigram_ves[str(li)]
                    bi = [((prev_idx * p1) ^ (idx * p2)) % self.bigram_table_size
                          for p1, p2 in self.bigram_hash_primes_per_layer[li]]
                    branch_residuals["bigram"].append(
                        torch.cat([lvs[k](bi[k]) for k in range(self.bigram_K)], dim=-1)
                    )
                for li in sorted(self.trigram_ve_layers):
                    lvs = self.trigram_ves[str(li)]
                    lp = self.trigram_hash_primes_per_layer[li]
                    ti = (
                        ((prev2_idx * lp[0]) ^ (prev_idx * lp[1]) ^ (idx * lp[2])) % self.trigram_table_size,
                        ((prev2_idx * lp[3]) ^ (prev_idx * lp[4]) ^ (idx * lp[5])) % self.trigram_table_size,
                    )
                    branch_residuals["trigram"].append(
                        torch.cat([lvs[0](ti[0]), lvs[1](ti[1])], dim=-1)
                    )
                for branch, residuals in branch_residuals.items():
                    if residuals:
                        branch_residual = torch.stack(residuals).sum(dim=0)
                        branch_rms = torch.sqrt(torch.mean(branch_residual.float() ** 2, dim=-1))
                        injection_stats[f"{branch}_rms"] = {
                            "sequence_rms": branch_rms.mean(dim=0).detach().cpu().tolist(),
                            "batch_rms": float(branch_rms.mean()),
                        }
                total_rms = torch.sqrt(torch.mean(ngram_residual.float() ** 2, dim=-1))
                injection_stats["total_rms"] = {
                    "sequence_rms": total_rms.mean(dim=0).detach().cpu().tolist(),
                    "batch_rms": float(total_rms.mean()),
                }
        x = self.transformer.drop(x)
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
            token_losses = F.cross_entropy(
                logits.float().view(-1, logits.size(-1)),
                targets.view(-1), ignore_index=-1, reduction="none"
            ).view_as(targets)
            loss = token_losses[targets.ne(-1)].mean()
            if return_token_losses or return_injection_stats:
                return token_losses if return_token_losses else loss, injection_stats
            return loss
        if return_token_losses:
            raise ValueError("return_token_losses requires targets")
        if return_injection_stats:
            return logits.float(), injection_stats
        return logits.float()


# ---------------------------------------------------------------------------
# 2. Optimizer: AdamW (backbone) + RMSProp (n-gram table), mixed grouping
# ---------------------------------------------------------------------------


class MixedOptimizer:
    """Minimal AdamW + RMSProp optimizer (no Muon).

    n-gram table params (value_embeds / bigram_ves / trigram_ves / *_gate)
    -> table_optimizer: RMSProp betas=(0.0, 0.99) bias-corrected (default),
       AdamW with table_betas, or SGD with momentum=table_betas[0].
    Everything else -> AdamW with betas=(0.8, 0.95), weight_decay=0.1 (matrices only).
    """

    def __init__(self, model: nn.Module, lr: float, ngram_betas, adam_betas,
                 weight_decay: float, table_optimizer: str = "rmsprop",
                 table_lr_scale: float = 1.0, table_betas=None):
        self.model = model
        self.lr = lr
        self.ngram_beta2 = ngram_betas[1]
        self.table_optimizer = table_optimizer
        self.table_lr_scale = table_lr_scale
        self.table_betas = tuple(table_betas) if table_betas is not None else tuple(ngram_betas)
        self.adam_betas = adam_betas
        self.weight_decay = weight_decay
        self.adam_steps = {}
        self.adam_exp_avg = {}
        self.adam_exp_avg_sq = {}
        self.rms_steps = {}
        self.rms_exp_avg_sq = {}
        self.table_exp_avg = {}  # adamw first moment / sgd momentum buffer
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

    def _table_rmsprop_step(self, name, p, lr_t):
        g = p.grad
        if g is None:
            return
        if name not in self.rms_steps:
            self.rms_steps[name] = 0
            self.rms_exp_avg_sq[name] = torch.zeros_like(p)
        self.rms_steps[name] += 1
        step = self.rms_steps[name]
        # beta2 for RMSProp EMA comes from table_betas.
        b2 = self.table_betas[1]
        exp_avg_sq = self.rms_exp_avg_sq[name]
        # no weight decay on n-gram tables
        exp_avg_sq.lerp_(g.square(), 1 - b2)
        bias2 = 1 - b2 ** step
        denom = (exp_avg_sq / bias2).sqrt() + 1e-10
        p.add_(g / denom, alpha=-lr_t)

    def _table_adamw_step(self, name, p, lr_t):
        g = p.grad
        if g is None:
            return
        if name not in self.rms_steps:
            self.rms_steps[name] = 0
            self.rms_exp_avg_sq[name] = torch.zeros_like(p)
            self.table_exp_avg[name] = torch.zeros_like(p)
        self.rms_steps[name] += 1
        step = self.rms_steps[name]
        b1, b2 = self.table_betas
        exp_avg = self.table_exp_avg[name]
        exp_avg_sq = self.rms_exp_avg_sq[name]
        exp_avg.lerp_(g, 1 - b1)
        exp_avg_sq.lerp_(g.square(), 1 - b2)
        bias1 = 1 - b1 ** step
        bias2 = 1 - b2 ** step
        denom = (exp_avg_sq / bias2).sqrt() + 1e-8
        p.add_(exp_avg / denom, alpha=-lr_t / bias1)

    def _table_sgd_step(self, name, p, lr_t):
        g = p.grad
        if g is None:
            return
        momentum = self.table_betas[0]
        buf = self.table_exp_avg.get(name)
        if buf is None:
            buf = torch.zeros_like(p)
            self.table_exp_avg[name] = buf
        buf.mul_(momentum).add_(g)
        p.add_(buf, alpha=-lr_t)

    def step(self, lr_mult: float = 1.0):
        lr_t = self.lr * lr_mult
        table_lr_t = lr_t * self.table_lr_scale
        with torch.no_grad():
            for name, p in self.adam_params:
                if p.requires_grad:
                    self._adamw_step(name, p, lr_t)
            for name, p in self.ngram_params:
                if not p.requires_grad:
                    continue
                if self.table_optimizer == "adamw":
                    self._table_adamw_step(name, p, table_lr_t)
                elif self.table_optimizer == "sgd":
                    self._table_sgd_step(name, p, table_lr_t)
                else:
                    self._table_rmsprop_step(name, p, table_lr_t)

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
                 device_batch_size: int, seed: int = 42, epoch_batches: int = 0):
        self.data_dir = data_dir
        self.shard_ids = list(shard_ids)
        self.sequence_len = sequence_len
        self.chunk_size = sequence_len + 1
        self.device_batch_size = device_batch_size
        self.seed = seed
        self.epoch_batches = epoch_batches
        self._buffers = {}
        self._epoch = 0
        self._batch_in_epoch = 0
        # total tokens per shard (lazy)
        self._shard_lens = {}
        # yielded batches per full epoch (drop_last per shard), for epoch-anchored LR
        if epoch_batches > 0:
            # Nested-prefix epoch length: the first `epoch_batches` device
            # batches from the shard prefix form one epoch; all epoch lengths
            # are strict prefixes of the same data stream.
            self._batches_per_epoch = epoch_batches
        else:
            self._batches_per_epoch = sum(
                self.shard_chunks(sid) // self.device_batch_size for sid in self.shard_ids)

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
        """Yield (input, target) tensors for each chunk in fixed order.

        With epoch_batches > 0, only the first `epoch_batches * device_batch_size`
        chunks of the train prefix are yielded each epoch (nested-prefix control).
        """
        budget = self.epoch_batches * self.device_batch_size if self.epoch_batches > 0 else None
        emitted = 0
        for sid in self.shard_ids:
            buf = self._load_shard(sid)
            n = self.shard_chunks(sid)
            for i in range(n):
                if budget is not None and emitted >= budget:
                    return
                start = i * self.chunk_size
                chunk = np.array(buf[start:start + self.chunk_size], dtype=np.int64)
                inp = torch.from_numpy(chunk[:-1])
                tgt = torch.from_numpy(chunk[1:])
                emitted += 1
                yield inp, tgt

    def epoch_progress(self) -> float:
        """Fraction of the current epoch completed (0..1 within an epoch)."""
        return self._batch_in_epoch / max(1, self._batches_per_epoch)

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
                    self._batch_in_epoch += 1
            self._epoch += 1
            self._batch_in_epoch = 0
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


def evaluate_val(model: NanoGPT, fixed_val_batches) -> float:
    """Val loss on a FIXED batch set captured once at startup and reused for
    every evaluation, so the val-loss curve always measures the same val data."""
    model.eval()
    losses = []
    with torch.no_grad():
        for inp, tgt in fixed_val_batches:
            loss = model(inp, targets=tgt)
            losses.append(loss.item())
    model.train()
    return float(np.mean(losses)) if losses else float("nan")


def evaluate_fixed_probe(model: NanoGPT, fixed_train_probe) -> float:
    """Loss on the FIXED train probe (same train batches every eval).

    Returns the mean loss over the probe.  Used with evaluate_val to form
    fixed-train gap = fixed_val − fixed_train_probe, which is the scaling
    main quantity (never the online batch-averaged train loss)."""
    model.eval()
    losses = []
    with torch.no_grad():
        for inp, tgt in fixed_train_probe:
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
        loader_iter = iter(loader)
        for _ in range(n_batches):
            try:
                inp, tgt = next(loader_iter)
            except StopIteration:
                break
            ptl = compute_per_token_loss(model, inp, tgt)
            for branch in accs:
                accs[branch].update(inp, ptl)
    model.train()
    return {branch: acc.summary() for branch, acc in accs.items()}


def evaluate_exact_freq(model: NanoGPT, loader, freq_index, n_batches: int,
                        vocab_size: int, branch: str) -> dict:
    """Run exact-frequency loss accumulation on n_batches from loader.

    Returns {exact_f: {token_count, distinct_contexts, mean_loss, loss_sum,
    loss_sq_sum}} for the given branch ("bigram"|"trigram").  Uses an
    independent iterator over `loader` (a list of batches or an iterable),
    so it never consumes the training stream.
    """
    from ngram_freq import ExactFreqLossAccumulator, compute_per_token_loss
    model.eval()
    acc = ExactFreqLossAccumulator(freq_index, vocab_size, branch)
    with torch.no_grad():
        loader_iter = iter(loader)
        for _ in range(n_batches):
            try:
                inp, tgt = next(loader_iter)
            except StopIteration:
                break
            ptl = compute_per_token_loss(model, inp, tgt)
            acc.update(inp, ptl)
    model.train()
    return acc.summary()


def compute_shared_contexts(model: NanoGPT, train_probe, val_probe,
                            freq_index, vocab_size, branch: str) -> dict:
    """Context-matched gap statistics.

    For contexts that appear in BOTH the fixed train probe and the fixed val
    probe, compute per-exact-f aggregated (train_loss, val_loss) differences.
    Returns {f: {shared_contexts, train_mean, val_mean, gap}} plus a count of
    shared contexts overall.
    """
    from ngram_freq import ExactFreqLossAccumulator, compute_per_token_loss
    model.eval()
    train_keys: dict = {}
    val_keys: dict = {}
    with torch.no_grad():
        for inp, tgt in train_probe:
            ptl = compute_per_token_loss(model, inp, tgt)
            acc = ExactFreqLossAccumulator(freq_index, vocab_size, branch)
            keys = acc._compute_keys(inp).cpu().numpy().ravel()
            losses = ptl.cpu().numpy().ravel()
            for k, l in zip(keys, losses):
                train_keys[int(k)] = float(l)
        for inp, tgt in val_probe:
            ptl = compute_per_token_loss(model, inp, tgt)
            acc = ExactFreqLossAccumulator(freq_index, vocab_size, branch)
            keys = acc._compute_keys(inp).cpu().numpy().ravel()
            losses = ptl.cpu().numpy().ravel()
            for k, l in zip(keys, losses):
                val_keys[int(k)] = float(l)
    # contexts present in both
    shared = set(train_keys) & set(val_keys)
    per_f = {}
    for k in shared:
        f = freq_index.bigram.get(k, 0) if branch == "bigram" else freq_index.trigram.get(k, 0)
        st = per_f.setdefault(int(f), {"shared_contexts": 0, "train_sum": 0.0, "val_sum": 0.0})
        st["shared_contexts"] += 1
        st["train_sum"] += train_keys[k]
        st["val_sum"] += val_keys[k]
    out = {}
    for f, st in per_f.items():
        n = st["shared_contexts"]
        out[f] = {
            "f": f,
            "shared_contexts": n,
            "train_mean": st["train_sum"] / n,
            "val_mean": st["val_sum"] / n,
            "gap": (st["val_sum"] - st["train_sum"]) / n,
        }
    model.train()
    return {"shared_total": len(shared), "per_f": out, "branch": branch}


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
    parser.add_argument("--val_interval", type=int, default=10)
    parser.add_argument("--val_batches", type=int, default=4)
    parser.add_argument("--table_norm_interval", type=int, default=10)
    parser.add_argument("--lr", type=float, default=0.004)
    parser.add_argument("--table_optimizer", default="rmsprop",
                        choices=["rmsprop", "adamw", "sgd"],
                        help="optimizer for n-gram table params (default rmsprop)")
    parser.add_argument("--table_lr_scale", type=float, default=2.0,
                        help="multiplier on the n-gram table LR (default 2.0; the SSOT standard)")
    parser.add_argument("--table_betas", default=None,
                        help="beta1,beta2 for table (adamw: both; sgd: beta1=momentum); default 0.0,0.99")
    parser.add_argument("--table_mult", type=int, default=64,
                        help="n-gram table size = vocab_size * table_mult (default 64)")
    parser.add_argument("--intervention", default="none",
                        choices=["none", "reset_table", "mask_readout",
                                 "freeze_table", "freeze_backbone"],
                        help="causal intervention fired at intervention_epoch boundary")
    parser.add_argument("--intervention_epoch", type=int, default=1,
                        help="0-indexed epoch at which the intervention fires (1 = start of epoch 2)")
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
    parser.add_argument("--freq_eval_interval", type=int, default=10)
    parser.add_argument("--freq_eval_batches", type=int, default=4)
    parser.add_argument("--lr_schedule_epochs", type=int, default=0,
                        help=">0: anchor LR schedule to this many epochs (epoch-based progress)")
    parser.add_argument("--epoch_batches", type=int, default=0,
                        help=">0: fix one epoch to exactly this many device batches "
                             "(nested-prefix epoch length); 0 = full shard length")
    parser.add_argument("--fixed_train_probe", type=int, default=4,
                        help="number of fixed train batches to hold for the "
                             "fixed-train probe (0 disables)")
    parser.add_argument("--probe_eval_interval", type=int, default=10,
                        help="steps between fixed-train/val probe evals (also fired "
                             "at every epoch boundary)")
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
        lr_schedule_epochs=args.lr_schedule_epochs,
        nanogpt_adam_lr=args.lr,
        table_optimizer=args.table_optimizer,
        table_lr_scale=args.table_lr_scale,
        table_mult=args.table_mult,
        intervention=args.intervention,
        intervention_epoch=args.intervention_epoch,
        data_dir=args.data_dir,
        train_shards=[int(x) for x in args.train_shards.split(",") if x.strip()],
        val_shards=[int(x) for x in args.val_shards.split(",") if x.strip()],
        epoch_batches=args.epoch_batches,
        out_dir=os.path.join(args.out_dir, args.run_id),
    )

    if args.table_betas:
        parts = [float(x) for x in args.table_betas.replace(";", ",").split(",") if x.strip()]
        assert len(parts) == 2 and all(0.0 <= b < 1.0 for b in parts), \
            "table_betas must be two values in [0, 1)"
        cfg.table_betas = tuple(parts)
    else:
        cfg.table_betas = (0.0, 0.99)  # β₂=0.99 is the scaling default (plan §1)

    set_seed(cfg.seed)
    os.makedirs(cfg.out_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[nglab] device={device} injection={cfg.nanogpt_ngram_injection_position} steps={cfg.max_steps}")

    # data
    train_ds = TokenizedShardDataset(cfg.data_dir, cfg.train_shards, cfg.sequence_len,
                                     cfg.device_batch_size, cfg.data_seed,
                                     epoch_batches=cfg.epoch_batches)
    val_ds = TokenizedShardDataset(cfg.data_dir, cfg.val_shards, cfg.sequence_len,
                                   cfg.device_batch_size, cfg.data_seed)
    train_iter = train_ds.iter_batches(device)
    val_iter = val_ds.iter_batches(device)
    # Fixed validation batches: captured once and reused for every val eval, so
    # the val-loss curve always measures the same val data.  The val-side
    # freq-bin eval uses its own moving window on val_iter; train is the main
    # queue consumed by training + train-side freq eval.
    fixed_val_batches = [
        next(val_iter) for _ in range(max(cfg.val_batches, args.freq_eval_batches))
    ]
    validation_batches = fixed_val_batches[:cfg.val_batches]
    # Fixed freq-bin val batches: use the same prefix as the scalar validation
    # set whenever the requested batch counts differ.
    fixed_freq_val_batches = fixed_val_batches[:args.freq_eval_batches]
    print(f"[nglab] fixed val batches: {len(fixed_val_batches)} x shape {tuple(fixed_val_batches[0][0].shape)} "
          f"+ {len(fixed_freq_val_batches)} freq-val batches")
    # Fixed train probe: captured from a SEPARATE dataset instance that owns its
    # own iterator, so grabbing the probe never consumes the training stream and
    # never advances the training epoch counter.  The probe is a fixed set of
    # train batches reused for every probe eval; its gap is fixed_val − fixed_train.
    fixed_train_probe = []
    probe_hash = None
    if args.fixed_train_probe > 0:
        probe_ds = TokenizedShardDataset(cfg.data_dir, cfg.train_shards, cfg.sequence_len,
                                         cfg.device_batch_size, cfg.data_seed,
                                         epoch_batches=cfg.epoch_batches)
        probe_iter = probe_ds.iter_batches(device)
        for _ in range(args.fixed_train_probe):
            fixed_train_probe.append(next(probe_iter))
        probe_hash = hashlib.sha256(
            b"".join(batch[0].cpu().numpy().tobytes() for batch in fixed_train_probe)).hexdigest()[:16]
        print(f"[nglab] fixed train probe: {len(fixed_train_probe)} batches, sha256={probe_hash}")
    grad_accum = max(1, cfg.total_batch_size // (cfg.device_batch_size * cfg.sequence_len))
    print(f"[nglab] grad_accum={grad_accum} train_chunks={train_ds.total_chunks()} "
          f"val_chunks={val_ds.total_chunks()}")

    # model
    model = NanoGPT(cfg).to(device)
    model.init_weights()
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[nglab] model params: {n_params/1e6:.1f}M")
    optimizer = MixedOptimizer(model, lr=cfg.nanogpt_adam_lr,
                               ngram_betas=cfg.ngram_table_betas,
                               adam_betas=cfg.adam_betas,
                               weight_decay=cfg.weight_decay,
                               table_optimizer=cfg.table_optimizer,
                               table_lr_scale=cfg.table_lr_scale,
                               table_betas=cfg.table_betas)

    # logs
    train_log = open(os.path.join(cfg.out_dir, "train_log.jsonl"), "w")
    table_log = open(os.path.join(cfg.out_dir, "table_norm.jsonl"), "w")
    probe_log = None
    if args.fixed_train_probe > 0:
        probe_log = open(os.path.join(cfg.out_dir, "fixed_train_loss.jsonl"), "w")
    fixed_train_freq_log = None
    freq_bin_log = None
    exact_freq_log = None
    freq_index_obj = None
    if args.freq_index and os.path.exists(args.freq_index):
        from ngram_freq import GlobalFrequencyIndex
        freq_index_obj = GlobalFrequencyIndex.load(args.freq_index)
        freq_bin_log = open(os.path.join(cfg.out_dir, "freq_bin_loss.jsonl"), "w")
        if fixed_train_probe:
            fixed_train_freq_log = open(
                os.path.join(cfg.out_dir, "fixed_train_freq_bin_loss.jsonl"), "w"
            )
            exact_freq_log = open(
                os.path.join(cfg.out_dir, "exact_freq_loss.jsonl"), "w"
            )
        print(f"[nglab] freq-bin eval enabled (index: {args.freq_index})")
        # Independent train-side diagnostic iterator: reads the same shards in
        # the same fixed order but owns its own dataset state, so freq-bin
        # diagnostics never consume training batches and never advance the
        # training epoch counter.
        freq_train_ds = TokenizedShardDataset(cfg.data_dir, cfg.train_shards,
                                              cfg.sequence_len,
                                              cfg.device_batch_size,
                                              cfg.data_seed)
        freq_train_iter = freq_train_ds.iter_batches(device)
    last_val_loss = float("nan")
    last_train_loss = float("nan")
    last_fixed_train_loss = float("nan")
    last_fixed_val_loss = float("nan")

    model.train()
    t0 = time.time()
    intervention_fired = False
    for step in range(cfg.max_steps):
        # gradient accumulation
        optimizer.zero_grad()
        accum_loss = 0.0
        for _ in range(grad_accum):
            try:
                inp, tgt = next(train_iter)
            except StopIteration:
                train_iter = train_ds.iter_batches(device)
                inp, tgt = next(train_iter)
            if not intervention_fired and train_ds._epoch >= cfg.intervention_epoch:
                model.apply_intervention(train_ds._epoch)
                intervention_fired = True
            loss = model(inp, targets=tgt) / grad_accum
            loss.backward()
            accum_loss += loss.item()
        train_loss = accum_loss
        if cfg.lr_schedule_epochs > 0:
            # epoch-anchored LR: all runs share the same per-epoch trajectory
            progress = min(1.0, (train_ds._epoch + train_ds.epoch_progress()) / cfg.lr_schedule_epochs)
        else:
            progress = (step + 1) / cfg.max_steps
        lr_mult = get_lr_multiplier(progress, cfg.warmdown_ratio)
        optimizer.step(lr_mult=lr_mult)

        # periodic val
        if (step + 1) % cfg.val_interval_steps == 0 or step == cfg.max_steps - 1:
            last_val_loss = evaluate_val(model, validation_batches)
            last_train_loss = train_loss
            entry = {
                "step": step + 1,
                "train_loss": train_loss,
                "val_loss": last_val_loss,
                "gap": last_val_loss - train_loss,
                "lr_mult": lr_mult,
                "epoch": train_ds._epoch + 1,
                "elapsed_s": time.time() - t0,
            }
            train_log.write(json.dumps(entry) + "\n")
            train_log.flush()
            print(f"[nglab] step {step+1:4d} | train {train_loss:.4f} | val {last_val_loss:.4f} "
                  f"| gap {last_val_loss-train_loss:+.4f} | epoch {train_ds._epoch+1} | "
                  f"lr_m {lr_mult:.2f} | {(time.time()-t0):.0f}s")

        # periodic fixed-train probe eval (interval + epoch boundary)
        if probe_log is not None and fixed_train_probe:
            epoch_boundary = (train_ds._batch_in_epoch == 0 and step > 0)
            if (step + 1) % args.probe_eval_interval == 0 or epoch_boundary or step == cfg.max_steps - 1:
                fixed_train_loss = evaluate_fixed_probe(model, fixed_train_probe)
                fixed_val_loss = evaluate_val(model, validation_batches)
                last_fixed_train_loss = fixed_train_loss
                last_fixed_val_loss = fixed_val_loss
                probe_entry = {
                    "step": step + 1,
                    "epoch": train_ds._epoch + 1,
                    "fixed_train_loss": fixed_train_loss,
                    "fixed_val_loss": fixed_val_loss,
                    "fixed_gap": fixed_val_loss - fixed_train_loss,
                    "lr_mult": lr_mult,
                }
                probe_log.write(json.dumps(probe_entry) + "\n")
                probe_log.flush()
                print(f"[nglab] probe step {step+1:4d} | ftrain {fixed_train_loss:.4f} "
                      f"| fval {fixed_val_loss:.4f} | fgap {fixed_val_loss-fixed_train_loss:+.4f} "
                      f"| epoch {train_ds._epoch+1}")
                if fixed_train_freq_log is not None:
                    fixed_train_freq = evaluate_freq_bins(
                        model,
                        fixed_train_probe,
                        freq_index_obj,
                        len(fixed_train_probe),
                        cfg.vocab_size,
                    )
                    fixed_train_freq_log.write(json.dumps({
                        "step": step + 1,
                        "epoch": train_ds._epoch + 1,
                        "fixed_train": fixed_train_freq,
                        "probe_batch_count": len(fixed_train_probe),
                        "probe_batch_sha256": probe_hash,
                    }) + "\n")
                    fixed_train_freq_log.flush()
                if exact_freq_log is not None:
                    # exact-frequency marginal on the fixed train probe and the
                    # fixed val batches, plus context-matched gap statistics.
                    ef_train = {b: evaluate_exact_freq(
                        model, fixed_train_probe, freq_index_obj,
                        len(fixed_train_probe), cfg.vocab_size, b)
                        for b in ("bigram", "trigram")}
                    ef_val = {b: evaluate_exact_freq(
                        model, validation_batches, freq_index_obj,
                        len(validation_batches), cfg.vocab_size, b)
                        for b in ("bigram", "trigram")}
                    shared = {b: compute_shared_contexts(
                        model, fixed_train_probe, validation_batches,
                        freq_index_obj, cfg.vocab_size, b)
                        for b in ("bigram", "trigram")}
                    exact_freq_log.write(json.dumps({
                        "step": step + 1,
                        "epoch": train_ds._epoch + 1,
                        "train": ef_train,
                        "val": ef_val,
                        "shared": shared,
                        "probe_batch_sha256": probe_hash,
                    }) + "\n")
                    exact_freq_log.flush()

        # periodic freq-bin eval (train + val)
        if freq_bin_log is not None and ((step + 1) % args.freq_eval_interval == 0 or step == cfg.max_steps - 1):
            # train freq-bin (independent diagnostic iterator: fresh train
            # batches, never touches the training stream)
            train_freq = evaluate_freq_bins(model, freq_train_iter, freq_index_obj,
                                            args.freq_eval_batches, cfg.vocab_size)
            # val freq-bin (same fixed val data every eval)
            val_freq = evaluate_freq_bins(model, fixed_freq_val_batches, freq_index_obj,
                                          args.freq_eval_batches, cfg.vocab_size)
            fb_entry = {
                "step": step + 1,
                "epoch": train_ds._epoch + 1,
                "train": train_freq,
                "val": val_freq,
            }
            freq_bin_log.write(json.dumps(fb_entry) + "\n")
            freq_bin_log.flush()

        # periodic table norm
        if (step + 1) % cfg.table_norm_interval_steps == 0:
            tn = table_param_rms(model)
            tn_entry = {"step": step + 1, **tn}
            table_log.write(json.dumps(tn_entry) + "\n")
            table_log.flush()

    train_log.close()
    table_log.close()
    if probe_log is not None:
        probe_log.close()
    if fixed_train_freq_log is not None:
        fixed_train_freq_log.close()
    if freq_bin_log is not None:
        freq_bin_log.close()
    if exact_freq_log is not None:
        exact_freq_log.close()

    # summary
    summary = {
        "run_id": args.run_id,
        "injection_position": cfg.nanogpt_ngram_injection_position,
        "steps": cfg.max_steps,
        "seed": cfg.seed,
        "epoch_batches": cfg.epoch_batches,
        "fixed_train_probe_sha256": probe_hash if args.fixed_train_probe > 0 else None,
        "fixed_train_probe_batches": len(fixed_train_probe),
        "probe_eval_interval": args.probe_eval_interval,
        "fixed_train_loss_log": (
            "fixed_train_loss.jsonl" if probe_log is not None else None
        ),
        "fixed_train_freq_bin_log": (
            "fixed_train_freq_bin_loss.jsonl"
            if fixed_train_freq_log is not None else None
        ),
        "exact_freq_log": (
            "exact_freq_loss.jsonl" if exact_freq_log is not None else None
        ),
        "freq_index": args.freq_index if args.freq_index else None,
        "final_train_loss": last_train_loss,
        "final_val_loss": last_val_loss,
        "final_gap": last_val_loss - last_train_loss,
        "final_fixed_train_loss": last_fixed_train_loss,
        "final_fixed_val_loss": last_fixed_val_loss,
        "final_fixed_gap": last_fixed_val_loss - last_fixed_train_loss,
        "n_params": n_params,
        "config": cfg.__dict__,
    }
    with open(os.path.join(cfg.out_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"[nglab] DONE. final gap = {last_val_loss - last_train_loss:+.4f}")
    print(f"[nglab] output: {cfg.out_dir}")


if __name__ == "__main__":
    main()
