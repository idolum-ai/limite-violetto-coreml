"""Bounded synthetic Violetto components, derived from pinned upstream code.

Sources: sources/value-model/modeling_backbone.py (Apache-2.0 model release)
and sources/runtime/.../src/limite_vllm/_components.py (Apache-2.0).
This is a feasibility graph, not the released model: FP32 arithmetic, explicit
FP32 opmath epsilon, BF16 tables/folded constants, and FP16 state. No real weights are used.
"""
from __future__ import annotations

import copy
import json
import math
from pathlib import Path
import sys

import torch
from torch import nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "sources/value-model"))
import modeling_backbone as upstream
from configuration_gmpt import GMPTConfig

# Verified on local CPU and pinned Torch 2.11 composite/CUDA source. The
# upstream component comment describing input-dtype epsilon is incorrect.
EPS = torch.finfo(torch.float32).eps


def norm(x):
    return x * torch.rsqrt((x * x).mean(-1, keepdim=True) + EPS)


def configuration(production=False):
    raw = json.loads((ROOT / "sources/value-model/backbone_config.json").read_text())
    if not production:
        raw.update(hidden_size=32, num_attention_heads=4, num_key_value_heads=2,
                   head_dim=8, intermediate_size=64, rope_n_pairs=2, ve_dim=8,
                   ve_stored_heads=2, attn_gate_channels=8, ve_gate_channels=4,
                   mudd_inter=8, sliding_window=4)
    return GMPTConfig(**raw)


def randomize(module, seed):
    generator = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for name, p in module.named_parameters():
            scale = 0.2 if p.ndim < 2 else 0.5 / math.sqrt(p.shape[-1])
            p.copy_(torch.randn(p.shape, generator=generator) * scale)


class Attention(nn.Module):
    def __init__(self, production=False, global_attention=False):
        super().__init__()
        self.config = configuration(production)
        c = self.config
        self.h, self.kv, self.d = c.num_attention_heads, c.num_key_value_heads, c.head_dim
        self.hidden = c.hidden_size
        self.capacity = 2048 if production else 16
        self.window = -1 if global_attention else c.sliding_window
        self.scale = c.attention_softmax_scale
        # Layers 1 and 7 both apply VE, XSA, gates; 7 is global.
        self.layer_idx = 7 if global_attention else 1
        source = upstream.GMPTAttention(c, self.layer_idx)
        randomize(source, 20260922)
        # Violetto multiplies BF16 projection weights by BF16-rounded scales.
        for name, scale in [("q_proj", 1.031), ("k_proj", 1.031), ("v_proj", 1.031), ("o_proj", 0.973)]:
            w = getattr(source, name).weight.detach().bfloat16()
            folded = (w * torch.tensor(scale).bfloat16()).float()
            self.register_buffer(name, folded)
        self.register_buffer("ve_gate", source.ve_gate.detach().bfloat16().float())
        self.register_buffer("attn_gate", source.attn_gate.detach().bfloat16().float())
        self.register_buffer("alpha", source.xsa_alpha.detach().float())
        positions = torch.arange(self.capacity).reshape(1, -1)
        cos, sin = upstream.build_rope_factors(c, positions)
        self.register_buffer("cos_table", cos[0, :, 0, :].float())
        self.register_buffer("sin_table", sin[0, :, 0, :].float())
        self.register_buffer("slots", torch.arange(self.capacity, dtype=torch.int32))
        self.register_buffer("k_cache", torch.zeros(1, self.capacity, self.kv, self.d, dtype=torch.float16))
        self.register_buffer("v_cache", torch.zeros_like(self.k_cache))
        self.eval()

    def reset(self):
        self.k_cache.zero_()
        self.v_cache.zero_()

    def project(self, x, ve, positions):
        q = F.linear(x, self.q_proj).reshape(1, -1, self.h, self.d)
        k = F.linear(x, self.k_proj).reshape(1, -1, self.kv, self.d)
        v = F.linear(x, self.v_proj).reshape(1, -1, self.kv, self.d)
        gate = self.config.ve_gate_scale * torch.sigmoid(F.linear(x[..., :self.config.ve_gate_channels], self.ve_gate))
        v = v + gate.unsqueeze(-1) * ve
        q, k = norm(q), norm(k)
        if self.window >= 0:
            cos = self.cos_table[positions.long()].reshape(1, -1, 1, self.d)
            sin = self.sin_table[positions.long()].reshape(1, -1, 1, self.d)
            q = cos * q + sin * q.reshape(1, -1, self.h, self.d // 2, 2).flip(-1).reshape(1, -1, self.h, self.d)
            k = cos * k + sin * k.reshape(1, -1, self.kv, self.d // 2, 2).flip(-1).reshape(1, -1, self.kv, self.d)
        return q, k, v

    def attend(self, q, k, v, query_positions, key_positions):
        k = k.repeat_interleave(self.h // self.kv, dim=2).permute(0, 2, 3, 1)
        v = v.repeat_interleave(self.h // self.kv, dim=2).permute(0, 2, 1, 3)
        scores = torch.matmul(q.permute(0, 2, 1, 3), k) * self.scale
        visible = key_positions[None, :] <= query_positions[:, None]
        if self.window >= 0:
            visible = visible & (key_positions[None, :] >= query_positions[:, None] - self.window)
        scores = scores.masked_fill(~visible.reshape(1, 1, -1, key_positions.shape[0]), float("-inf"))
        return torch.matmul(torch.softmax(scores, dim=-1), v).permute(0, 2, 1, 3)

    def finish(self, y, current_v, x):
        v = current_v.repeat_interleave(self.h // self.kv, dim=2)
        vn = v / torch.clamp(torch.sqrt((v * v).sum(-1, keepdim=True)), min=self.config.xsa_normalize_eps)
        proj = (y * vn).sum(-1, keepdim=True)
        y = y - torch.tanh(self.alpha).reshape(1, 1, self.h, 1) * proj * vn
        gate = self.config.attn_gate_scale * torch.sigmoid(F.linear(x[..., :self.config.attn_gate_channels], self.attn_gate))
        y = y * gate.unsqueeze(-1)
        return F.linear(y.reshape(1, -1, self.hidden), self.o_proj)

    def forward(self, x, ve, positions):
        q, k, v = self.project(x, ve, positions)
        # Each call supplies contiguous positions. Dynamic slice assignment is
        # the converter's documented cache-update path; scatter with expanded
        # symbolic indices failed its shape inference. Assignment narrows to
        # buffer dtype, and Core ML's state pass inserts the corresponding cast.
        start, end = positions[0], positions[-1] + 1
        self.k_cache[:, start:end] = k
        self.v_cache[:, start:end] = v
        y = self.attend(q, self.k_cache.float(), self.v_cache.float(), positions, self.slots)
        return self.finish(y, v, x)

    def full_prefix(self, x, ve, round_cache=False):
        pos = torch.arange(x.shape[1], dtype=torch.int32)
        q, k, v = self.project(x, ve, pos)
        kr, vr = (k.half().float(), v.half().float()) if round_cache else (k, v)
        return self.finish(self.attend(q, kr, vr, pos, pos), v, x)

    def upstream_reference(self, x, ve, bf16=False):
        source = upstream.GMPTAttention(self.config, self.layer_idx)
        with torch.no_grad():
            for name in ("q_proj", "k_proj", "v_proj", "o_proj"):
                getattr(source, name).weight.copy_(getattr(self, name))
            source.ve_gate.copy_(self.ve_gate)
            source.attn_gate.copy_(self.attn_gate)
            source.xsa_alpha.copy_(self.alpha)
        if bf16:
            source.to(torch.bfloat16)
            source.xsa_alpha = nn.Parameter(self.alpha.clone())
            x, ve = x.bfloat16(), ve.bfloat16()
        positions = torch.arange(x.shape[1]).reshape(1, -1)
        cos, sin = upstream.build_rope_factors(self.config, positions)
        if not bf16:
            cos, sin = cos.float(), sin.float()
        return source(x, torch.tensor(1.0), torch.tensor(1.0), ve, cos, sin,
                      None, None, torch.zeros(x.shape[1], dtype=torch.long)).float()


class Mixer(nn.Module):
    def __init__(self, production=False):
        super().__init__()
        self.config = configuration(production)
        source = upstream.GMPTMudd(self.config)
        randomize(source, 20260923)
        # Snapshot casts performed at point of use by the BF16 reference.
        for name, p in source.named_parameters():
            self.register_buffer(name, p.detach().bfloat16().float())
        self.eval()

    def combine(self, taps, current, layer, r_way):
        inner = F.gelu(F.linear(norm(current), self.dense1))
        dense = self.dense2_mlp if r_way else self.dense2
        bias = self.bias_mlp if r_way else self.bias
        weights = F.linear(inner, dense[layer]) + bias[layer]
        out = weights[..., 0:1] * taps[0]
        for i in (1, 2):
            out = out + weights[..., i:i+1] * taps[i]
        return out

    def forward(self, taps, current):
        # Test each real mixer site and both branches without fusing their order.
        return tuple(self.combine(taps, current, layer, r) for layer in (24, 47) for r in (False, True))

    def upstream_reference(self, taps, current, bf16=False):
        source = upstream.GMPTMudd(self.config)
        with torch.no_grad():
            for name, p in source.named_parameters():
                p.copy_(getattr(self, name))
        if bf16:
            taps, current = taps.bfloat16(), current.bfloat16()
        return tuple(source(list(taps.unbind(0)), current, layer, r_way=r).float()
                     for layer in (24, 47) for r in (False, True))
