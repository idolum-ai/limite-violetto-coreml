"""Inference-only GMPT backbone for Hugging Face value-model loading."""
from __future__ import annotations
from types import SimpleNamespace
import torch
import torch.nn.functional as F
from torch import Tensor, nn
from configuration_gmpt import GMPTConfig

IGNORE_INDEX = -100
DTYPE_BY_NAME = {"float32": torch.float32, "bfloat16": torch.bfloat16}
flash_attn_varlen_func = None
flash_attn_3_varlen_func = None


def configure_attention(backend):
    global flash_attn_varlen_func, flash_attn_3_varlen_func
    if backend == "flash_attention_3":
        from flash_attn_interface import flash_attn_varlen_func as impl
        def deterministic(*args, **kwargs):
            return impl(*args, deterministic=True, **kwargs)
        flash_attn_3_varlen_func = deterministic
    elif backend == "flash_attention_2":
        from flash_attn import flash_attn_varlen_func as impl
        flash_attn_varlen_func = impl
    elif backend != "eager":
        raise ValueError("Supported backends: eager, flash_attention_2, flash_attention_3")


def get_cu_seqlens_from_position_ids(position_ids):
    # This release scores one unpadded sequence with contiguous positions.
    # Packed multi-document input is not supported: a reset position would
    # silently let a later document attend to an earlier one.
    n = position_ids.shape[1]
    expected = torch.arange(n, device=position_ids.device, dtype=position_ids.dtype)
    if position_ids.shape[0] != 1 or not torch.equal(position_ids[0], expected):
        raise ValueError(
            "position_ids must be a single contiguous range starting at 0; "
            "packed multi-document sequences are not supported."
        )
    return torch.tensor([0, n], device=position_ids.device, dtype=torch.int32), n


class GMPTPreTrainedModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

    def post_init(self):
        # Parameters are restored by strict checkpoint loading.
        pass


def _validate_gmpt_ids(
    ids: Tensor,
    valid_vocab_size: int,
    *,
    kind: str,
    allow_ignore_index: bool = False,
) -> None:
    """Validate IDs before embedding lookup."""
    invalid_mask = (ids < 0) | (ids >= valid_vocab_size)
    if allow_ignore_index:
        invalid_mask &= ids != IGNORE_INDEX
    if not torch.any(invalid_mask):
        return
    invalid_ids = ids[invalid_mask]
    observed = sorted(set(int(token) for token in invalid_ids.detach().cpu().tolist()))[:8]
    raise ValueError(
        f"GMPT {kind} must be in [0, {valid_vocab_size})"
        + (f" or equal ignore_index={IGNORE_INDEX}" if allow_ignore_index else "")
        + f"; padded vocabulary rows are not tokens; invalid IDs include {observed}."
    )


def validate_gmpt_input_ids(input_ids: Tensor, valid_vocab_size: int) -> None:
    _validate_gmpt_ids(input_ids, valid_vocab_size, kind="input_ids")


def rms_norm(x: Tensor) -> Tensor:
    """Gain-free RMS norm with torch's dtype-dependent default epsilon."""
    return F.rms_norm(x, (x.size(-1),))


class GMPTRMSNorm(nn.Module):
    """Parameterless RMS norm exposed as ``model.norm``."""

    def forward(self, hidden_states: Tensor) -> Tensor:
        return rms_norm(hidden_states)


def build_rope_factors(config: GMPTConfig, position_ids: Tensor) -> tuple[Tensor, Tensor]:
    """cos/sin for the partial interleaved rotary.

    The frequency ladder is ``(1/base) ** linspace(0, 1, n_pairs)`` — *not* the
    usual ``arange(0, dim, 2) / dim`` — repeated per pair and zero-padded over
    the unrotated tail, and the odd lanes of ``sin`` are negated. cos/sin are
    materialised in bfloat16 before use, which is part of the numerics.
    """
    device = position_ids.device
    head_dim, n_pairs = config.head_dim, config.rope_n_pairs
    base = float(config.rope_base_local)
    freq = (1.0 / base) ** torch.linspace(0, 1, steps=n_pairs, dtype=torch.float32, device=device)
    freq = freq.repeat_interleave(2)
    freq = torch.cat([freq, freq.new_zeros(head_dim - 2 * n_pairs)])

    theta = position_ids.to(torch.float32).unsqueeze(-1) * freq
    cos = theta.cos().to(torch.bfloat16)
    sin = theta.sin().to(torch.bfloat16)
    sin[..., 1::2] *= -1
    # [batch, tokens, 1, head_dim] so it broadcasts over heads.
    return cos.unsqueeze(-2), sin.unsqueeze(-2)


def apply_rotary(x: Tensor, cos: Tensor, sin: Tensor) -> Tensor:
    """Adjacent-pair interleaving via ``flip(-1)``."""
    x_flip = x.view(*x.shape[:-1], x.shape[-1] // 2, 2).flip(-1).view(x.shape)
    return cos * x + sin * x_flip


def _seq_ids_from_cu_seqlens(cu_seqlens: Tensor, total_tokens: int) -> Tensor:
    lengths = cu_seqlens[1:] - cu_seqlens[:-1]
    return torch.repeat_interleave(
        torch.arange(lengths.numel(), device=cu_seqlens.device), lengths.to(torch.long)
    )[:total_tokens]


class GMPTAttention(nn.Module):
    def __init__(self, config: GMPTConfig, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.head_dim = config.head_dim
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads
        self.num_kv_groups = self.num_heads // self.num_kv_heads
        self.ve_stored_heads = config.ve_stored_heads
        self.scaling = config.attention_softmax_scale
        self.is_global = config.is_global_layer(layer_idx)
        # `k >= q - sliding_window`, inclusive of the query token. flash-attn's
        # `window_size=(left, 0)` admits `k in [q - left, q]`, so `left` is the
        # window itself, NOT `window - 1`.
        self.window = -1 if self.is_global else config.sliding_window
        self.applies_rope = not (self.is_global and config.global_nope)
        self.has_ve = layer_idx in set(config.ve_layers)
        self.has_xsa = config.xsa and layer_idx in set(config.xsa_layers)
        self.attn_gate_channels = config.attn_gate_channels
        self.attn_gate_scale = config.attn_gate_scale

        self.q_proj = nn.Linear(config.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(config.hidden_size, self.num_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(config.hidden_size, self.num_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, config.hidden_size, bias=False)

        if self.has_xsa:
            self.xsa_alpha = nn.Parameter(torch.zeros(self.num_heads))
        if self.has_ve:
            self.ve_gate = nn.Parameter(torch.zeros(self.ve_stored_heads, config.ve_gate_channels))
        if self.attn_gate_channels:
            self.attn_gate = nn.Parameter(torch.zeros(self.num_heads, self.attn_gate_channels))

    def _scaled(self, weight: Tensor, scale: Tensor, dtype: torch.dtype) -> Tensor:
        """Multiply weights by the learned scale in fp32, then round once."""
        return (scale.to(torch.float32).view(()) * weight).to(dtype)

    def project(
        self,
        hidden_states: Tensor,
        qkv_scale: Tensor,
        value_embeds: Tensor | None,
        cos: Tensor | None,
        sin: Tensor | None,
    ) -> tuple[Tensor, Tensor, Tensor]:
        dtype = hidden_states.dtype
        batch, tokens, _ = hidden_states.shape
        q = F.linear(hidden_states, self._scaled(self.q_proj.weight, qkv_scale, dtype))
        k = F.linear(hidden_states, self._scaled(self.k_proj.weight, qkv_scale, dtype))
        v = F.linear(hidden_states, self._scaled(self.v_proj.weight, qkv_scale, dtype))

        q = q.view(batch, tokens, self.num_heads, self.head_dim)
        k = k.view(batch, tokens, self.num_kv_heads, self.head_dim)
        v = v.view(batch, tokens, self.num_kv_heads, self.head_dim)

        if self.has_ve and value_embeds is not None:
            gate_weight = self.ve_gate[: self.num_kv_heads].to(dtype)
            gate = self.config.ve_gate_scale * torch.sigmoid(
                F.linear(hidden_states[..., : gate_weight.size(-1)], gate_weight)
            )
            v = v + gate.unsqueeze(-1) * value_embeds.to(dtype)

        # q/k norm comes after the value-embedding step and before rotary.
        q, k = rms_norm(q), rms_norm(k)
        if self.applies_rope and cos is not None:
            q, k = apply_rotary(q, cos, sin), apply_rotary(k, cos, sin)
        return q, k, v

    def _flash(self, q: Tensor, k: Tensor, v: Tensor, cu_seqlens: Tensor, max_seqlen: int) -> Tensor:
        impl = self.config._attn_implementation
        func = flash_attn_3_varlen_func if impl == "flash_attention_3" else flash_attn_varlen_func
        if func is None:
            raise ImportError(f"GMPT was configured with _attn_implementation={impl!r} but it is not installed.")
        out = func(
            q[0],
            k[0],
            v[0],
            cu_seqlens,
            cu_seqlens,
            max_seqlen,
            max_seqlen,
            causal=True,
            softmax_scale=self.scaling,
            window_size=(self.window, 0) if self.window >= 0 else (-1, -1),
        )
        if isinstance(out, tuple):
            out = out[0]
        return out.view(1, q.shape[1], self.num_heads, self.head_dim)

    def _eager(self, q: Tensor, k: Tensor, v: Tensor, seq_ids: Tensor) -> Tensor:
        """fp32 attention with explicit causal and local-window masks."""
        k = k.repeat_interleave(self.num_kv_groups, dim=2)
        v = v.repeat_interleave(self.num_kv_groups, dim=2)
        positions = torch.arange(q.shape[1], device=q.device)
        query, key = positions[:, None], positions[None, :]
        mask = (key <= query) & (seq_ids[:, None] == seq_ids[None, :])
        if self.window >= 0:
            mask &= key >= (query - self.window)
        logits = torch.einsum("bthd,bshd->bhts", q.float(), k.float()) * self.scaling
        logits = logits.masked_fill(~mask[None, None], float("-inf"))
        probs = logits.softmax(-1)
        return torch.einsum("bhts,bshd->bthd", probs, v.float()).to(v.dtype)

    def forward(
        self,
        hidden_states: Tensor,
        qkv_scale: Tensor,
        o_scale: Tensor,
        value_embeds: Tensor | None,
        cos: Tensor | None,
        sin: Tensor | None,
        cu_seqlens: Tensor | None,
        max_seqlen: int | None,
        seq_ids: Tensor | None,
    ) -> Tensor:
        q, k, v = self.project(hidden_states, qkv_scale, value_embeds, cos, sin)
        if cu_seqlens is not None:
            attn = self._flash(q, k, v, cu_seqlens, max_seqlen)
        else:
            attn = self._eager(q, k, v, seq_ids)

        if self.has_xsa:
            # Remove the component along the VE-adjusted value direction.
            value_dir = F.normalize(
                v.repeat_interleave(self.num_kv_groups, dim=2).float(),
                dim=-1,
                eps=self.config.xsa_normalize_eps,
            )
            projection = (attn.float() * value_dir).sum(-1, keepdim=True)
            alpha = torch.tanh(self.xsa_alpha.float()).view(1, 1, self.num_heads, 1)
            attn = attn - (alpha * projection * value_dir).to(attn.dtype)

        if self.attn_gate_channels:
            gate = self.attn_gate_scale * torch.sigmoid(
                F.linear(
                    hidden_states[..., : self.attn_gate_channels],
                    self.attn_gate.to(hidden_states.dtype),
                )
            )
            attn = attn * gate.to(attn.dtype).unsqueeze(-1)

        attn = attn.reshape(hidden_states.shape[0], hidden_states.shape[1], -1)
        return F.linear(attn, self._scaled(self.o_proj.weight, o_scale, attn.dtype))


class GMPTLayerScales(nn.Module):
    """The six learned per-layer scalars stored as one-element parameters."""

    def __init__(self):
        super().__init__()
        for name in ("qkv", "o", "resid_attn", "post_attn", "resid_mlp", "post_mlp"):
            self.register_parameter(name, nn.Parameter(torch.ones(1)))


class GMPTMLP(nn.Module):
    """Squared-ReLU or SwiGLU feed-forward branch."""

    def __init__(self, config: GMPTConfig):
        super().__init__()
        self.up_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)
        self.mlp_type = config.mlp_type
        if self.mlp_type == "swiglu":
            self.gate_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)

    def forward(self, hidden_states: Tensor) -> Tensor:
        up = self.up_proj(hidden_states)
        if self.mlp_type == "relu2":
            activated = F.relu(up).square()
        elif self.mlp_type == "swiglu":
            activated = F.silu(self.gate_proj(hidden_states)) * up
        else:  # Configuration validation fails before construction.
            raise AssertionError(f"unsupported mlp_type {self.mlp_type!r}")
        return self.down_proj(activated)


class GMPTDecoderLayer(nn.Module):
    def __init__(self, config: GMPTConfig, layer_idx: int):
        super().__init__()
        self.layer_idx = layer_idx
        self.self_attn = GMPTAttention(config, layer_idx)
        self.mlp = GMPTMLP(config)
        self.scales = GMPTLayerScales()

    def forward(
        self,
        hidden_states: Tensor,
        attn_input: Tensor,
        attn_residual: Tensor,
        value_embeds: Tensor | None,
        cos: Tensor | None,
        sin: Tensor | None,
        cu_seqlens: Tensor | None,
        max_seqlen: int | None,
        seq_ids: Tensor,
    ) -> Tensor:
        attn_out = self.self_attn(
            attn_input,
            self.scales.qkv,
            self.scales.o,
            value_embeds,
            cos,
            sin,
            cu_seqlens,
            max_seqlen,
            seq_ids,
        )
        # No plain additive residual: both sides carry a learned scalar.
        mixed = self.scales.resid_attn.to(attn_residual.dtype) * attn_residual + self.scales.post_attn.to(
            attn_out.dtype
        ) * attn_out
        mlp_out = self.mlp(rms_norm(mixed))
        return self.scales.resid_mlp.to(mixed.dtype) * mixed + self.scales.post_mlp.to(mlp_out.dtype) * mlp_out


class GMPTMudd(nn.Module):
    """Dynamic dense residual mixing.

    ``dense2`` and ``bias`` are indexed by **absolute** layer, so they are kept
    at full depth even though only a couple of layers carry a mixer. The whole
    mixer runs in the hidden dtype and accumulates the taps left-to-right; a
    stacked contraction is mathematically equal and numerically different.
    """

    def __init__(self, config: GMPTConfig):
        super().__init__()
        self.dense1 = nn.Parameter(torch.zeros(config.mudd_inter, config.hidden_size))
        self.dense2 = nn.Parameter(torch.zeros(config.num_hidden_layers, config.mudd_taps, config.mudd_inter))
        self.bias = nn.Parameter(torch.zeros(config.num_hidden_layers, config.mudd_taps))
        self.uses_r_way = config.mudd_mlp
        if self.uses_r_way:
            self.dense2_mlp = nn.Parameter(
                torch.zeros(config.num_hidden_layers, config.mudd_taps, config.mudd_inter)
            )
            self.bias_mlp = nn.Parameter(torch.zeros(config.num_hidden_layers, config.mudd_taps))

    def forward(self, taps: list[Tensor], current: Tensor, layer_idx: int, *, r_way: bool = False) -> Tensor:
        count = len(taps)
        inner = F.gelu(F.linear(rms_norm(current), self.dense1.to(current.dtype)))
        if r_way:
            if not self.uses_r_way:
                raise ValueError("MUDD R way requested by a config that carries no R-way tensors")
            dense2, bias = self.dense2_mlp, self.bias_mlp
        else:
            dense2, bias = self.dense2, self.bias
        weights = torch.einsum("btk,mk->btm", inner, dense2[layer_idx, :count].to(inner.dtype))
        weights = weights + bias[layer_idx, :count].to(weights.dtype)
        out = weights[..., 0:1].type_as(taps[0]) * taps[0]
        for idx in range(1, count):
            out = out + weights[..., idx : idx + 1].type_as(taps[idx]) * taps[idx]
        return out


class GMPTModel(GMPTPreTrainedModel):
    def __init__(self, config: GMPTConfig):
        super().__init__(config)
        self.vocab_size = config.vocab_size
        # No padding row is reserved in the learned token table.
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.value_embeds = nn.Embedding(config.vocab_size, config.ve_stored_heads * config.ve_dim)
        self.layers = nn.ModuleList([GMPTDecoderLayer(config, idx) for idx in range(config.num_hidden_layers)])
        self.norm = GMPTRMSNorm()
        self.mudd = GMPTMudd(config) if config.mudd else None
        # Only history entries some mixer actually reads are retained; keeping
        # the full 48-deep history would cost as much memory as the activations.
        self.retained_taps = sorted(
            {tap for layer, taps in config.mudd_tap_idx.items() for tap in taps if tap != int(layer)}
        )
        self.gradient_checkpointing = False
        self.post_init()

    def get_input_embeddings(self) -> nn.Module:
        return self.embed_tokens

    def set_input_embeddings(self, value: nn.Module) -> None:
        self.embed_tokens = value

    def _value_embeddings(self, input_ids: Tensor) -> Tensor | None:
        """Narrow the value-embedding stream to the key-value head count.

        A checkpoint stores either query-head width or key-value-head width.
        The explicit ``ve_stored_heads`` field selects the layout; query-width
        tables are narrowed at runtime and key-value-width tables are used as-is.

        ``.contiguous()`` preserves a stable key-value-head layout after slicing.
        """
        if not self.config.ve_layers:
            return None
        config = self.config
        embeds = self.value_embeds(input_ids)
        embeds = embeds.view(*input_ids.shape, config.ve_stored_heads, config.ve_dim)
        if config.ve_dim < config.head_dim:
            embeds = F.pad(embeds, (0, config.head_dim - config.ve_dim))
        if config.ve_stored_heads > config.num_key_value_heads:
            embeds = embeds[..., : config.num_key_value_heads, :]
        return embeds.contiguous()

    def forward(
        self,
        input_ids: Tensor | None = None,
        attention_mask: Tensor | None = None,
        position_ids: Tensor | None = None,
        inputs_embeds: Tensor | None = None,
        **kwargs,
    ) -> SimpleNamespace:
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")
        if input_ids is None:
            raise ValueError("GMPT needs input_ids: the value-embedding stream is a second token lookup.")
        valid_vocab_size = self.config.tokenizer_vocab_size
        validate_gmpt_input_ids(input_ids, valid_vocab_size)
        if position_ids is None:
            position_ids = torch.arange(input_ids.shape[1], device=input_ids.device).unsqueeze(0)

        config = self.config
        if config._attn_implementation in ("flash_attention_2", "flash_attention_3"):
            cu_seqlens, max_seqlen = get_cu_seqlens_from_position_ids(position_ids)
            torch._dynamo.mark_dynamic(cu_seqlens, 0)
            seq_ids = None
        else:
            cu_seqlens, max_seqlen = None, None
            packed, _ = get_cu_seqlens_from_position_ids(position_ids)
            seq_ids = _seq_ids_from_cu_seqlens(packed, position_ids.shape[-1])

        cos, sin = build_rope_factors(config, position_ids)
        value_embeds = self._value_embeddings(input_ids)

        hidden_states = rms_norm(self.embed_tokens(input_ids))
        # `history[j]` is the input to layer `j`: history[0] is the normed
        # embedding, history[j] is the output of layer j-1.
        history: dict[int, Tensor] = {0: hidden_states} if 0 in self.retained_taps else {}

        for layer_idx, decoder_layer in enumerate(self.layers):
            taps = config.tap_indices(layer_idx)
            if taps is not None:
                tap_values = [history[tap] if tap != layer_idx else hidden_states for tap in taps]
                attn_input = rms_norm(self.mudd(tap_values, hidden_states, layer_idx))
                attn_residual = (
                    self.mudd(tap_values, hidden_states, layer_idx, r_way=True)
                    if self.mudd.uses_r_way
                    else hidden_states
                )
            else:
                attn_input = rms_norm(hidden_states)
                attn_residual = hidden_states

            hidden_states = decoder_layer(
                hidden_states,
                attn_input,
                attn_residual,
                value_embeds,
                cos,
                sin,
                cu_seqlens,
                max_seqlen,
                seq_ids,
            )
            if layer_idx + 1 in self.retained_taps:
                history[layer_idx + 1] = hidden_states

        return SimpleNamespace(last_hidden_state=self.norm(hidden_states))
