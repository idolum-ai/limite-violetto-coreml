"""Configuration schema and validation for GMPT models."""

from typing import Any

from transformers.configuration_utils import PretrainedConfig

#: Numerical conventions supported by the backbone.
SUPPORTED = {
    "mlp_type": {"relu2", "swiglu"},
    "pos_mode": {"rope"},
    "qk_norm": {"rms_pre_rope"},
    "rms_norm_eps_mode": {"torch_finfo_default"},
    "rope_style": {"interleaved_pairs_odd_lane_sign_flip"},
    "rope_cos_sin_dtype": {"bfloat16"},
    "mudd_accumulation": {"ordered_left_to_right"},
    "ve_head_slice": {"first_num_key_value_heads"},
    "lm_head_precision_mode": {"oracle_exact"},
    "softcap_kind": {"sigmoid"},
}

#: The reference applies `k >= q - sliding_window`, i.e. the window is inclusive
#: of the query token, so a local layer sees `sliding_window + 1` keys. Matching
#: the raw number against an exclusive-convention kernel silently drops the
#: oldest key on every local layer.
SLIDING_WINDOW_CONVENTION = (
    "k >= q - sliding_window, inclusive of the query token (span = sliding_window + 1 keys)"
)


class GMPTConfig(PretrainedConfig):
    """Architecture and numerical settings for the GMPT backbone."""

    model_type = "gmpt"
    has_no_defaults_at_init = True
    keys_to_ignore_at_inference = ["past_key_values"]

    base_model_tp_plan = {
        "layers.*.self_attn.q_proj": "colwise",
        "layers.*.self_attn.k_proj": "colwise",
        "layers.*.self_attn.v_proj": "colwise",
        "layers.*.self_attn.o_proj": "rowwise",
        "layers.*.mlp.gate_proj": "colwise",
        "layers.*.mlp.up_proj": "colwise",
        "layers.*.mlp.down_proj": "rowwise",
    }
    base_model_pp_plan = {
        "embed_tokens": (["input_ids"], ["inputs_embeds"]),
        "layers": (["hidden_states"], ["hidden_states"]),
        "norm": (["hidden_states"], ["hidden_states"]),
    }

    def __init__(
        self,
        # -- shape ------------------------------------------------------------
        vocab_size: int = 151680,
        padded_vocab_size: int | None = None,
        tokenizer_vocab_size: int | None = None,
        hidden_size: int = 1280,
        intermediate_size: int = 5120,
        num_hidden_layers: int = 48,
        num_attention_heads: int = 10,
        num_key_value_heads: int = 2,
        head_dim: int = 128,
        mlp_ratio: int = 4,
        mlp_type: str = "relu2",
        max_position_embeddings: int = 8192,
        tie_word_embeddings: bool = True,
        # -- normalisation ----------------------------------------------------
        rms_norm_has_weight: bool = False,
        rms_norm_eps_mode: str = "torch_finfo_default",
        qk_norm: str = "rms_pre_rope",
        # -- attention --------------------------------------------------------
        attention_softmax_scale: float = 0.1,
        sliding_window: int = 1024,
        sliding_window_convention: str = SLIDING_WINDOW_CONVENTION,
        global_window: int = -1,
        global_layers: list[int] | None = None,
        global_every: int = 4,
        global_nope: bool = True,
        attn_gate_channels: int = 0,
        attn_gate_scale: float = 2.0,
        attn_gate_applied: str = "per_head_before_o_proj",
        attention_dropout: float = 0.0,
        # -- rotary -----------------------------------------------------------
        pos_mode: str = "rope",
        rope_frac: float = 0.5,
        rope_base_local: float = 1024.0,
        rope_base_global: float = 1024.0,
        rope_per_layer: bool = False,
        rope_n_pairs: int | None = None,
        rope_style: str = "interleaved_pairs_odd_lane_sign_flip",
        rope_cos_sin_dtype: str = "bfloat16",
        # -- value embeddings -------------------------------------------------
        ve_dim: int = 128,
        ve_stored_heads: int | None = None,
        ve_layers: list[int] | None = None,
        ve_gate_channels: int = 12,
        ve_gate_scale: float = 2.0,
        ve_head_slice: str = "first_num_key_value_heads",
        ve_applied_before_qk_norm: bool = True,
        # -- cross-head attention correction ----------------------------------
        xsa: bool = True,
        xsa_layers: list[int] | None = None,
        xsa_normalize_eps: float = 1e-4,
        # -- dynamic dense residual mixing ------------------------------------
        mudd: bool = True,
        mudd_at: list[int] | None = None,
        mudd_layers: list[int] | None = None,
        mudd_taps: int = 3,
        mudd_inter: int = 32,
        mudd_tap_idx: dict[str, list[int]] | None = None,
        mudd_hist_convention: str = "hist[0] = rms_norm(embedding); hist[j] = output of layer j-1",
        mudd_accumulation: str = "ordered_left_to_right",
        mudd_mlp: bool = False,
        mudd_r_site: str | None = None,
        # -- output head ------------------------------------------------------
        softcap_logits: dict[str, Any] | None = None,
        final_softcap: float = 0.0,
        lm_head_precision_mode: str = "oracle_exact",
        lm_head_compute_dtype: str | None = None,  # legacy field; refused below
        # -- tokens -----------------------------------------------------------
        pad_token_id: int | None = 151643,
        bos_token_id: int | None = 151643,
        eos_token_id: int | list[int] | None = 151645,
        # -- provenance (carried through, never read by the model) ------------
        source_format: str | None = None,
        source_metadata_assumptions: list[str] | None = None,
        checkpoint_step: int | None = None,
        checkpoint_world_size: int | None = None,
        **kwargs,
    ):
        self.vocab_size = vocab_size
        self.padded_vocab_size = padded_vocab_size
        self.tokenizer_vocab_size = tokenizer_vocab_size
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.head_dim = head_dim
        self.mlp_ratio = mlp_ratio
        self.mlp_type = mlp_type
        self.max_position_embeddings = max_position_embeddings

        self.rms_norm_has_weight = rms_norm_has_weight
        self.rms_norm_eps_mode = rms_norm_eps_mode
        self.qk_norm = qk_norm

        self.attention_softmax_scale = attention_softmax_scale
        self.sliding_window = sliding_window
        self.sliding_window_convention = sliding_window_convention
        self.global_window = global_window
        self.global_every = global_every
        self.global_layers = sorted(int(x) for x in (global_layers or []))
        self.global_nope = global_nope
        self.attn_gate_channels = attn_gate_channels
        self.attn_gate_scale = attn_gate_scale
        self.attn_gate_applied = attn_gate_applied
        self.attention_dropout = attention_dropout

        self.pos_mode = pos_mode
        self.rope_frac = rope_frac
        self.rope_base_local = rope_base_local
        self.rope_base_global = rope_base_global
        self.rope_per_layer = rope_per_layer
        self.rope_n_pairs = (
            rope_n_pairs if rope_n_pairs is not None else max(1, int(head_dim * rope_frac) // 2)
        )
        self.rope_style = rope_style
        self.rope_cos_sin_dtype = rope_cos_sin_dtype

        self.ve_dim = ve_dim
        self.ve_stored_heads = ve_stored_heads if ve_stored_heads is not None else num_attention_heads
        self.ve_layers = sorted(int(x) for x in (ve_layers or []))
        self.ve_gate_channels = ve_gate_channels
        self.ve_gate_scale = ve_gate_scale
        self.ve_head_slice = ve_head_slice
        self.ve_applied_before_qk_norm = ve_applied_before_qk_norm

        self.xsa = xsa
        self.xsa_layers = sorted(int(x) for x in (xsa_layers or []))
        self.xsa_normalize_eps = xsa_normalize_eps

        self.mudd = mudd
        self.mudd_at = sorted(int(x) for x in (mudd_at or []))
        self.mudd_layers = sorted(int(x) for x in (mudd_layers if mudd_layers is not None else self.mudd_at))
        self.mudd_taps = mudd_taps
        self.mudd_inter = mudd_inter
        self.mudd_tap_idx = {str(k): [int(i) for i in v] for k, v in (mudd_tap_idx or {}).items()}
        self.mudd_hist_convention = mudd_hist_convention
        self.mudd_accumulation = mudd_accumulation
        self.mudd_mlp = mudd_mlp
        self.mudd_r_site = mudd_r_site

        self.softcap_logits = dict(softcap_logits or {"kind": "sigmoid", "a": 23.0, "b": 5.0, "c": 7.5})
        self.final_softcap = final_softcap
        # Refuse the legacy dtype field because it does not identify which
        # roundable head stage it controls.
        if lm_head_compute_dtype is not None:
            raise ValueError(
                f"GMPT config carries the superseded lm_head_compute_dtype={lm_head_compute_dtype!r}. "
                "That field named only one of the head's three roundable stages and is not "
                "reinterpreted; re-export the checkpoint with a current converter."
            )
        self.lm_head_precision_mode = lm_head_precision_mode

        self.source_format = source_format
        self.source_metadata_assumptions = source_metadata_assumptions
        self.checkpoint_step = checkpoint_step
        self.checkpoint_world_size = checkpoint_world_size

        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )

        self.validate_architecture()

    # -- derived ---------------------------------------------------------------

    @property
    def ve_layer_to_ordinal(self) -> dict[int, int]:
        """Map absolute value-embedding layer indices to compact gate ordinals."""
        return {layer: ordinal for ordinal, layer in enumerate(self.ve_layers)}

    def tap_indices(self, layer_idx: int) -> list[int] | None:
        return self.mudd_tap_idx.get(str(layer_idx))

    def is_global_layer(self, layer_idx: int) -> bool:
        return layer_idx in set(self.global_layers)

    # -- validation ------------------------------------------------------------

    def validate_architecture(self) -> None:
        for field, allowed in (
            ("mlp_type", SUPPORTED["mlp_type"]),
            ("pos_mode", SUPPORTED["pos_mode"]),
            ("qk_norm", SUPPORTED["qk_norm"]),
            ("rms_norm_eps_mode", SUPPORTED["rms_norm_eps_mode"]),
            ("rope_style", SUPPORTED["rope_style"]),
            ("rope_cos_sin_dtype", SUPPORTED["rope_cos_sin_dtype"]),
            ("mudd_accumulation", SUPPORTED["mudd_accumulation"]),
            ("ve_head_slice", SUPPORTED["ve_head_slice"]),
            ("lm_head_precision_mode", SUPPORTED["lm_head_precision_mode"]),
        ):
            value = getattr(self, field)
            if value not in allowed:
                raise NotImplementedError(f"GMPT does not implement {field}={value!r}; supported: {sorted(allowed)}")

        if self.rms_norm_has_weight:
            raise NotImplementedError("GMPT RMS norm carries no learnable gain (rms_norm_has_weight must be False).")
        if self.sliding_window_convention != SLIDING_WINDOW_CONVENTION:
            raise NotImplementedError(
                f"Unexpected sliding_window_convention {self.sliding_window_convention!r}. "
                "The window span is an off-by-one trap; refusing to guess."
            )
        if self.softcap_logits.get("kind") not in SUPPORTED["softcap_kind"]:
            raise NotImplementedError(f"GMPT implements only sigmoid logit softcapping, got {self.softcap_logits!r}")
        if self.final_softcap:
            raise NotImplementedError("GMPT does not implement the pre-head tanh cap (final_softcap must be 0).")
        if self.attn_gate_channels < 0:
            raise ValueError(f"attn_gate_channels must be non-negative, got {self.attn_gate_channels}.")
        if self.attn_gate_channels:
            if self.attn_gate_channels > self.hidden_size:
                raise ValueError(
                    f"attn_gate_channels ({self.attn_gate_channels}) must not exceed hidden_size ({self.hidden_size})."
                )
            if self.attn_gate_scale != 2.0:
                raise NotImplementedError(
                    "This attention gate is verified only as 2 * sigmoid(...); "
                    f"got attn_gate_scale={self.attn_gate_scale!r}."
                )
            if self.attn_gate_applied != "per_head_before_o_proj":
                raise NotImplementedError(
                    "GMPT implements the attention gate only per head before o_proj; "
                    f"got attn_gate_applied={self.attn_gate_applied!r}."
                )
        if not self.global_nope:
            raise NotImplementedError("GMPT implements rotary-free global layers only (global_nope must be True).")
        if self.rope_per_layer and self.rope_base_local != self.rope_base_global:
            raise NotImplementedError(
                "rope_per_layer with distinct bases is unreachable while global layers skip rotary entirely."
            )

        if self.num_attention_heads % self.num_key_value_heads != 0:
            raise ValueError(
                f"num_attention_heads ({self.num_attention_heads}) must be divisible by "
                f"num_key_value_heads ({self.num_key_value_heads})."
            )
        if self.num_attention_heads * self.head_dim != self.hidden_size:
            raise ValueError(
                f"num_attention_heads * head_dim ({self.num_attention_heads * self.head_dim}) "
                f"must equal hidden_size ({self.hidden_size})."
            )
        if self.mlp_type == "relu2" and self.mlp_ratio * self.hidden_size != self.intermediate_size:
            raise ValueError(
                f"mlp_ratio * hidden_size ({self.mlp_ratio * self.hidden_size}) "
                f"must equal intermediate_size ({self.intermediate_size})."
            )
        if self.ve_dim > self.head_dim:
            raise ValueError(f"ve_dim ({self.ve_dim}) must not exceed head_dim ({self.head_dim}).")
        if self.ve_gate_channels > self.hidden_size:
            raise ValueError(f"ve_gate_channels ({self.ve_gate_channels}) must not exceed hidden_size.")
        if self.ve_stored_heads not in (self.num_attention_heads, self.num_key_value_heads):
            raise NotImplementedError(
                f"ve_stored_heads={self.ve_stored_heads}; only query-width or key-value-width tables are supported."
            )
        if self.mudd_mlp and self.mudd_r_site != "resid":
            raise NotImplementedError(
                "MUDD R way is implemented only at the attention residual base; "
                f"got mudd_r_site={self.mudd_r_site!r}."
            )
        if self.padded_vocab_size is None:
            raise ValueError("padded_vocab_size is mandatory for GMPT; matrix width must be explicit.")
        if self.padded_vocab_size != self.vocab_size:
            raise ValueError(
                f"vocab_size ({self.vocab_size}) is the width of the embedding and head matrices and must "
                f"equal padded_vocab_size ({self.padded_vocab_size})."
            )
        if self.tokenizer_vocab_size is None:
            raise ValueError("tokenizer_vocab_size is mandatory for GMPT; padded head rows are not tokens.")
        if not 0 < self.tokenizer_vocab_size <= self.vocab_size:
            raise ValueError(
                f"tokenizer_vocab_size ({self.tokenizer_vocab_size}) must be in (0, vocab_size={self.vocab_size}]."
            )
        for name, layers in (
            ("global_layers", self.global_layers),
            ("ve_layers", self.ve_layers),
            ("xsa_layers", self.xsa_layers),
            ("mudd_layers", self.mudd_layers),
        ):
            if layers and (layers[0] < 0 or layers[-1] >= self.num_hidden_layers):
                raise ValueError(f"{name}={layers} is out of range for num_hidden_layers={self.num_hidden_layers}.")

        if self.mudd:
            if sorted(int(k) for k in self.mudd_tap_idx) != list(self.mudd_layers):
                raise ValueError(
                    f"mudd_tap_idx keys {sorted(self.mudd_tap_idx)} must match mudd_layers {self.mudd_layers}. "
                    "The tap topology is inferred upstream and pinned here; a mismatch is silent."
                )
            for layer, taps in self.mudd_tap_idx.items():
                if len(taps) != self.mudd_taps:
                    raise ValueError(f"mudd_tap_idx[{layer}] has {len(taps)} taps, expected {self.mudd_taps}.")
                if max(taps) > int(layer):
                    raise ValueError(
                        f"mudd_tap_idx[{layer}]={taps} reads a history entry that does not exist yet at layer {layer}."
                    )
        elif self.mudd_layers:
            raise ValueError("mudd is disabled but mudd_layers is non-empty.")

        if not self.xsa and self.xsa_layers:
            raise ValueError("xsa is disabled but xsa_layers is non-empty.")


__all__ = ["GMPTConfig", "SLIDING_WINDOW_CONVENTION"]
