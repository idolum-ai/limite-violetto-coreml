"""Opt-in Core ML request warmup matching the qualified CPU cache policy.

The ordinary :class:`Pipeline` remains unchanged. This wrapper recomputes a
cached prefix while its length is below 16 tokens, using fresh Core ML states,
then transfers those states back to the caller's request only on success.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .runtime import Pipeline, Request


WARMUP_TOKENS = 16


@dataclass
class TinyPrefixRequest(Request):
    """A normal pipeline request plus its owned, bounded warmup token prefix."""

    warmup_ids: list[int] = field(default_factory=list)

    def close(self) -> None:
        super().close()
        self.warmup_ids.clear()


class TinyPrefixPipeline(Pipeline):
    """Opt-in ``Pipeline`` variant with CPU-equivalent short-cache warmup.

    For cached calls made while ``0 < request.length < 16``, this rebuilds the
    cache from the exact owned prefix plus the new IDs in fresh state objects.
    It forwards capture callbacks only for positions belonging to the new
    input, even when the recomputation crosses the ordinary 32-token chunk
    boundary.
    """

    runtime_adapter = "tiny_prefix_replay"

    def __init__(self, compute="CPU_ONLY", model_dir=None, identity_manifest=None):
        if compute != "CPU_ONLY":
            raise ValueError("Tiny-prefix warmup mirrors the CPU reference and requires CPU_ONLY.")
        super().__init__(compute, model_dir, identity_manifest)

    def request(self) -> TinyPrefixRequest:
        base = self._new_base_request()
        return TinyPrefixRequest(base.owner, base.states)

    def _new_base_request(self) -> Request:
        return super().request()

    def _validate(self, ids: list[int], request: TinyPrefixRequest) -> None:
        if not isinstance(request, TinyPrefixRequest):
            raise ValueError("Expected a request created by TinyPrefixPipeline.")
        super().validate(ids, request)

    def _base_forward(self, ids: list[int], request: Request, capture=None):
        return super().forward(ids, request, capture)

    def forward(self, ids: list[int], request: TinyPrefixRequest, capture=None):
        self._validate(ids, request)
        old_length = request.length
        if old_length < WARMUP_TOKENS and len(request.warmup_ids) != old_length:
            raise ValueError("Tiny-prefix history does not match the request length; reset the request.")
        if old_length >= WARMUP_TOKENS and request.warmup_ids:
            raise ValueError("Tiny-prefix history must be empty after warmup; reset the request.")
        if 0 < old_length < WARMUP_TOKENS:
            combined = [*request.warmup_ids, *ids]
            fresh = self._new_base_request()
            positions: dict[str, int] = {}

            def trim_capture(name: str, value: np.ndarray) -> None:
                start = positions.get(name, 0)
                count = value.shape[1]
                end = start + count
                positions[name] = end
                if capture is None:
                    return
                keep_start = max(old_length, start)
                keep_end = min(len(combined), end)
                if keep_start < keep_end:
                    capture(name, value[:, keep_start - start : keep_end - start].copy())

            try:
                scores = self._base_forward(combined, fresh, trim_capture if capture is not None else None)
                if fresh.length != len(combined) or not fresh.valid:
                    raise RuntimeError("Fresh warmup request did not complete the full prefix.")
                # Publish the replacement state only after the entire forward
                # (including its head projection) completed successfully.
                old_states = request.states
                request.states = fresh.states
                fresh.states = []
                request.length = fresh.length
                request.valid = True
                request.warmup_ids = combined if request.length < WARMUP_TOKENS else []
                fresh.close()
                del old_states
                return scores
            except BaseException:
                request.valid = False
                raise
            finally:
                fresh.close()

        scores = self._base_forward(ids, request, capture)
        request.warmup_ids = [*request.warmup_ids, *ids] if request.length < WARMUP_TOKENS else []
        return scores


def select_pipeline(tiny_prefix_replay: bool = False):
    """Return the standard or opt-in adapter class without changing defaults."""
    if type(tiny_prefix_replay) is not bool:
        raise ValueError("tiny_prefix_replay must be a boolean.")
    return TinyPrefixPipeline if tiny_prefix_replay else Pipeline
