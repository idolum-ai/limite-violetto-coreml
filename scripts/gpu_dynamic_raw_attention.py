"""Experimental attention finish: dynamic query rows, fixed K2048, raw einsum scores."""
import torch
import torch.nn.functional as F
import coreml_port.graphs as graphs


def finish_parts(layer, x, residual, q, keys, values, v, positions, slots):
    count = q.shape[1]
    rows = torch.minimum(torch.tensor(32, dtype=torch.int32), positions[-1] + 1)
    query = torch.cat((q, torch.zeros(1, 31, 10, 128)), dim=1)[:, :rows]
    pos = torch.cat((positions, torch.zeros(31, dtype=torch.int32)))[:rows]
    expanded_keys = keys.repeat_interleave(5, dim=2)
    expanded_values = values.repeat_interleave(5, dim=2).permute(0, 2, 1, 3)
    scores = torch.einsum("bthd,bshd->bhts", query, expanded_keys) * .1
    mask = slots[None, :] <= pos[:, None]
    if layer.window >= 0:
        mask = mask & (slots[None, :] >= pos[:, None] - layer.window)
    weights = graphs.softmax(scores.masked_fill(~mask[None, None], float("-inf")))
    weighted = graphs.Q(torch.matmul(weights, expanded_values).permute(0, 2, 1, 3)[:, :count])
    vn = graphs.xsa_normalize(v.repeat_interleave(5, dim=2))
    y = graphs.Q(weighted - graphs.Q(layer.alpha * graphs.xsa_dot(weighted, vn) * vn))
    gate = graphs.Q(2 * graphs.Q(torch.sigmoid(graphs.Q(graphs.linear(x[..., :128], layer.ag)))))
    y = graphs.Q(y * gate.unsqueeze(-1))
    out = graphs.Q(graphs.linear(y.reshape(1, -1, 1280), layer.o))
    mixed = graphs.Q(graphs.Q(layer.resid_attn * residual) + graphs.Q(layer.post_attn * out))
    h = graphs.norm(mixed)
    activated = graphs.Q(graphs.Q(F.silu(graphs.Q(graphs.linear(h, layer.gate)))) * graphs.Q(graphs.linear(h, layer.up)))
    mlp = graphs.Q(graphs.linear(activated, layer.down))
    return graphs.Q(graphs.Q(layer.resid_mlp * mixed) + graphs.Q(layer.post_mlp * mlp)), weighted


def finish(layer, x, residual, q, keys, values, v, positions, slots):
    return finish_parts(layer, x, residual, q, keys, values, v, positions, slots)[0]
