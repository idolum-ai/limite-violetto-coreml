"""Experimental explicit BF16 ties-to-even rounding for Core ML GPU probes.

This is a local diagnostic candidate. Do not promote it without shard and
whole-model fidelity gates.
"""
import torch


def Q_even(x):
    exponent = torch.floor(torch.log2(torch.clamp(x.abs(), min=1e-30))) - 7
    powers = 2. ** torch.arange(-128, 128, dtype=torch.float32)
    step = powers[(exponent + 128).long()]
    scaled = x / step
    low = torch.floor(scaled)
    fraction = scaled - low
    is_odd = low - 2 * torch.floor(low / 2)
    advance = torch.where(fraction == 0.5, is_odd, (fraction > 0.5).float())
    return (low + advance) * step


def project_layer0_key_even(self, x, ve, cos, sin, capture=None):
    """Copy of Layer.project with only the layer-0 key rounding replaced."""
    import coreml_port.graphs as graphs

    q = graphs.Q(graphs.linear(x, self.q)).reshape(1, -1, 10, 128)
    kraw = graphs.linear(x, self.k)
    k = Q_even(kraw)
    if capture is not None:
        capture(kraw, k)
    k = k.reshape(1, -1, 2, 128)
    v = graphs.Q(graphs.linear(x, self.v)).reshape(1, -1, 2, 128)
    if self.ve:
        gate = graphs.Q(2 * graphs.Q(torch.sigmoid(graphs.Q(graphs.linear(x[..., :12], self.vg)))))
        v = graphs.Q(v + graphs.Q(gate.unsqueeze(-1) * ve))
    q, k = graphs.norm(q), graphs.norm(k)
    if self.rope:
        q = graphs.Q(graphs.Q(cos * q) + graphs.Q(sin * q.reshape(1, -1, 10, 64, 2).flip(-1).reshape(1, -1, 10, 128)))
        k = graphs.Q(graphs.Q(cos * k) + graphs.Q(sin * k.reshape(1, -1, 2, 64, 2).flip(-1).reshape(1, -1, 2, 128)))
    return q, k, v
