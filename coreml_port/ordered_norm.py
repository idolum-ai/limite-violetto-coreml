"""Experimental pinned CPU sum order for RMS widths 128 and 1280.

PyTorch SumKernel.cpp: vector width 4, four interleaved vector accumulators,
chunks of 16 accumulator rows, then sequential vector and scalar folds.
This bounded implementation only covers the two model normalization widths.
"""
import torch
from portable.model import EPS

def ordered_sum(x):
    width=x.shape[-1]
    if width not in (128,1280):raise ValueError('Only model RMS widths 128/1280 are supported.')
    total=None
    for start in range(0,width,256):
        chunk=x[...,start:start+16]
        for j in range(start+16,min(start+256,width),16):chunk=chunk+x[...,j:j+16]
        total=chunk if total is None else total+chunk
    folded=((total[...,:4]+total[...,4:8])+total[...,8:12])+total[...,12:16]
    return ((folded[...,0:1]+folded[...,1:2])+folded[...,2:3])+folded[...,3:4]

def raw_norm(x,divisor):
    mean=ordered_sum(x*x)/divisor
    denom=torch.sqrt(mean+EPS)
    return x*(torch.ones_like(denom)/denom)

class NormWrapped(torch.nn.Module):
    """Trace all RMS calls with a runtime width input, preserving real division."""
    def __init__(self,inner):super().__init__();self.inner=inner
    def forward(self,*args):
        from . import graphs
        divisor=args[-1];original=graphs.norm
        def norm(value):
            width_divisor=divisor*.1 if value.shape[-1]==128 else divisor
            return graphs.Q(raw_norm(value,width_divisor))
        graphs.norm=norm
        try:return self.inner(*args[:-1])
        finally:graphs.norm=original
