"""Experimental source-order softmax for fixed-capacity FP32 attention.

Mirrors pinned Torch CPU's four-lane sum followed by (0+2)+(1+3).
Native exponential implementations may still differ between runtimes.
Use real division: the converter's reciprocal op adds epsilon=1e-4.
Not selected in the ordinary runtime and not a general fidelity guarantee.
"""
import torch
exponential=torch.exp

def softmax(x):
    ex=exponential(x-x.max(-1,keepdim=True).values)
    total=ex[...,:4]
    for j in range(4,x.shape[-1],4):total=total+ex[...,j:j+4]
    total=(total[...,0:1]+total[...,2:3])+(total[...,1:2]+total[...,3:4])
    return ex*(torch.ones_like(total)/total)
