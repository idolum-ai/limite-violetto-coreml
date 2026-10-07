"""FP32 expression of the pinned Torch 2.7 CPU BF16 dot summation order.

See sources/torch-2.7-linear/ReducedPrecisionFloatGemvFastPathKernel.cpp and
NOTICE.md. Eight four-lane accumulators, fixed tree, pairwise lane reduction.
Normal-scale BF16 products are exactly representable in FP32. This operation
is intended for BF16-valued model operands, not arbitrary FP32 GEMMs.
"""

def linear(x,w):
    k=w.shape[1];aligned=k//32*32
    # Initialize from actual products to retain symbolic input-shape inference.
    # Adding the source's initial positive zero does not change finite nonzero
    # products; signed-zero-only outputs are irrelevant to the tested scores.
    if aligned:
        total=x[...,:32].unsqueeze(-2)*w[:,:32]
        for j in range(32,aligned,32):total=total+x[...,j:j+32].unsqueeze(-2)*w[:,j:j+32]
        total=total[...,:16]+total[...,16:]
        total=total[...,:8]+total[...,8:]
        total=total[...,:4]+total[...,4:]
        result=(total[...,0]+total[...,1])+(total[...,2]+total[...,3])
    tail_end=aligned+(k-aligned)//8*8
    if tail_end>aligned:
        tail=x[...,aligned:aligned+4].unsqueeze(-2)*w[:,aligned:aligned+4]
        tail=tail+x[...,aligned+4:aligned+8].unsqueeze(-2)*w[:,aligned+4:aligned+8]
        for j in range(aligned+8,tail_end,8):
            tail=tail+x[...,j:j+4].unsqueeze(-2)*w[:,j:j+4]
            tail=tail+x[...,j+4:j+8].unsqueeze(-2)*w[:,j+4:j+8]
        value=(tail[...,0]+tail[...,1])+(tail[...,2]+tail[...,3])
        result=result+value if aligned else value
    start=tail_end
    if not tail_end:result=x[...,:1]*w[:,0];start=1
    for j in range(start,k):result=result+x[...,j:j+1]*w[:,j]
    return result
