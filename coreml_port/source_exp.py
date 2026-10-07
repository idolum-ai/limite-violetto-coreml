"""Experimental nonpositive source exponential using compensated FP32 FMA.

Polynomial/range reduction derived from SLEEF xexpf at commit
56e1f79cb140fb9326d612d0be06b5250565cade. Copyright Naoki Shibata and
contributors 2010-2024. Boost Software License retained in sources/sleef-pinned.
Only intended for finite/nonpositive softmax arguments and masked -infinity.
Compensation is empirically qualified on the recorded domain, not general FMA.
"""
import torch

def fma(a,b,c):
    a=a if isinstance(a,torch.Tensor) else torch.tensor(a,dtype=torch.float32)
    b=b if isinstance(b,torch.Tensor) else torch.tensor(b,dtype=torch.float32)
    c=c if isinstance(c,torch.Tensor) else torch.tensor(c,dtype=torch.float32)
    ta=a*4097.;ah=ta-(ta-a);al=a-ah
    tb=b*4097.;bh=tb-(tb-b);bl=b-bh
    product=a*b
    e1=product-ah*bh;e2=e1-al*bh;e3=e2-ah*bl;error=al*bl-e3
    total=product+c;bb=total-product
    remainder=(product-(total-bb))+(c-bb)
    return total+(error+remainder)

def exp(x):
    d=torch.clamp(x,min=-104,max=0)
    q=torch.round(d*1.4426950408889634)
    s=fma(q,-0.693145751953125,d)
    s=fma(q,-1.428606765330187e-6,s)
    u=0.00019852761761285365
    for c in [0.0013930435525253415,0.008333360776305199,0.041666485369205475,0.1666666716337204,0.5]:u=fma(u,s,c)
    u=1+fma(s*s,u,s)
    half=torch.floor(q*.5);powers=2.**torch.arange(-80,1,dtype=torch.float32)
    u=(u*powers[(half+80).long()])*powers[(q-half+80).long()]
    return torch.where(x < -104,torch.zeros_like(x),u)
