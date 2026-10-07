"""Whole-model Core ML candidate; explicit finite-range BF16 rounding.

Derived from the pinned Apache-2.0 GMPT backbone. FP32 weight constants preserve
BF16 source values. Stateful KV is FP16, an independently measured difference.
"""
import torch
from torch import nn
import torch.nn.functional as F
from portable.model import EPS, upstream

# Injection point for precision experiments; default is unchanged Torch linear.
linear = F.linear
softmax = lambda x: x.softmax(-1)
xsa_normalize = lambda x: F.normalize(x,dim=-1,eps=1e-4)
xsa_dot = lambda y,vn: (y*vn).sum(-1,keepdim=True)
gelu_factory = nn.GELU


def prepare_queries(q, positions):
    # Historical policy; experimental conversions inject their own geometry.
    return (torch.cat((q,torch.zeros(1,31,10,128)),dim=1)[:,:32],
            torch.cat((positions,torch.zeros(31,dtype=torch.int32)))[:32])


def attention_scores(q, keys):
    return torch.matmul(q.permute(0,2,1,3),keys.permute(0,2,3,1))*.1


def attention_product(weights, values, count):
    return torch.matmul(weights,values).permute(0,2,1,3)[:,:count]


def Q(x):
    exponent=torch.floor(torch.log2(torch.clamp(x.abs(),min=1e-30)))-7
    powers=2.**torch.arange(-128,128,dtype=torch.float32)
    step=powers[(exponent+128).long()]
    return torch.round(x/step)*step


def norm(x):return Q(x*torch.rsqrt((x*x).mean(-1,keepdim=True)+EPS))


def buf(module,name,value):module.register_buffer(name,value.detach().float().clone())


class Embedding(nn.Module):
    def __init__(self,m):
        super().__init__();buf(self,'tokens',m.backbone.embed_tokens.weight);buf(self,'values',m.backbone.value_embeds.weight)
    def forward(self,ids):return norm(F.embedding(ids.long(),self.tokens)),F.embedding(ids.long(),self.values).reshape(1,-1,2,128)


class Head(nn.Module):
    def __init__(self,m):super().__init__();buf(self,'weight',m.backbone.embed_tokens.weight)
    def forward(self,x):return 23*torch.sigmoid((Q(linear(x,self.weight))+5)/7.5)


class Layer(nn.Module):
    def __init__(self,m,i):
        super().__init__();l=m.backbone.layers[i];a=l.self_attn
        self.i=i;self.window=a.window;self.rope=a.applies_rope;self.ve=a.has_ve
        for n in ['q','k','v','o']:buf(self,n,a._scaled(getattr(a,n+'_proj').weight,l.scales.o if n=='o' else l.scales.qkv,torch.bfloat16))
        for n in ['up','gate','down']:buf(self,n,getattr(l.mlp,n+'_proj').weight)
        for n in ['resid_attn','post_attn','resid_mlp','post_mlp']:buf(self,n,getattr(l.scales,n).bfloat16())
        buf(self,'alpha',torch.tanh(a.xsa_alpha).reshape(1,1,10,1));buf(self,'ag',a.attn_gate.bfloat16())
        if self.ve:buf(self,'vg',a.ve_gate.bfloat16())
        if i in [24,47]:
            self.gelu_fn=gelu_factory()
            mix=m.backbone.mudd
            buf(self,'dense1',mix.dense1.bfloat16())
            for n in ['dense2','bias','dense2_mlp','bias_mlp']:buf(self,n,getattr(mix,n)[i].bfloat16())
    def mix(self,x,t0,t1,r=False):
        inner=Q(self.gelu_fn(Q(linear(norm(x),self.dense1))))
        w=Q(Q(linear(inner,self.dense2_mlp if r else self.dense2))+(self.bias_mlp if r else self.bias))
        return Q(Q(Q(w[...,0:1]*t0)+Q(w[...,1:2]*t1))+Q(w[...,2:3]*x))
    def project(self,x,ve,cos,sin):
        q=Q(linear(x,self.q)).reshape(1,-1,10,128);k=Q(linear(x,self.k)).reshape(1,-1,2,128);v=Q(linear(x,self.v)).reshape(1,-1,2,128)
        if self.ve:
            gate=Q(2*Q(torch.sigmoid(Q(linear(x[...,:12],self.vg)))))
            v=Q(v+Q(gate.unsqueeze(-1)*ve))
        q,k=norm(q),norm(k)
        if self.rope:
            q=Q(Q(cos*q)+Q(sin*q.reshape(1,-1,10,64,2).flip(-1).reshape(1,-1,10,128)))
            k=Q(Q(cos*k)+Q(sin*k.reshape(1,-1,2,64,2).flip(-1).reshape(1,-1,2,128)))
        return q,k,v
    def finish(self,x,residual,q,keys,values,v,positions,slots):
        count=q.shape[1];q,pos=prepare_queries(q,positions)
        k=keys.repeat_interleave(5,dim=2);val=values.repeat_interleave(5,dim=2).permute(0,2,1,3)
        scores=attention_scores(q,k)
        mask=slots[None,:]<=pos[:,None]
        if self.window>=0:mask=mask & (slots[None,:]>=pos[:,None]-self.window)
        weights=softmax(scores.masked_fill(~mask[None,None],float('-inf')))
        y=Q(attention_product(weights,val,count))
        vn=xsa_normalize(v.repeat_interleave(5,dim=2))
        y=Q(y-Q(self.alpha*xsa_dot(y,vn)*vn))
        gate=Q(2*Q(torch.sigmoid(Q(linear(x[...,:128],self.ag)))))
        y=Q(y*gate.unsqueeze(-1))
        out=Q(linear(y.reshape(1,-1,1280),self.o))
        mixed=Q(Q(self.resid_attn*residual)+Q(self.post_attn*out));h=norm(mixed)
        activated=Q(Q(F.silu(Q(linear(h,self.gate))))*Q(linear(h,self.up)))
        mlp=Q(linear(activated,self.down))
        return Q(Q(self.resid_mlp*mixed)+Q(self.post_mlp*mlp))


class Group(nn.Module):
    def __init__(self,m,start,capacity=2048):
        super().__init__();self.start=start;self.layers=nn.ModuleList([Layer(m,i) for i in range(start,start+4)])
        cos,sin=upstream.build_rope_factors(m.config,torch.arange(capacity)[None,:])
        buf(self,'cos',cos[0,:,0]);buf(self,'sin',sin[0,:,0]);self.register_buffer('slots',torch.arange(capacity,dtype=torch.int32))
        for j in range(4):
            self.register_buffer(f'k_cache_{j}',torch.zeros(1,capacity,2,128,dtype=torch.float16))
            self.register_buffer(f'v_cache_{j}',torch.zeros(1,capacity,2,128,dtype=torch.float16))
    def forward(self,x,ve,positions,tap0,tap12,tap23):
        cos=self.cos[positions.long()].reshape(1,-1,1,128);sin=self.sin[positions.long()].reshape(1,-1,1,128)
        start,end=positions[0],positions[-1]+1
        for j,l in enumerate(self.layers):
            if l.i in [24,47]:
                tap=tap12 if l.i==24 else tap23
                a=norm(l.mix(x,tap0,tap));r=l.mix(x,tap0,tap,True)
            else:a=norm(x);r=x
            q,k,v=l.project(a,ve,cos,sin)
            kc=getattr(self,f'k_cache_{j}');vc=getattr(self,f'v_cache_{j}')
            kc[:,start:end]=k;vc[:,start:end]=v
            x=l.finish(a,r,q,kc.float(),vc.float(),v,positions,self.slots)
            if l.i==22:tap23=x
        if self.start==44:x=norm(x)
        return x,tap23+0.
