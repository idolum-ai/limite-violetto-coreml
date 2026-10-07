"""Strictly loaded Violetto with full-prefix and request-local cached execution.

Reuses the pinned Apache-2.0 Paradigma value-model backbone. The actor's tensor
names/scalar layout, mixed dtypes, output head, window convention, and EOS are
taken from the pinned Violetto runtime/config. Sources are never modified.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
from pathlib import Path
import re
import sys
from typing import Callable

import torch
from torch import nn
import torch.nn.functional as F
from safetensors import safe_open
from safetensors.torch import load_file

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'sources/value-model'))
import modeling_backbone as upstream
from configuration_gmpt import GMPTConfig

CHECKPOINT_SHA='9dc7e89e30acf473629f98ac3ade535fb05cc044fe7e399b78d0bc957283a5f5'
EPS=torch.finfo(torch.float32).eps
SCALES={'qkv':'self_attn.qkv_scale','o':'self_attn.o_scale',
        'resid_attn':'resid_lambda_attn','post_attn':'post_lambda_attn',
        'resid_mlp':'resid_lambda_mlp','post_mlp':'post_lambda_mlp'}


def rms(x):
    # Explicit opmath epsilon, verified against Torch 2.11 source in Step 2.
    return F.rms_norm(x,(x.shape[-1],),eps=EPS)


def attention_mask(positions,end,window):
    key_pos=torch.arange(end,device=positions.device)
    mask=key_pos[None,:]<=positions[:,None]
    if window>=0:mask=mask & (key_pos[None,:]>=positions[:,None]-window)
    return mask


def attend(q,keys,values,positions,window,scaling,groups):
    """Original FP32 eager reduction, isolated for numerical diagnosis."""
    mask=attention_mask(positions,keys.shape[1],window)
    expanded_k=keys.repeat_interleave(groups,dim=2)
    expanded_v=values.repeat_interleave(groups,dim=2)
    weights=torch.einsum('bthd,bshd->bhts',q.float(),expanded_k.float())*scaling
    weights=weights.masked_fill(~mask[None,None],float('-inf')).softmax(-1)
    return torch.einsum('bhts,bshd->bthd',weights,expanded_v.float()).to(values.dtype)


def attend_min_rows(q,keys,values,positions,window,scaling,groups,minimum):
    """Keep short CPU queries on the measured consistent reduction path.

    Dummy query rows cannot affect real queries and are discarded. Keys/values,
    their masks and their BF16 rounding stay unchanged. Clamp to the key count
    so full-prefix execution is never padded, even for very short inputs.
    """
    count=q.shape[1];rows=min(minimum,keys.shape[1])
    if count>=rows:return attend(q,keys,values,positions,window,scaling,groups)
    padded=q.new_zeros((q.shape[0],rows,q.shape[2],q.shape[3]))
    padded[:,-count:]=q
    padded_positions=positions.new_zeros(rows);padded_positions[-count:]=positions
    return attend(padded,keys,values,padded_positions,window,scaling,groups)[:,-count:]


def checkpoint_key(name):
    m=re.fullmatch(r'layers\.(\d+)\.scales\.(.+)',name)
    return (f'model.layers.{m[1]}.{SCALES[m[2]]}',True) if m else ('model.'+name,False)


def expected_dtype(name):
    if name.endswith('.weight') or name in {'embed_tokens.weight','value_embeds.weight'}:
        return torch.bfloat16
    return torch.float32


@dataclass
class Cache:
    owner: object
    capacity: int
    layers: int
    length: int=0
    valid: bool=True
    keys: list=field(default_factory=list)
    values: list=field(default_factory=list)
    warmup_ids: torch.Tensor|None=None

    def __post_init__(self):
        self.keys=[None]*self.layers
        self.values=[None]*self.layers

    def reset(self):
        self.keys=[None]*self.layers
        self.values=[None]*self.layers
        self.length=0
        self.valid=True
        self.warmup_ids=None


class Violetto(nn.Module):
    def __init__(self,backbone,config,load_report,capacity=2048,attention_min_query_rows=0):
        super().__init__()
        self.backbone=backbone
        self.config=config
        self.load_report=load_report
        self.capacity=capacity
        self.attention_min_query_rows=attention_min_query_rows
        self.cache_warmup_tokens=16 if self.device.type=='cpu' and attention_min_query_rows else 0
        self._cache_owner=object()

    @property
    def device(self):return self.backbone.embed_tokens.weight.device

    def new_cache(self):return Cache(self._cache_owner,self.capacity,self.config.num_hidden_layers)

    def validate_ids(self,ids,cache=None):
        if ids.dtype!=torch.long or ids.ndim!=2 or ids.shape[0]!=1 or ids.shape[1]<1:
            raise ValueError('Expected nonempty int64 token IDs with shape [1, length].')
        upstream.validate_gmpt_input_ids(ids,self.config.tokenizer_vocab_size)
        previous=0
        if cache is not None:
            if cache.owner is not self._cache_owner:raise ValueError('Cache belongs to another model.')
            if not cache.valid:raise ValueError('Cache invalid after an interrupted forward; reset it.')
            previous=cache.length
        if previous+ids.shape[1]>self.capacity:raise ValueError('Total context exceeds configured capacity; no truncation.')
        if ids.device!=self.device:raise ValueError('Token IDs must be on the model device.')

    @torch.inference_mode()
    def reference_hidden(self,ids):
        """Pinned upstream full-prefix backbone, with the real mapped actor weights."""
        self.validate_ids(ids)
        return self.backbone(input_ids=ids).last_hidden_state

    def scores(self,hidden):
        if self.config.lm_head_precision_mode!='oracle_exact':
            raise ValueError('Only the pinned oracle_exact head is admitted.')
        # Tied head; BF16 projection result, then FP32 sigmoid transformation.
        raw=F.linear(hidden,self.backbone.embed_tokens.weight.to(hidden.dtype))
        c=self.config.softcap_logits
        return c['a']*torch.sigmoid((raw.float()+c['b'])/c['c'])

    @torch.inference_mode()
    def forward(self,ids,cache=None,capture:Callable|None=None):
        self.validate_ids(ids,cache)
        # The source's very short CPU prefixes themselves change under longer
        # matrix shapes. Refresh early KV states until the tested 16-token
        # threshold; subsequent decoding uses only the ordinary cached path.
        if cache is not None and 0<cache.length<self.cache_warmup_tokens:
            if cache.warmup_ids is None:raise ValueError('Cache numerical policy changed; reset the cache.')
            combined=torch.cat((cache.warmup_ids,ids),dim=1)
            fresh=self.new_cache();count=ids.shape[1]
            callback=(lambda name,value:capture(name,value[:,-count:])) if capture else None
            try:
                result=self.forward(combined,fresh,callback)
            except BaseException:
                cache.valid=False
                raise
            cache.keys,cache.values=fresh.keys,fresh.values
            cache.length,cache.valid=fresh.length,fresh.valid
            cache.warmup_ids=fresh.warmup_ids
            return result[:,-count:]
        start=cache.length if cache else 0
        end=start+ids.shape[1]
        positions=torch.arange(start,end,device=self.device,dtype=torch.long)
        cos,sin=upstream.build_rope_factors(self.config,positions[None,:])
        ve=self.backbone._value_embeddings(ids)
        x=rms(self.backbone.embed_tokens(ids))
        history={0:x} if 0 in self.backbone.retained_taps else {}
        if capture:capture('embedding_norm',x)
        try:
            for i,layer in enumerate(self.backbone.layers):
                taps=self.config.tap_indices(i)
                if taps is not None:
                    values=[history[t] if t!=i else x for t in taps]
                    attn_input=rms(self.backbone.mudd(values,x,i))
                    residual=self.backbone.mudd(values,x,i,r_way=True)
                    if capture:
                        capture(f'layer_{i}_mixer_attention',attn_input)
                        capture(f'layer_{i}_mixer_residual',residual)
                else:
                    attn_input=rms(x);residual=x
                attn=layer.self_attn
                q,k,v=attn.project(attn_input,layer.scales.qkv,ve,cos,sin)
                if cache:
                    if cache.keys[i] is None:
                        shape=(1,cache.capacity,attn.num_kv_heads,attn.head_dim)
                        cache.keys[i]=torch.empty(shape,device=k.device,dtype=k.dtype)
                        cache.values[i]=torch.empty(shape,device=v.device,dtype=v.dtype)
                    cache.keys[i][:,start:end].copy_(k)
                    cache.values[i][:,start:end].copy_(v)
                    keys,values=cache.keys[i][:,:end],cache.values[i][:,:end]
                else:keys,values=k,v
                y=attend_min_rows(q,keys,values,positions,attn.window,attn.scaling,attn.num_kv_groups,self.attention_min_query_rows)
                if attn.has_xsa:
                    vn=F.normalize(v.repeat_interleave(attn.num_kv_groups,dim=2).float(),dim=-1,eps=self.config.xsa_normalize_eps)
                    proj=(y.float()*vn).sum(-1,keepdim=True)
                    alpha=torch.tanh(attn.xsa_alpha.float()).view(1,1,attn.num_heads,1)
                    y=y-(alpha*proj*vn).to(y.dtype)
                if attn.attn_gate_channels:
                    gate=attn.attn_gate_scale*torch.sigmoid(F.linear(attn_input[...,:attn.attn_gate_channels],attn.attn_gate.to(attn_input.dtype)))
                    y=y*gate.to(y.dtype).unsqueeze(-1)
                y=y.reshape(1,ids.shape[1],-1)
                out=F.linear(y,attn._scaled(attn.o_proj.weight,layer.scales.o,y.dtype))
                x=layer.scales.resid_attn.to(residual.dtype)*residual+layer.scales.post_attn.to(out.dtype)*out
                mlp=layer.mlp(rms(x))
                x=layer.scales.resid_mlp.to(x.dtype)*x+layer.scales.post_mlp.to(mlp.dtype)*mlp
                if i+1 in self.backbone.retained_taps:history[i+1]=x
                if capture:capture(f'layer_{i}',x)
            x=rms(x)
            if capture:capture('final_hidden',x)
            if cache:
                cache.length=end
                cache.warmup_ids=ids.detach().clone() if end<self.cache_warmup_tokens else None
            return x
        except BaseException:
            if cache:cache.valid=False
            raise


def load_model(device='cpu',capacity=2048,verify_hash=True,attention_min_query_rows=None):
    if attention_min_query_rows is None:attention_min_query_rows=16 if torch.device(device).type=='cpu' else 0
    if not isinstance(attention_min_query_rows,int) or attention_min_query_rows<0:
        raise ValueError('Attention minimum query rows must be a nonnegative integer.')
    path=ROOT/'sources/model/model.safetensors'
    if verify_hash:
        h=hashlib.sha256()
        with path.open('rb') as file:
            for block in iter(lambda:file.read(8*1024*1024),b''):h.update(block)
        if h.hexdigest()!=CHECKPOINT_SHA:raise ValueError('Checkpoint SHA256 mismatch.')
    raw=json.loads((ROOT/'sources/model/config.json').read_text())
    if raw['model_type']!='limite' or raw['final_softcap']!=0 or raw['mudd_mlp'] is not True:
        raise ValueError('This adapter admits only the pinned Violetto architecture.')
    config=GMPTConfig(**raw)
    config._attn_implementation='eager'
    with torch.device('meta'):backbone=upstream.GMPTModel(config)
    targets=dict(backbone.named_parameters())
    mappings={name:checkpoint_key(name) for name in targets}
    expected={source for source,scalar in mappings.values()}
    with safe_open(path,framework='pt',device='cpu') as file:actual=set(file.keys())
    if actual!=expected:
        raise ValueError(f'Checkpoint inventory mismatch: missing={sorted(expected-actual)}, unexpected={sorted(actual-expected)}')
    state=load_file(str(path),device='cpu')
    mapped={};inventory=[]
    for name,param in targets.items():
        source,scalar=mappings[name];value=state[source]
        expected_shape=() if scalar else tuple(param.shape)
        if tuple(value.shape)!=expected_shape:raise ValueError(f'Shape mismatch: {source}')
        if value.dtype!=expected_dtype(name):raise ValueError(f'Dtype mismatch: {source}: {value.dtype}')
        if not torch.isfinite(value).all():raise ValueError(f'Non-finite weight: {source}')
        mapped[name]=value.reshape(1) if scalar else value
        inventory.append({'source':source,'target':name,'source_shape':list(value.shape),'target_shape':list(mapped[name].shape),'dtype':str(value.dtype),'elements':value.numel(),'finite':True})
    backbone.load_state_dict(mapped,strict=True,assign=True)
    # Device movement deliberately supplies no dtype: retain BF16 matrices and
    # FP32 scalars, gates, XSA and mixer tensors exactly as stored.
    backbone.to(device=device).eval().requires_grad_(False)
    report={'checkpoint_sha256':CHECKPOINT_SHA,'hash_verified_this_load':verify_hash,
            'tensor_count':len(inventory),'parameter_count':sum(v['elements'] for v in inventory),
            'missing':[],'unexpected':[],'shape_errors':[],'dtype_errors':[],
            'all_weights_finite':True,'device':str(device),'attention_min_query_rows':attention_min_query_rows,'inventory':inventory}
    return Violetto(backbone,config,report,capacity,attention_min_query_rows).eval().requires_grad_(False)
