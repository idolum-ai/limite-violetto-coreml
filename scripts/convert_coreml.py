"""Convert one bounded full-model component. Run separate processes per shard."""
import argparse,gc,json,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import torch,numpy as np,coremltools as ct
from portable.model import load_model
from coreml_port.graphs import Embedding,Head,Group
p=argparse.ArgumentParser();p.add_argument('component');p.add_argument('--arithmetic',choices=['fp32','ordered'],default='fp32');p.add_argument('--softmax',choices=['native','ordered'],default='native');p.add_argument('--exp',choices=['native','source'],default='native');p.add_argument('--norm',choices=['native','ordered'],default='native');p.add_argument('--xsa',choices=['native','ordered'],default='native');p.add_argument('--gelu',choices=['native','lookup'],default='native');p.add_argument('--score-dot',choices=['default','tiny-sequential','tiny-native'],default='default');p.add_argument('--product',choices=['default','paired'],default='default');p.add_argument('--queries',choices=['static','dynamic-product'],default='static');p.add_argument('--output-dir',default='artifacts/04-coreml/model');p.add_argument('--evidence-dir',default='evidence/04-coreml');args=p.parse_args()
if args.queries=='dynamic-product':
    import coreml_port.graphs as graphs
    from coreml_port.dynamic_queries import prepare,scores,product
    graphs.prepare_queries=prepare;graphs.attention_scores=scores;graphs.attention_product=product
if args.product=='paired':
    if args.queries!='dynamic-product':p.error('--product paired requires --queries dynamic-product')
    from coreml_port.paired_product import product
    graphs.attention_product=product
if args.score_dot=='tiny-sequential':
    if args.queries!='dynamic-product':p.error('--score-dot tiny-sequential requires --queries dynamic-product')
    from coreml_port.tiny_scores import scores
    graphs.attention_scores=scores
if args.score_dot=='tiny-native':
    if args.queries!='dynamic-product':p.error('--score-dot tiny-native requires --queries dynamic-product')
    from coreml_port.tiny_native_scores import scores
    graphs.attention_scores=scores
if args.exp=='source':
    if args.softmax!='ordered':p.error('--exp source requires --softmax ordered')
    import coreml_port.ordered_attention as attention
    from coreml_port.source_exp import exp
    attention.exponential=exp
if args.arithmetic=='ordered':
    import coreml_port.graphs as graphs
    from coreml_port.ordered_linear import linear
    graphs.linear=linear
if args.softmax=='ordered':
    import coreml_port.graphs as graphs
    from coreml_port.ordered_attention import softmax
    graphs.softmax=softmax
if args.xsa=='ordered':
    import coreml_port.graphs as graphs
    from coreml_port.ordered_xsa import ordered_xsa_normalize,ordered_xsa_dot
    graphs.xsa_normalize=ordered_xsa_normalize;graphs.xsa_dot=ordered_xsa_dot
if args.gelu=='lookup':
    import coreml_port.graphs as graphs
    from coreml_port.bf16_gelu import BF16Gelu
    graphs.gelu_factory=BF16Gelu
torch.set_num_threads(4);begin=time.time();m=load_model();name=args.component
seq=ct.RangeDim(1,32,default=3)
def tensor(n,shape,dtype=np.float32):return ct.TensorType(name=n,shape=shape,dtype=dtype)
if name=='embedding':
    graph=Embedding(m);sample=(torch.ones(1,3,dtype=torch.int32),);inputs=[tensor('ids',(1,seq),np.int32)];outputs=['x','ve']
elif name=='head':
    graph=Head(m);sample=(torch.zeros(1,1,1280),);inputs=[tensor('x',(1,1,1280))];outputs=['scores']
else:
    graph=Group(m,int(name)*4);x=torch.randn(1,3,1280)*.1
    sample=(x,torch.zeros(1,3,2,128),torch.arange(3,dtype=torch.int32),x.clone(),x.clone(),x.clone())
    inputs=[tensor('x',(1,seq,1280)),tensor('ve',(1,seq,2,128)),tensor('positions',(seq,),np.int32)]+[tensor(n,(1,seq,1280)) for n in ['tap0','tap12','tap23']];outputs=['y','tap23_out']
if args.norm=='ordered' and name!='head':
    from coreml_port.ordered_norm import NormWrapped
    graph=NormWrapped(graph);sample=(*sample,torch.tensor([1280.]));inputs.append(tensor('norm_divisor',(1,)))
del m;gc.collect();graph.eval();print('loaded',name,flush=True)
with torch.inference_mode():traced=torch.jit.trace(graph,sample,check_trace=False)
states=[ct.StateType(name=n,wrapped_type=ct.TensorType(shape=v.shape,dtype=np.float16)) for n,v in graph.named_buffers() if '_cache_' in n]
model=ct.convert(traced,inputs=inputs,outputs=[tensor(n,None) for n in outputs],states=states,convert_to='mlprogram',compute_precision=ct.precision.FLOAT32,minimum_deployment_target=ct.target.macOS15,skip_model_load=True)
path=ROOT/args.output_dir/f'{name}.mlpackage';path.parent.mkdir(exist_ok=True,parents=True);model.save(str(path))
report={'component':name,'arithmetic':args.arithmetic,'softmax':args.softmax,'exp':args.exp,'norm':args.norm,'xsa':args.xsa,'gelu':args.gelu,'queries':args.queries,'product':args.product,'score_dot':args.score_dot,'seconds':time.time()-begin,'package':str(path.relative_to(ROOT)),'state_names':[s.name for s in states],'status':'converted; runtime validation pending'}
(ROOT/args.evidence_dir/f'convert-{name}.json').write_text(json.dumps(report,indent=2)+'\n');print(report,flush=True)
