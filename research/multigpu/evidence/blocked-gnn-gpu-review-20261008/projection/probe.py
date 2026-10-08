import json
from dataclasses import replace
from pathlib import Path
import torch
from sdm._kernels import segment_multi_reduce
from sdm.models.kumo.relational.invariant_gnn import InvariantGNN
from research.multigpu.blocked_gnn import destination_statistics
from research.multigpu.test_blocked_gnn_review import graph_from_degrees
state=torch.load('/home/ubuntu/kumo-multigpu/models/hub/models--nvidia--Kumo-Relational/snapshots/2bd603d3d8f25f67a7aaa8579908595e20567e22/classifier.pt',map_location='cpu',weights_only=True)
m=InvariantGNN(512).eval(); m.load_state_dict({k[4:]:v for k,v in state.items() if k.startswith('gnn.')}); m.cuda()
g=graph_from_degrees(torch.tensor([0,1,3,17]).repeat(65)[:257]); g=replace(g,row=g.row.cuda(),col=g.col.cuda(),colptr=g.colptr.cuda(),edge_type=g.edge_type.cuda())
x=torch.arange(257*512).reshape(257,512).float().mul(.01).sin().cuda()
for precision in ['fp32','autocast_bf16']:
 with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16,enabled=precision=='autocast_bf16'):
  edge=m.get_edge_type_emb(3,dtype=x.dtype,generator=torch.Generator(device='cuda').manual_seed(1729)); src=m.src_lin(x)
  fullstats=segment_multi_reduce(src,g.row,edge,g.colptr,g.edge_type)
  chunks=[destination_statistics(src,g,edge,a,min(a+64,257)) for a in range(0,257,64)]; cat=torch.cat(chunks)
  skip=m.skip_lin(x); full=torch.addmm(skip,fullstats.flatten(1),m.aggregation_lin.weight.T)
  blocked=torch.cat([torch.addmm(skip[a:min(a+64,257)],chunks[i].flatten(1),m.aggregation_lin.weight.T) for i,a in enumerate(range(0,257,64))])
  print(json.dumps({'precision':precision,'statistics_equal':torch.equal(cat,fullstats),'statistics_max_error':float((cat.float()-fullstats.float()).abs().max()),'first_projection_max_error':float((full.float()-blocked.float()).abs().max()),'first_projection_mean_error':float((full.float()-blocked.float()).abs().mean()),'projection_dtype':str(full.dtype),'allow_tf32':torch.backends.cuda.matmul.allow_tf32}))
