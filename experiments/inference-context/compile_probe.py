import contextlib, json, torch
from sdm._inference import inference_mode

rows=[]
@torch.compiler.disable
def eager_identity(x): return x
for mode in ['inference','no_grad','grad','none']:
 for fullgraph in [False, True]:
  torch._dynamo.reset()
  def fn(x):
   with inference_mode(mode):
    y=x.sin()
    if not fullgraph: y=eager_identity(y)
    return y.cos()
  try:
   compiled=torch.compile(fn,fullgraph=fullgraph)
   x=torch.randn(8,requires_grad=True)
   actual=compiled(x)
   expected=fn(x)
   torch.testing.assert_close(actual,expected)
   assert actual.requires_grad==expected.requires_grad
   rows.append({'mode':mode,'fullgraph':fullgraph,'pass':True})
  except Exception as ex:
   rows.append({'mode':mode,'fullgraph':fullgraph,'pass':False,'error':str(ex)[:1600]})
print(json.dumps({'torch':torch.__version__,'results':rows},indent=2))
