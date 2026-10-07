import contextlib,json,torch
from sdm._inference import inference_mode

def state(): return torch.is_grad_enabled(),torch.is_inference_mode_enabled()
count=0
for ambient_name,ambient in [('grad',torch.enable_grad),('no_grad',torch.no_grad),('inference',torch.inference_mode)]:
 for mode in ['inference','no_grad','grad','none']:
  for fail in [False,True]:
   start=state()
   with ambient():
    before=state()
    manager=inference_mode(mode)
    assert state()==before,('construction',ambient_name,mode)
    try:
     with manager as entered:
      assert entered is None
      actual=state()
      expected={'inference':(False,True),'no_grad':(False,before[1]),'grad':(True,before[1]),'none':before}[mode]
      assert actual==expected,(ambient_name,mode,actual,expected)
      if fail: raise ValueError('sentinel')
    except ValueError as ex:
     assert str(ex)=='sentinel'
    assert state()==before,('restore',ambient_name,mode)
   assert state()==start
   count+=1
print(json.dumps({'torch':torch.__version__,'nested_and_exception_cases':count,'pass':True}))
