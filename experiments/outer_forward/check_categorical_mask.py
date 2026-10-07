import torch
from sdm import TableTensor,Stype
from sdm.models.base import _categorical_mask
from sdm.processing.execution import MemberContext
for fullgraph in (False,True):
 torch._dynamo.reset()
 compiled=torch.compile(_categorical_mask,fullgraph=fullgraph,dynamic=True)
 for rows,names,flags,count in [(4,('a','b'),(True,False),1),(7,('b','a'),(False,True),1),(5,('a','b'),(True,True),2)]:
  table=TableTensor(columns={'numerical':names},numerical=torch.randn(rows,2))
  target=TableTensor.from_tensor(torch.randn(rows,1))
  members=[MemberContext(table,target,None,{n:Stype.categorical if v else Stype.numerical for n,v in zip(names,flags)}) for _ in range(count)]
  with torch.inference_mode():
   torch.testing.assert_close(compiled(members),_categorical_mask(members),atol=0,rtol=0)
 print(torch.__version__,fullgraph,'pass changing rows/schema/member count')
