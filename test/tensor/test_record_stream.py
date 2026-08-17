import torch

from sdm import (
    CategoricalTensor,
    ColumnarTensor,
    NullableTensor,
    StringTensor,
    TableTensor,
    VarLenTensor,
)
from sdm.testing import onlyCUDA


@onlyCUDA
def test_record_stream() -> None:
    device = torch.device("cuda")
    tensors = (
        CategoricalTensor(
            code=torch.tensor([[0]], device=device),
            categories=(torch.tensor([1], device=device),),
        ),
        ColumnarTensor((torch.tensor([1], device=device),)),
        NullableTensor.from_list([1, None], device=device),
        StringTensor.from_list(["hi", None], device=device),
        TableTensor.from_tensor(torch.tensor([[1.0]], device=device)),
        VarLenTensor.from_list([[1], None], device=device),
    )
    stream = torch.cuda.Stream()

    for tensor in tensors:
        assert tensor.record_stream(stream) is None
