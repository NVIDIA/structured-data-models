import torch

import sdm
from sdm.processing import ShuffleColumns
from sdm.tensor import EnsembleTable

device, columns = torch.device("cuda"), ("a", "b", "c", "d")
table = sdm.TableTensor.from_tensor(
    torch.arange(8, dtype=torch.float32, device=device).reshape(2, 4),
    columns=columns,
)
ensemble = EnsembleTable(table, num_members=4)
processor = ShuffleColumns(method="latin")
output = processor.fit_transform_ensemble(
    ensemble,
    generator=torch.Generator(device=device).manual_seed(7),
)
orders = [output.table(i).columns[sdm.Stype.numerical] for i in range(4)]
assert all({order[i] for order in orders} == set(columns) for i in range(4))

restored = ShuffleColumns(method="latin")
restored.load_state_dict(processor.state_dict())
again = restored.transform_ensemble(ensemble)
assert all(output.table(i).equal(again.table(i)) for i in range(4))
print("latin=True, state_dict=True")
