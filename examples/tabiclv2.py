import torch
from sdm import TableTensor
from sdm.models import TabICLv2
from sklearn.datasets import load_breast_cancer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

df = load_breast_cancer(as_frame=True).frame
table = TableTensor(  # TODO Replace with `from_pandas` call.
    columns={"numerical": df.columns[:-1], "categorical": ["target"]},
    numerical=torch.from_numpy(df.to_numpy()[:, :-1]).float(),
    categorical=torch.from_numpy(df["target"].to_numpy()).view(-1, 1),
    device=device,
)
# table = TableTensor.from_pandas(df, device=device)

model = TabICLv2(device=device)

with torch.amp.autocast(device.type, torch.bfloat16, enabled=table.is_cuda):
    model(
        x=table.drop_columns("target"),
        y=table[:300, "target"],
    )
