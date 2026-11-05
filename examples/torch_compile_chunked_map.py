# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import torch

from sdm.models.tabfm.cell_embedding import CellEmbedding

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = (
    CellEmbedding(channels=8, group_size=2, num_frequencies=4)
    .to(device)
    .eval()
)
compiled = torch.compile(
    model, backend="inductor", fullgraph=True, dynamic=None
)
mask = torch.tensor([False, True, False], device=device)

print(f"PyTorch {torch.__version__}, device={device}", flush=True)
with torch.inference_mode():
    for call, rows in enumerate((7, 12, 20, 35, 7, 64, 81), start=1):
        x = torch.randn(rows, 3, device=device)
        expected = model(x, mask, batch_size_limit=48)
        print(f"BEGIN call={call} rows={rows}", flush=True)
        if call > 3:
            with torch.compiler.set_stance("fail_on_recompile"):
                actual = compiled(x, mask, batch_size_limit=48)
        else:
            actual = compiled(x, mask, batch_size_limit=48)
        torch.testing.assert_close(actual, expected)
        graphs = torch._dynamo.utils.counters["stats"]["unique_graphs"]
        print(
            f"END call={call} graphs={graphs} correctness=passed", flush=True
        )
