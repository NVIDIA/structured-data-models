# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Check actual Kumo predictions across source trees; never time inference."""

import argparse
import hashlib
import json
from pathlib import Path

import torch

import sdm
from sdm.models import KumoTabular


@torch.inference_mode()
def main() -> None:
    """Save baseline predictions or compare predictions with that baseline."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifacts", type=Path, required=True)
    parser.add_argument(
        "--phase", choices=("baseline", "after"), required=True
    )
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    torch.set_num_threads(4)
    device = torch.device(args.device)
    data_generator = torch.Generator().manual_seed(797)
    numerical = torch.randn(48, 8, generator=data_generator)
    numerical[::7, 1] = torch.nan
    x = sdm.TableTensor(
        numerical=numerical,
        categorical=sdm.CategoricalTensor(
            torch.randint(12, (48, 1), generator=data_generator),
            categories=(torch.arange(12),),
        ),
    ).to(device)
    results = {}
    for task in ("classification", "regression"):
        # Identical non-constant weights make this sensitive to preprocessing.
        torch.manual_seed(1344)
        model = KumoTabular(task=task, size="small", pretrained=False)
        for parameter in model.parameters():
            if not parameter.any():
                torch.nn.init.normal_(parameter, std=0.02)
        weights = hashlib.sha256(
            b"".join(p.detach().numpy().tobytes() for p in model.parameters())
        ).hexdigest()
        model = model.to(device).eval()
        y = (
            sdm.TableTensor(
                categorical=sdm.CategoricalTensor(
                    torch.arange(32).remainder(3).view(-1, 1),
                    categories=(torch.tensor([10, 20, 30]),),
                )
            )
            if task == "classification"
            else sdm.TableTensor.from_tensor(
                torch.linspace(-2, 3, 32).view(-1, 1)
            )
        ).to(device)
        forward = model(
            x[:32],
            y,
            x[32:],
            num_estimators=4,
            generator=torch.Generator(device=device).manual_seed(1361),
        )
        model.fit(
            x[:32],
            y,
            num_estimators=4,
            generator=torch.Generator(device=device).manual_seed(1361),
        )
        prediction = model.predict(x[32:])
        results[task] = {
            "weights_sha256": weights,
            "forward": forward.cpu(),
            "fit_predict": prediction.cpu(),
        }
    path = args.artifacts / f"prediction-{args.device}-baseline.pt"
    if args.phase == "baseline":
        if path.exists():
            raise FileExistsError(path)
        torch.save(results, path)
    else:
        reference = torch.load(path, weights_only=False)
        for task, result in results.items():
            assert (
                result["weights_sha256"] == reference[task]["weights_sha256"]
            )
            for operation in ("forward", "fit_predict"):
                actual, expected = (
                    result[operation],
                    reference[task][operation],
                )
                assert actual.column_groups == expected.column_groups
                torch.testing.assert_close(
                    actual.numerical, expected.numerical, rtol=1e-5, atol=1e-6
                )
        torch.save(
            results, args.artifacts / f"prediction-{args.device}-after.pt"
        )
    print(
        json.dumps(
            {"phase": args.phase, "source": sdm.__file__, "status": "passed"}
        )
    )


if __name__ == "__main__":
    main()
