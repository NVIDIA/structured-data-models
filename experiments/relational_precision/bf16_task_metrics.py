"""Score saved same-dtype predictions on the exact validation query subsets."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import accuracy_score, brier_score_loss, roc_auc_score

from sdm import CategoricalTensor

parser = argparse.ArgumentParser()
parser.add_argument("--data", required=True)
parser.add_argument("--evidence", required=True)
parser.add_argument("--output", required=True)
args = parser.parse_args()
data = np.load(args.data)
labels = data["y"][data["validation_ids"]]
context_labels = torch.tensor(data["y"][data["train_ids"][:256]]).view(-1, 1)
classes = CategoricalTensor.from_tensor(context_labels).categories[0].tolist()
assert classes == [0, 1], classes


def metrics(prediction, rows):
    prediction = prediction.double().numpy()
    target = labels[:rows]
    return {
        "accuracy": accuracy_score(target, prediction.argmax(-1)),
        "roc_auc": roc_auc_score(target, prediction[:, 1]),
        "log_loss": float(-np.log(prediction[np.arange(rows), target]).mean()),
        "brier": brier_score_loss(target, prediction[:, 1]),
    }


result = {"classes": classes, "scope": "First 32/128 validation rows, 256 context rows", "cases": []}
for name in ["inner-control", "inner-backend-eager", "inner-emulate-casts"]:
    path = Path(args.evidence) / f"tabular-bf16-{name}.predictions.pt"
    saved = torch.load(path, weights_only=False)
    seen = set()
    for key, pair in saved.items():
        phase, rows, _ = key.split("-")
        rows = int(rows)
        if phase != "compiled" or rows in seen:
            continue
        seen.add(rows)
        actual, expected = pair["actual"], pair["reference"]
        eager, compiled = metrics(expected, rows), metrics(actual, rows)
        result["cases"].append({
            "arm": name, "rows": rows,
            "label_counts": np.bincount(labels[:rows], minlength=2).tolist(),
            "eager": eager, "compiled": compiled,
            "compiled_minus_eager": {key: compiled[key] - eager[key] for key in eager},
            "max_probability_difference": (actual - expected).abs().max().item(),
            "failed_probability_values": (~torch.isclose(actual, expected, atol=1e-5, rtol=1e-4)).sum().item(),
            "changed_classes": (actual.argmax(-1) != expected.argmax(-1)).sum().item(),
        })
Path(args.output).write_text(json.dumps(result, indent=2) + "\n")
print(json.dumps(result, indent=2))
