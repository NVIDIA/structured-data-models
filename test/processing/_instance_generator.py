from collections.abc import Callable
from dataclasses import dataclass

import torch

import sdm.processing as sp
from sdm import (
    CategoricalTensor,
    ColumnarTensor,
    StringTensor,
    Stype,
    TableTensor,
)


@dataclass(frozen=True)
class ProcessorInputs:
    fit: TableTensor
    query: TableTensor


InputFactory = Callable[[torch.device, torch.dtype], ProcessorInputs]


def make_mixed_inputs(
    device: torch.device | None = None,
    dtype: torch.dtype = torch.float32,
) -> ProcessorInputs:
    device = device or torch.device("cpu")
    columns = {
        "numerical": ("value", "constant", "other"),
        "categorical": ("category",),
        "datetime": ("timestamp",),
        "text": ("document",),
        "id": ("row_id",),
    }
    categories = (StringTensor.from_list(["a", "b"], device=device),)
    numerical = torch.arange(12, device=device, dtype=dtype).reshape(4, 3)
    numerical[:, 1] = 1
    fit = TableTensor(
        columns=columns,
        numerical=numerical,
        categorical=CategoricalTensor(
            code=torch.tensor(
                [[0], [0], [1], [-1]], dtype=torch.int32, device=device
            ),
            categories=categories,
        ),
        datetime=torch.arange(4, device=device, dtype=torch.int64)[:, None]
        * 86_400_000_000,
        text=StringTensor.from_list(
            [["alpha beta"], ["beta"], ["gamma"], ["alpha"]],
            device=device,
        ),
        id=ColumnarTensor((torch.arange(10, 14, device=device),)),
    )
    query = fit.replace_blocks(
        numerical=numerical + 2,
        categorical=CategoricalTensor(
            code=torch.tensor(
                [[-1], [1], [0], [-1]], dtype=torch.int32, device=device
            ),
            categories=categories,
        ),
        datetime=fit.datetime + 60_000_000,
        text=StringTensor.from_list(
            [["alpha"], ["beta gamma"], ["unseen"], [""]],
            device=device,
        ),
        id=ColumnarTensor((torch.arange(20, 24, device=device),)),
    )
    return ProcessorInputs(fit=fit, query=query)


def _make_impute_mean_inputs(
    device: torch.device,
    dtype: torch.dtype,
) -> ProcessorInputs:
    inputs = make_mixed_inputs(device, dtype)
    numerical = inputs.query.numerical.clone()
    numerical[0, 0] = float("nan")
    return ProcessorInputs(
        fit=inputs.fit,
        query=inputs.query.replace_blocks(numerical=numerical),
    )


def _make_align_categories_inputs(
    device: torch.device,
    dtype: torch.dtype,
) -> ProcessorInputs:
    inputs = make_mixed_inputs(device, dtype)
    categorical = CategoricalTensor(
        code=torch.tensor(
            [[0], [1], [-1], [0]], dtype=torch.int32, device=device
        ),
        categories=(StringTensor.from_list(["b", "c"], device=device),),
    )
    return ProcessorInputs(
        fit=inputs.fit,
        query=inputs.query.replace_blocks(categorical=categorical),
    )


def _make_reduction_inputs(
    device: torch.device,
    dtype: torch.dtype,
) -> ProcessorInputs:
    fit = TableTensor.from_tensor(
        torch.tensor(
            [[0.0, 1.0], [2.0, 3.0], [4.0, 5.0], [6.0, 7.0]],
            device=device,
            dtype=dtype,
        )
    )
    query = TableTensor.from_tensor(fit.numerical + 1.0)
    return ProcessorInputs(fit=fit, query=query)


def _identity(table: TableTensor) -> TableTensor:
    return table


@dataclass(frozen=True)
class ProcessorContractCase:
    name: str
    factory: Callable[[], sp.Processor]
    make_inputs: InputFactory = make_mixed_inputs


PROCESSOR_CONTRACT_CASES = (
    ProcessorContractCase("identity", sp.Identity),
    ProcessorContractCase("callable", lambda: sp.Callable(_identity)),
    ProcessorContractCase("drop-stypes", lambda: sp.DropStypes(Stype.id)),
    ProcessorContractCase("to-numerical", sp.ToNumerical),
    ProcessorContractCase("shuffle-columns", sp.ShuffleColumns),
    ProcessorContractCase("select-columns", lambda: sp.SelectColumns(2)),
    ProcessorContractCase("tfidf", lambda: sp.TFIDF(ngram_range=(2, 2))),
    ProcessorContractCase("clip", lambda: sp.Clip(-2.0, 6.0)),
    ProcessorContractCase("clip-quantiles", sp.ClipQuantiles),
    ProcessorContractCase("clip-sigma", sp.ClipSigma),
    ProcessorContractCase(
        "impute-mean", sp.ImputeMean, _make_impute_mean_inputs
    ),
    ProcessorContractCase("power-transform", sp.PowerTransform),
    ProcessorContractCase(
        "quantile-transform",
        lambda: sp.QuantileTransform(n_quantiles=4, subsample=None),
    ),
    ProcessorContractCase("standardize", sp.Standardize),
    ProcessorContractCase("drop-constant-columns", sp.DropConstantColumns),
    ProcessorContractCase("pca", lambda: sp.PCA(2)),
    ProcessorContractCase("random-projection", lambda: sp.RandomProjection(2)),
    ProcessorContractCase(
        "align-categories",
        sp.AlignCategories,
        _make_align_categories_inputs,
    ),
    ProcessorContractCase("shuffle-categories", sp.ShuffleCategories),
    ProcessorContractCase("impute-mode", sp.ImputeMode),
    ProcessorContractCase(
        "add-calendar-fields", lambda: sp.AddCalendarFields(["month"])
    ),
    ProcessorContractCase("softmax", sp.Softmax),
    ProcessorContractCase(
        "reduce-estimators",
        sp.ReduceEstimators,
        _make_reduction_inputs,
    ),
    ProcessorContractCase(
        "ensemble-adapter",
        lambda: sp.EnsembleProcessorAdapter(sp.Standardize()),
    ),
    ProcessorContractCase(
        "sequential",
        lambda: sp.Sequential(
            sp.StypeDispatch(
                numerical=sp.Choice(
                    sp.Standardize(),
                    sp.ShuffleColumns(),
                    method="round_robin",
                )
            ),
            sp.StypeDispatch(numerical=sp.ShuffleColumns()),
        ),
    ),
    ProcessorContractCase(
        "stype-dispatch",
        lambda: sp.StypeDispatch(
            numerical=sp.Standardize(),
            categorical=sp.Identity(),
        ),
    ),
    ProcessorContractCase(
        "choice",
        lambda: sp.Choice(
            sp.Standardize(),
            sp.ShuffleColumns(),
            method="round_robin",
        ),
    ),
)
