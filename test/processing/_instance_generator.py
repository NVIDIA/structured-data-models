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

InputFactory = Callable[[], tuple[TableTensor, TableTensor]]


def make_mixed_inputs() -> tuple[TableTensor, TableTensor]:
    categories = (StringTensor.from_list(["a", "b"]),)
    numerical = torch.arange(12, dtype=torch.float32).reshape(4, 3)
    numerical[:, 1] = 1
    context = TableTensor(
        numerical=numerical,
        categorical=CategoricalTensor(
            code=torch.tensor(
                [[0], [0], [1], [-1]], dtype=torch.int32
            ),
            categories=categories,
        ),
        datetime=torch.arange(4, dtype=torch.int64)[:, None]
        * 86_400_000_000,
        text=StringTensor.from_list(
            [["alpha beta"], ["beta"], ["gamma"], ["alpha"]],
        ),
        id=ColumnarTensor((torch.arange(10, 14),)),
    )
    query = TableTensor(
        numerical=numerical + 2,
        categorical=CategoricalTensor(
            code=torch.tensor(
                [[-1], [1], [0], [-1]], dtype=torch.int32
            ),
            categories=categories,
        ),
        datetime=context.datetime + 60_000_000,
        text=StringTensor.from_list(
            [["alpha"], ["beta gamma"], ["unseen"], [""]],
        ),
        id=ColumnarTensor((torch.arange(20, 24),)),
    )
    return context, query


def _make_impute_mean_inputs() -> tuple[TableTensor, TableTensor]:
    context, query = make_mixed_inputs()
    numerical = query.numerical.clone()
    numerical[0, 0] = float("nan")
    return context, query.replace_blocks(numerical=numerical)


def _make_align_categories_inputs() -> tuple[TableTensor, TableTensor]:
    context, query = make_mixed_inputs()
    categorical = CategoricalTensor(
        code=torch.tensor(
            [[0], [1], [-1], [0]], dtype=torch.int32
        ),
        categories=(StringTensor.from_list(["b", "c"]),),
    )
    return context, query.replace_blocks(categorical=categorical)


def _make_reduction_inputs() -> tuple[TableTensor, TableTensor]:
    context = TableTensor.from_tensor(
        torch.tensor(
            [[0.0, 1.0], [2.0, 3.0], [4.0, 5.0], [6.0, 7.0]],
            dtype=torch.float32,
        )
    )
    query = TableTensor.from_tensor(context.numerical + 1.0)
    return context, query


@dataclass(frozen=True)
class ProcessorCase:
    processor: sp.Processor
    make_inputs: InputFactory = make_mixed_inputs


PROCESSOR_CASES = (
    ProcessorCase(sp.Identity()),
    ProcessorCase(sp.Callable(lambda table: table)),
    ProcessorCase(sp.DropStypes(Stype.id)),
    ProcessorCase(sp.ToNumerical()),
    ProcessorCase(sp.ShuffleColumns()),
    ProcessorCase(sp.SelectColumns(2)),
    ProcessorCase(sp.TFIDF(ngram_range=(2, 2))),
    ProcessorCase(sp.Clip(-2.0, 6.0)),
    ProcessorCase(sp.ClipQuantiles()),
    ProcessorCase(sp.ClipSigma()),
    ProcessorCase(sp.ImputeMean(), _make_impute_mean_inputs),
    ProcessorCase(sp.PowerTransform()),
    ProcessorCase(
        sp.QuantileTransform(n_quantiles=4, subsample=None),
    ),
    ProcessorCase(sp.Standardize()),
    ProcessorCase(sp.DropConstantColumns()),
    ProcessorCase(sp.PCA(2)),
    ProcessorCase(sp.RandomProjection(2)),
    ProcessorCase(sp.AlignCategories(), _make_align_categories_inputs),
    ProcessorCase(sp.ShuffleCategories()),
    ProcessorCase(sp.ImputeMode()),
    ProcessorCase(sp.AddCalendarFields(["month"])),
    ProcessorCase(sp.Softmax()),
    ProcessorCase(sp.ReduceEstimators(), _make_reduction_inputs),
    ProcessorCase(sp.EnsembleProcessorAdapter(sp.Standardize())),
    ProcessorCase(
        sp.Sequential(
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
    ProcessorCase(
        sp.StypeDispatch(
            numerical=sp.Standardize(),
            categorical=sp.Identity(),
        ),
    ),
    ProcessorCase(
        sp.Choice(
            sp.Standardize(),
            sp.ShuffleColumns(),
            method="round_robin",
        ),
    ),
)
