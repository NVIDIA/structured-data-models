import pytest
import torch

import sdm.processing as sp
from sdm import CategoricalTensor, StringTensor, Stype, TableTensor
from sdm.tensor import EnsembleTable
from sdm.testing import withCUDA


def test_truncated_svd_replays_fitted_preprocessor_for_query() -> None:
    categories = (StringTensor.from_list(["a", "b"]),)
    context = TableTensor(
        numerical=torch.tensor([[10.0], [12.0]]),
        categorical=CategoricalTensor(
            code=torch.tensor([[0], [1]]), categories=categories
        ),
    )
    query = TableTensor(
        numerical=torch.tensor([[10.0], [14.0]]),
        categorical=CategoricalTensor(
            code=torch.tensor([[0], [-1]]), categories=categories
        ),
    )
    processor = sp.TruncatedSVD(
        num_components=1,
        preprocessor=sp.StypeDispatch(
            numerical=sp.Standardize(),
            categorical=sp.OneHot(),
        ),
    )
    fitted = processor.fit_transform(
        context,
        generator=torch.Generator().manual_seed(0),
    )

    output = processor.transform(query)

    assert output.columns[Stype.numerical][-1] == "svd_0"
    assert output.numerical[:, 0].equal(query.numerical[:, 0])
    assert output.categorical.equal(query.categorical)
    assert output.numerical[0, -1] == fitted.numerical[0, -1]
    assert output.numerical[1, -1].isfinite()


@withCUDA
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_truncated_svd_sqrt_adds_positive_component_count(
    device: torch.device,
    dtype: torch.dtype,
) -> None:
    table = TableTensor.from_tensor(torch.eye(9, device=device, dtype=dtype))

    output = sp.TruncatedSVD(num_components="sqrt").fit_transform(
        table,
        generator=torch.Generator().manual_seed(0),
    )

    assert output.numerical.size(-1) == 12
    assert output.numerical.dtype == dtype
    assert output.numerical[:, -3:].isfinite().all()
    assert output.columns[Stype.numerical][-3:] == ("svd_0", "svd_1", "svd_2")


def test_truncated_svd_shared_pool_selections_replay() -> None:
    context = TableTensor.from_tensor(
        torch.diag(torch.tensor([4.0, 2.0, 1.0]))
    )
    query = TableTensor.from_tensor(torch.tensor([[3.0, 5.0, 7.0]]))
    processor = sp.TruncatedSVD(num_components=1, pool_size="sum")
    processor.fit_transform_ensemble(
        EnsembleTable(context, num_members=2),
        generator=torch.Generator().manual_seed(0),
    )

    first = processor.transform_ensemble(EnsembleTable(query, num_members=2))
    second = processor.transform_ensemble(EnsembleTable(query, num_members=2))

    for member_id in range(2):
        assert first.table(member_id).numerical.size(-1) == 4
        assert first.table(member_id).equal(second.table(member_id))


def test_truncated_svd_avoids_existing_generated_names() -> None:
    table = TableTensor.from_tensor(torch.eye(5))
    processor = sp.TruncatedSVD(1) + sp.TruncatedSVD(1)

    output = processor.fit_transform(
        table,
        generator=torch.Generator().manual_seed(0),
    )

    assert output.columns[Stype.numerical][-2:] == ("svd_0", "svd_1")
