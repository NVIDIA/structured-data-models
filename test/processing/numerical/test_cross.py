import torch

from sdm import Stype, TableTensor
from sdm.processing import CrossFeatures
from sdm.tensor import EnsembleTable
from sdm.testing import withCUDA


@withCUDA
def test_cross_features_replays_member_pairs(device: torch.device) -> None:
    context = TableTensor.from_tensor(
        torch.tensor([[2.0, 3.0, 5.0], [7.0, 11.0, 13.0]], device=device),
        columns=("a", "b", "c"),
    )
    query = TableTensor.from_tensor(
        torch.tensor([[17.0, 19.0, 23.0]], device=device),
        columns=("a", "b", "c"),
    )
    processor = CrossFeatures(num_crosses=1)
    generator = torch.Generator().manual_seed(4)

    context_out = processor.fit_transform_ensemble(
        EnsembleTable(context, num_members=2),
        generator=generator,
    )
    query_out = processor.transform_ensemble(
        EnsembleTable(query, num_members=2)
    )

    products = {6.0: 323.0, 10.0: 391.0, 15.0: 437.0}
    member_products = []
    for member_id in range(2):
        context_member = context_out.table(member_id)
        query_member = query_out.table(member_id)
        assert context_member.columns == query_member.columns
        assert context_member.columns[Stype.numerical][-1] == "cross_0"
        context_product = context_member.numerical[0, -1].item()
        query_product = query_member.numerical[0, -1].item()
        assert query_product == products[context_product]
        member_products.append(context_product)
    assert member_products[0] != member_products[1]


def test_cross_features_sqrt_selects_distinct_non_self_pairs() -> None:
    values = torch.tensor([[2.0, 3.0, 5.0, 7.0, 11.0, 13.0, 17.0, 19.0, 23.0]])
    table = TableTensor.from_tensor(values)
    processor = CrossFeatures(num_crosses="sqrt")

    output = processor.fit_transform(
        table,
        generator=torch.Generator().manual_seed(0),
    )

    crosses = output.numerical[0, -3:].tolist()
    pair_products = {
        values[0, left].item() * values[0, right].item()
        for left in range(9)
        for right in range(left + 1, 9)
    }
    assert len(set(crosses)) == 3
    assert set(crosses) <= pair_products


def test_cross_features_avoids_existing_generated_names() -> None:
    table = TableTensor.from_tensor(torch.tensor([[2.0, 3.0, 5.0]]))
    processor = CrossFeatures(1) + CrossFeatures(1)

    output = processor.fit_transform(
        table,
        generator=torch.Generator().manual_seed(0),
    )

    assert output.columns[Stype.numerical][-2:] == ("cross_0", "cross_1")
