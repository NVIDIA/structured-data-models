import torch

from sdm import CategoricalTensor, StringTensor, Stype, TableTensor
from sdm.models.tabfm._scheduling import _ShiftClasses, _ShuffleFeatures
from sdm.tensor import EnsembleTable
from sdm.testing import withCUDA


def _target(device: torch.device) -> TableTensor:
    return TableTensor(
        columns={Stype.categorical: ("target",)},
        categorical=CategoricalTensor(
            code=torch.tensor(
                [[0], [1], [2], [-1], [0], [1]],
                device=device,
            ),
            categories=(
                StringTensor.from_list(
                    ["gamma", "alpha", "beta"],
                    device=device,
                ),
            ),
        ),
    )


@withCUDA
def test_vectorized_schedules_are_balanced_and_preserve_labels(
    device: torch.device,
) -> None:
    features = TableTensor.from_tensor(
        torch.arange(12.0, device=device).view(6, 2),
        columns=("x0", "x1"),
    )
    target = _target(device)
    generator = torch.Generator(device=device).manual_seed(0)
    shuffled = _ShuffleFeatures().fit_transform_ensemble(
        EnsembleTable(features, num_members=8),
        generator=generator,
    )
    shifted = _ShiftClasses().fit_transform_ensemble(
        EnsembleTable(target, num_members=8),
        generator=generator,
    )

    feature_orders = [
        shuffled.table(index).columns[Stype.numerical] for index in range(8)
    ]
    class_orders = [
        tuple(shifted.table(index).categorical.categories[0].tolist())
        for index in range(8)
    ]
    feature_order_set = {("x0", "x1"), ("x1", "x0")}
    class_order_set = {
        ("alpha", "beta", "gamma"),
        ("beta", "gamma", "alpha"),
        ("gamma", "alpha", "beta"),
    }
    assert {
        order: feature_orders.count(order) for order in feature_order_set
    } == dict.fromkeys(feature_order_set, 4)
    assert set(class_orders) == class_order_set
    assert sorted(class_orders.count(order) for order in class_order_set) == [
        2,
        3,
        3,
    ]
    assert class_orders[0] == ("gamma", "alpha", "beta")

    expected_labels = ["gamma", "alpha", "beta", None, "gamma", "alpha"]
    for index in range(8):
        categorical = shifted.table(index).categorical
        category = categorical.categories[0].tolist()
        decoded = [
            None if code < 0 else category[code]
            for code in categorical.code[:, 0].tolist()
        ]
        assert decoded == expected_labels

    empty = TableTensor.from_tensor(
        features.numerical[:, :0],
        columns=(),
    )
    empty_output = _ShuffleFeatures().fit_transform_ensemble(
        EnsembleTable(empty, num_members=2),
        generator=generator,
    )
    assert all(empty_output.table(index).size(-1) == 0 for index in range(2))


def test_class_shift_remainder_is_randomized() -> None:
    target = _target(torch.device("cpu"))
    extra_orders = set()
    for seed in range(12):
        shifted = _ShiftClasses().fit_transform_ensemble(
            EnsembleTable(target, num_members=4),
            generator=torch.Generator().manual_seed(seed),
        )
        orders = [
            tuple(shifted.table(index).categorical.categories[0].tolist())
            for index in range(4)
        ]
        counts = {order: orders.count(order) for order in set(orders)}
        assert sorted(counts.values()) == [1, 1, 2]
        extra_orders.add(
            next(order for order, count in counts.items() if count == 2)
        )
    assert len(extra_orders) == 3


def test_wide_feature_fit_replays_member_selection() -> None:
    columns = tuple(f"x{index}" for index in range(501))
    context = TableTensor.from_tensor(
        torch.arange(2004.0).view(4, 501),
        columns=columns,
    )
    query = TableTensor.from_tensor(
        torch.arange(501.0).view(1, 501),
        columns=columns,
    )
    processor = _ShuffleFeatures()
    transformed = processor.fit_transform_ensemble(
        EnsembleTable(context, num_members=2),
        generator=torch.Generator().manual_seed(0),
    )
    transformed_query = processor.transform_ensemble(
        EnsembleTable(query, num_members=2)
    )

    orders = []
    for index in range(2):
        output = transformed.table(index)
        query_output = transformed_query.table(index)
        order = output.columns[Stype.numerical]
        indices = torch.tensor([columns.index(column) for column in order])
        assert len(order) == 500
        assert len(set(order)) == 500
        assert query_output.columns[Stype.numerical] == order
        torch.testing.assert_close(
            query_output.numerical,
            query.numerical.index_select(-1, indices),
        )
        orders.append(order)
    assert orders[0] != orders[1]


@withCUDA
def test_wide_feature_sampling_has_bounded_owned_state(
    device: torch.device,
) -> None:
    def sample() -> torch.Tensor:
        return _ShuffleFeatures._sample_ordered(
            10_000_000,
            device=device,
            generator=torch.Generator(device=device).manual_seed(0),
        )

    first = sample()
    second = sample()
    assert torch.equal(first, second)
    assert first.dtype == torch.long
    assert first.device == device
    assert first.numel() == 500
    assert first.unique().numel() == first.numel()
    assert first.ge(0).all()
    assert first.lt(10_000_000).all()
    assert (
        first.untyped_storage().nbytes()
        == first.numel() * first.element_size()
    )
