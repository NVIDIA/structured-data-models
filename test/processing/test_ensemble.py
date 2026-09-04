import pytest
import torch

from sdm import Recipe, StringTensor, Stype, TableTensor
from sdm.processing import (
    PCA,
    EnsembleInvertibleMixin,
    EnsembleProcessor,
    EnsembleProcessorAdapter,
    InvertibleMixin,
    Processor,
    Standardize,
)
from sdm.tensor import EnsembleTable


# TODO: Replace these stubs with real EnsembleProcessor subclasses once they
# land, and exercise the EnsembleProcessor contract through those instead.
# Generator forwarding is only observable through a stochastic transformation
# and is therefore left to those processors as well.
class IdentityEnsembleProcessor(EnsembleProcessor):
    handles_stypes = frozenset({Stype.numerical})
    requires_fit = True

    def _fit_ensemble(
        self,
        ensemble_table: EnsembleTable,
        *,
        generator: torch.Generator | None = None,
    ) -> None:
        pass

    def _transform_ensemble(
        self,
        ensemble_table: EnsembleTable,
    ) -> EnsembleTable:
        return ensemble_table


class FusedEnsembleProcessor(IdentityEnsembleProcessor):
    def __init__(self) -> None:
        super().__init__()
        self.used_fused_transform = False

    def _fit_transform_ensemble(
        self,
        ensemble_table: EnsembleTable,
        *,
        generator: torch.Generator | None = None,
    ) -> EnsembleTable:
        self.used_fused_transform = True
        return ensemble_table


class InvertibleIdentityEnsembleProcessor(
    IdentityEnsembleProcessor,
    EnsembleInvertibleMixin,
):
    def _inverse_transform_ensemble(
        self,
        ensemble_table: EnsembleTable,
    ) -> EnsembleTable:
        return EnsembleTable(ensemble_table.table(0), num_members=1)


class _StatelessProcessor(Processor, InvertibleMixin):
    handles_stypes = frozenset({Stype.numerical})
    requires_fit = False

    def _transform(self, table: TableTensor) -> TableTensor:
        return table.replace_blocks(numerical=-table.numerical)

    def _inverse_transform(self, table: TableTensor) -> TableTensor:
        return self._transform(table)


class _TextLength(Processor):
    handles_stypes = frozenset({Stype.text})
    requires_fit = False

    def _transform(self, table: TableTensor) -> TableTensor:
        text = table.text.reshape(-1).tolist()
        numerical = torch.tensor(
            [len(value or "") for value in text],
            dtype=torch.float32,
            device=table.device,
        ).reshape(table.text.shape)
        return TableTensor(
            columns={
                Stype.numerical: tuple(
                    f"{column}_length" for column in table.columns[Stype.text]
                )
            },
            numerical=numerical,
        )


def _text_table(lengths: list[tuple[int, int]]) -> TableTensor:
    return TableTensor.from_tensor(
        StringTensor.from_list(
            [["a" * left, "b" * right] for left, right in lengths]
        ),
        columns=("left", "right"),
    )


def test_ensemble_processor_preserves_member_order_and_metadata() -> None:
    first = TableTensor.from_tensor(
        torch.tensor([[1.0], [2.0]]),
        columns=("first",),
    )
    second = TableTensor.from_tensor(
        torch.tensor([[3.0], [4.0]]),
        columns=("second",),
    )
    ensemble_table = EnsembleTable.from_tables(
        tables=(first, second),
        member_table_ids=(1, 0, 1),
    )
    processor = IdentityEnsembleProcessor()

    output = processor.fit_transform_ensemble(ensemble_table)

    assert output is ensemble_table
    assert output.num_members == 3
    assert output.table(0).columns == second.columns
    assert output.table(1).columns == first.columns
    assert output.table(2).columns == second.columns
    assert processor.transform_ensemble(ensemble_table) is ensemble_table


def test_ensemble_processor_requires_fit_before_transform() -> None:
    table = TableTensor.from_tensor(torch.ones(2, 1))
    ensemble_table = EnsembleTable(table, num_members=2)
    processor = IdentityEnsembleProcessor()

    with pytest.raises(RuntimeError, match="not fitted"):
        processor.transform_ensemble(ensemble_table)

    assert processor.fit_ensemble(ensemble_table) is processor
    assert processor.transform_ensemble(ensemble_table) is ensemble_table


def test_ensemble_invertible_mixin_requires_fit_and_delegates() -> None:
    table = TableTensor.from_tensor(torch.ones(2, 1))
    ensemble_table = EnsembleTable(table, num_members=2)
    processor = InvertibleIdentityEnsembleProcessor()

    with pytest.raises(RuntimeError, match="not fitted"):
        processor.inverse_transform_ensemble(ensemble_table)

    processor.fit_transform_ensemble(ensemble_table)
    output = processor.inverse_transform_ensemble(ensemble_table)

    assert output.num_members == 1
    assert output.table(0).equal(table)
    assert processor.inverse_transform(table).equal(table)


def test_ensemble_processor_noops() -> None:
    table = TableTensor.from_tensor(torch.ones(2, 1, dtype=torch.int64))
    ensemble_table = EnsembleTable(table, num_members=2)
    processor = IdentityEnsembleProcessor()

    assert processor.fit_ensemble(ensemble_table) is processor
    assert processor.fit_transform_ensemble(ensemble_table) is ensemble_table
    assert processor.transform_ensemble(ensemble_table) is ensemble_table


def test_ensemble_processor_supports_table_tensor_lifecycle() -> None:
    table = TableTensor.from_tensor(torch.ones(2, 1))
    fitted = IdentityEnsembleProcessor()
    processor = IdentityEnsembleProcessor()

    assert fitted.fit(table) is fitted
    assert fitted.transform(table).equal(table)
    assert processor.fit_transform(table).equal(table)
    assert processor.transform(table).equal(table)
    assert processor(table).equal(table)


def test_ensemble_processor_uses_fused_table_tensor_lifecycle() -> None:
    table = TableTensor.from_tensor(torch.ones(2, 1))
    processor = FusedEnsembleProcessor()

    assert processor.fit_transform(table).equal(table)
    assert processor.used_fused_transform


def test_ensemble_processor_passthrough_for_empty_supported_blocks() -> None:
    empty_ensemble_table = EnsembleTable(
        TableTensor.from_tensor(torch.empty(2, 0)),
        num_members=2,
    )
    ensemble_table = EnsembleTable(
        TableTensor.from_tensor(torch.ones(2, 1)),
        num_members=2,
    )
    processor = IdentityEnsembleProcessor()

    assert processor.fit_ensemble(empty_ensemble_table) is processor
    assert (
        processor.fit_transform_ensemble(empty_ensemble_table)
        is empty_ensemble_table
    )
    assert (
        processor.transform_ensemble(empty_ensemble_table)
        is empty_ensemble_table
    )
    with pytest.raises(RuntimeError, match="not fitted"):
        processor.transform_ensemble(ensemble_table)


def _two_group_ensemble_table() -> EnsembleTable:
    first = TableTensor.from_tensor(
        torch.tensor([[1.0], [3.0]]), columns=("first",)
    )
    second = TableTensor.from_tensor(
        torch.tensor([[2.0], [6.0]]), columns=("second",)
    )
    return EnsembleTable.from_tables(
        tables=(first, second),
        member_table_ids=(1, 0, 1),
    )


def test_adapter_fits_each_member_separately() -> None:
    ensemble_table = _two_group_ensemble_table()
    processor = EnsembleProcessorAdapter(Standardize(with_std=False))

    processor.fit_ensemble(ensemble_table)
    output = processor.transform_ensemble(ensemble_table)

    assert output.num_members == 3
    assert output.table(0).numerical.tolist() == [[-2.0], [2.0]]
    assert output.table(1).numerical.tolist() == [[-1.0], [1.0]]
    assert output.table(2).equal(output.table(0))


def test_adapter_preserves_member_state_when_group_layout_changes() -> None:
    context = EnsembleTable.from_tables(
        tables=(
            TableTensor.from_tensor(torch.tensor([[1.0], [3.0]])),
            TableTensor.from_tensor(torch.tensor([[10.0], [20.0], [30.0]])),
        ),
        member_table_ids=(0, 1),
    )
    query = EnsembleTable(
        TableTensor.from_tensor(torch.tensor([[2.0], [4.0]])),
        num_members=2,
    )
    processor = EnsembleProcessorAdapter(Standardize(with_std=False))

    processor.fit_ensemble(context)
    output = processor.transform_ensemble(query)

    assert output.table(0).numerical.tolist() == [[0.0], [2.0]]
    assert output.table(1).numerical.tolist() == [[-18.0], [-16.0]]


def test_adapter_rejects_different_member_count_after_fit() -> None:
    table = TableTensor.from_tensor(torch.tensor([[1.0], [2.0]]))
    processor = EnsembleProcessorAdapter(Standardize())
    processor.fit_ensemble(EnsembleTable(table, num_members=2))

    with pytest.raises(RuntimeError, match=r"fitted with 2.*but got 3"):
        processor.transform_ensemble(EnsembleTable(table, num_members=3))


def test_adapter_state_dict_restores_member_state_order() -> None:
    context = EnsembleTable.from_tables(
        tables=tuple(
            TableTensor.from_tensor(torch.tensor([[value], [value + 2.0]]))
            for value in (0.0, 10.0, 20.0)
        ),
        member_table_ids=(0, 1, 2),
    )
    query = EnsembleTable(
        TableTensor.from_tensor(torch.tensor([[4.0]])),
        num_members=3,
    )
    processor = EnsembleProcessorAdapter(Standardize(with_std=False))
    processor.fit_ensemble(context)
    expected = processor.transform_ensemble(query)

    restored = EnsembleProcessorAdapter(Standardize(with_std=False))
    restored.fit_ensemble(EnsembleTable(context.table(0), num_members=1))
    restored.load_state_dict(processor.state_dict())
    output = restored.transform_ensemble(query)

    for member_id in range(query.num_members):
        assert output.table(member_id).equal(expected.table(member_id))


def test_recipe_encodes_shared_text_before_member_fitted_transforms() -> None:
    num_members = 8
    context_tables = tuple(
        _text_table(
            [
                (offset + 1, 1),
                (offset + 3, 2),
                (offset + 5, 1),
            ]
        )
        for offset in range(num_members)
    )
    context = EnsembleTable.from_tables(
        tables=context_tables,
        member_table_ids=range(num_members),
    )
    query_table = _text_table([(10, 3), (12, 2)])
    query = EnsembleTable(query_table, num_members=num_members)
    encoder = EnsembleProcessorAdapter(_TextLength())
    recipe = Recipe(features=(encoder, PCA(num_components=1)))

    recipe.features.fit_transform_ensemble(context)
    shared_query = encoder.transform_ensemble(query)
    output = recipe.features.transform_ensemble(query)

    shared_group = tuple(shared_query)
    assert len(shared_group) == 1
    assert shared_group[0].size(0) == 1
    assert shared_group[0].numerical.dtype == torch.float32
    output_group = tuple(output)
    assert len(output_group) == 1
    assert output_group[0].size(0) == num_members
    assert output_group[0].numerical.dtype == torch.float32

    reference_encoder = _TextLength()
    encoded_query = reference_encoder.transform(query_table)
    expected = tuple(
        PCA(num_components=1)
        .fit(reference_encoder.transform(table))
        .transform(encoded_query)
        for table in context_tables
    )
    for member_id, expected_table in enumerate(expected):
        torch.testing.assert_close(
            output.table(member_id).numerical,
            expected_table.numerical,
        )

    assert not output.table(0).equal(output.table(num_members - 1))


def test_adapter_fit_transform_matches_fit_then_transform() -> None:
    ensemble_table = _two_group_ensemble_table()
    processor = EnsembleProcessorAdapter(Standardize(with_std=False))
    combined = EnsembleProcessorAdapter(Standardize(with_std=False))

    processor.fit_ensemble(ensemble_table)
    output = processor.transform_ensemble(ensemble_table)
    expected = combined.fit_transform_ensemble(ensemble_table)

    for member_id in range(ensemble_table.num_members):
        assert output.table(member_id).equal(expected.table(member_id))


def test_adapter_inverse_restores_input() -> None:
    ensemble_table = _two_group_ensemble_table()
    processor = EnsembleProcessorAdapter(Standardize(with_std=False))

    output = processor.fit_transform_ensemble(ensemble_table)
    restored = processor.inverse_transform_ensemble(output)

    for member_id in range(ensemble_table.num_members):
        assert restored.table(member_id).equal(ensemble_table.table(member_id))


def test_stateless_adapter_supports_transform_and_inverse() -> None:
    table = TableTensor.from_tensor(torch.tensor([[1.0], [2.0]]))
    processor = EnsembleProcessorAdapter(_StatelessProcessor())

    transformed = processor.inverse_transform(table)

    assert transformed.numerical.tolist() == [[-1.0], [-2.0]]
    assert processor.transform(transformed).equal(table)
    assert processor.inverse_transform(transformed).equal(table)


def test_adapter_rejects_inverse_for_non_invertible_processor() -> None:
    table = TableTensor.from_tensor(torch.ones(2, 1))
    processor = EnsembleProcessorAdapter(
        Processor.as_processor(lambda value: value)
    )

    with pytest.raises(AttributeError, match="inverse_transform"):
        processor.inverse_transform(table)
