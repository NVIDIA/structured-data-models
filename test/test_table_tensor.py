import torch

from schemafm import Stype, TableTensor


def make_table() -> TableTensor:
    data = torch.arange(12, dtype=torch.float32).reshape(4, 3)
    mask = torch.ones_like(data, dtype=torch.bool)
    return TableTensor(
        data,
        names=('age', 'income', 'fraud'),
        stypes=(Stype.numerical, Stype.numerical, Stype.categorical),
        mask=mask,
    )


def test_table_tensor_validates_names_and_stypes():
    table = TableTensor(
        torch.randn(2, 2),
        names=('age', 'fraud'),
        stypes=('numerical', 'categorical'),
    )

    assert table.column_index('fraud') == 1
    assert table.indices_for(Stype.numerical) == (0,)
    assert table.select(('fraud',)).column_names == ('fraud',)


def test_string_column_indexing_returns_one_dimensional_table_tensor():
    table = make_table()

    fraud = table[:, 'fraud']

    assert isinstance(fraud, TableTensor)
    assert fraud.shape == (4,)
    assert fraud.column_names == ('fraud',)
    assert fraud.stypes == (Stype.categorical,)
    assert torch.equal(fraud.values, table.values[:, 2])


def test_list_column_indexing_returns_two_dimensional_table_tensor():
    table = make_table()

    selected = table[:, ['income', 'fraud']]

    assert isinstance(selected, TableTensor)
    assert selected.shape == (4, 2)
    assert selected.column_names == ('income', 'fraud')
    assert torch.equal(selected.values, table.values[:, [1, 2]])


def test_row_slicing_preserves_table_tensor_metadata():
    table = make_table()

    sliced = table[:2]

    assert isinstance(sliced, TableTensor)
    assert sliced.shape == (2, 3)
    assert sliced.column_names == table.column_names
    assert sliced.stypes == table.stypes
    assert sliced.mask is not None
    assert sliced.mask.shape == (2, 3)


def test_row_extraction_returns_plain_tensor():
    table = make_table()

    row = table[0]

    assert isinstance(row, torch.Tensor)
    assert not isinstance(row, TableTensor)
    assert row.shape == (3,)


def test_select_and_drop_update_metadata():
    table = make_table()

    selected = table.select(('age', 'fraud'))
    dropped = table.drop('income')

    assert selected.column_names == ('age', 'fraud')
    assert dropped.column_names == ('age', 'fraud')
    assert torch.equal(selected.values, dropped.values)


def test_clone_and_to_preserve_metadata():
    table = make_table()

    cloned = table.clone()
    converted = table.to(dtype=torch.float64)

    assert isinstance(cloned, TableTensor)
    assert cloned.column_names == table.column_names
    assert cloned.stypes == table.stypes
    assert cloned.mask is not table.mask

    assert isinstance(converted, TableTensor)
    assert converted.dtype == torch.float64
    assert converted.column_names == table.column_names
    assert converted.stypes == table.stypes
    assert converted.mask is not None
    assert converted.mask.dtype == torch.bool


def test_row_cat_preserves_metadata_when_metadata_matches():
    table = make_table()

    out = torch.cat([table[:2], table[2:]], dim=0)

    assert isinstance(out, TableTensor)
    assert out.column_names == table.column_names
    assert out.stypes == table.stypes
    assert torch.equal(out.values, table.values)


def test_general_math_returns_plain_tensor():
    table = make_table()

    out = table + 1

    assert isinstance(out, torch.Tensor)
    assert not isinstance(out, TableTensor)
    assert torch.equal(out, table.values + 1)
