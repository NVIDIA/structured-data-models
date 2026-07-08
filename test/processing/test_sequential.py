from sdm import TableTensor
from sdm.processing import Clip, Sequential, StandardScale


def test_empty(table: TableTensor) -> None:
    assert repr(Sequential()) == "Sequential()"
    Sequential().fit(table)
    assert Sequential().transform(table) is table
    assert Sequential().fit_transform(table) is table
    assert Sequential().inverse_transform(table) is table


def test_sequential(table: TableTensor) -> None:
    processor = Sequential(StandardScale(), Clip())
    assert repr(processor) == "Sequential(\n  StandardScale(),\n  Clip(),\n)"

    processor.fit(table.select_stypes("numerical"))
    out = processor.transform(table.select_stypes("numerical"))
    assert not out.numerical.allclose(table.numerical)

    out = processor.fit_transform(table.select_stypes("numerical"))
    assert not out.numerical.allclose(table.numerical)

    out = processor.inverse_transform(out)
    assert out.numerical.allclose(table.numerical)
