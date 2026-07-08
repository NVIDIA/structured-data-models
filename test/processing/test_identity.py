from sdm import TableTensor
from sdm.processing import Identity


def test_identity(table: TableTensor) -> None:
    processor = Identity()
    assert repr(processor) == "Identity()"

    assert processor.transform(table) is table
    assert processor.inverse_transform(table) is table
