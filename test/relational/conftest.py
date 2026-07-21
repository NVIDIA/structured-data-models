import pandas as pd
import pytest
from sdm import RelationalData, Stype, TableTensor


@pytest.fixture
def temporal_data() -> RelationalData:
    return RelationalData(
        tables={
            "roots": TableTensor.from_pandas(
                df=pd.DataFrame({"root_id": [0]}),
                stypes={"root_id": Stype.id},
            ),
            "first": TableTensor.from_pandas(
                df=pd.DataFrame(
                    {
                        "first_id": [10, 11, 12],
                        "root_id": [0, 0, 0],
                        "time": pd.to_datetime([3, 1, 2], unit="s"),
                    }
                ),
                stypes={
                    "first_id": Stype.id,
                    "root_id": Stype.id,
                    "time": Stype.datetime,
                },
            ),
            "second": TableTensor.from_pandas(
                df=pd.DataFrame(
                    {
                        "second_id": [20, 21],
                        "first_id": [10, 12],
                        "time": pd.to_datetime([8, 9], unit="s"),
                    }
                ),
                stypes={
                    "second_id": Stype.id,
                    "first_id": Stype.id,
                    "time": Stype.datetime,
                },
            ),
        },
        relationships=[
            {
                "left_table": "first",
                "left_column": "root_id",
                "right_table": "roots",
                "right_column": "root_id",
            },
            {
                "left_table": "second",
                "left_column": "first_id",
                "right_table": "first",
                "right_column": "first_id",
            },
        ],
    )
