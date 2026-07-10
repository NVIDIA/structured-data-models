from collections.abc import Mapping
from dataclasses import replace
from typing import cast

import pytest
import torch
from sdm import (
    CategoricalTensor,
    ColumnarTensor,
    RelatedTables,
    Relationship,
    TableTensor,
)
from sdm.cache import Cache
from sdm.models import KumoRFM, Model
from sdm.models.kumorfm.model import _KumoRFM
from sdm.relational.sampler import EXAMPLE_ID
from test.models.kumorfm.sample_utils import with_full_sample


class _SmallKumoRFM(KumoRFM):
    def __init__(self) -> None:
        Model.__init__(self)
        self.cls_model = _core(num_classes=3, num_quantiles=0)
        self.reg_model = _core(num_classes=0, num_quantiles=5)
        self.eval()


def _core(num_classes: int, num_quantiles: int) -> _KumoRFM:
    return _KumoRFM(
        num_classes=num_classes,
        num_quantiles=num_quantiles,
        cell_channels=4,
        channels=8,
        num_embedding_layers=1,
        num_embedding_heads=1,
        num_inducing_points=2,
        group_size=1,
        num_readout_tokens=2,
        num_icl_layers=1,
        num_icl_heads=2,
    )


def _table(
    *,
    examples: list[int],
    ids: Mapping[str, list[int]],
    values: list[float],
) -> TableTensor:
    category_values = [int(value) % 4 for value in values]
    categories = tuple(dict.fromkeys(category_values))
    category_codes = {value: index for index, value in enumerate(categories)}
    return TableTensor(
        columns={
            "numerical": ("value",),
            "categorical": ("kind",),
            "id": (EXAMPLE_ID, *ids),
        },
        numerical=torch.tensor(values, dtype=torch.float).unsqueeze(-1),
        categorical=CategoricalTensor(
            data=torch.tensor(
                [category_codes[value] for value in category_values],
                dtype=torch.long,
            ).unsqueeze(-1),
            categories=(torch.tensor(categories, dtype=torch.long),),
        ),
        id=ColumnarTensor(
            (
                torch.tensor(examples, dtype=torch.long),
                *(
                    torch.tensor(ids[column], dtype=torch.long)
                    for column in ids
                ),
            )
        ),
    )


def _related_tables(
    entity_values: list[float],
    event_values: Mapping[str, list[float]] | None = None,
    event_examples: Mapping[str, list[int]] | None = None,
) -> RelatedTables:
    event_values = {} if event_values is None else event_values
    event_examples = {} if event_examples is None else event_examples
    num_entities = len(entity_values)
    tables = {
        "entity": _table(
            examples=list(range(num_entities)),
            ids={"entity_id": list(range(100, 100 + num_entities))},
            values=entity_values,
        )
    }
    relationships: list[Relationship] = []
    for table_index, (table_name, values) in enumerate(event_values.items()):
        num_events = len(values)
        examples = event_examples.get(table_name, list(range(num_events)))
        tables[table_name] = _table(
            examples=examples,
            ids={
                "event_id": list(
                    range(
                        1_000 * (table_index + 1),
                        1_000 * (table_index + 1) + num_events,
                    )
                ),
                "entity_id": [100 + example for example in examples],
            },
            values=values,
        )
        relationships.append(
            Relationship(
                left_table=table_name,
                left_columns=(EXAMPLE_ID, "entity_id"),
                right_table="entity",
                right_columns=(EXAMPLE_ID, "entity_id"),
            )
        )

    related_tables = RelatedTables(
        tables=tables,
        relationships=relationships,
        task_links=(
            {
                "task_columns": (EXAMPLE_ID, "entity_id"),
                "table": "entity",
                "table_columns": (EXAMPLE_ID, "entity_id"),
            },
        ),
    )
    if relationships:
        return with_full_sample(
            related_tables,
            num_task_rows=num_entities,
        )
    return related_tables


def _two_hop_tables(
    root_values: list[float],
    *,
    extra_entity_values: Mapping[int, float],
    item_values: Mapping[int, float] | None,
) -> RelatedTables:
    num_roots = len(root_values)
    root_examples = list(range(num_roots))
    extra_examples = list(extra_entity_values)
    tables = {
        "entity": _table(
            examples=root_examples + extra_examples,
            ids={
                "entity_id": [100 + index for index in root_examples]
                + [200 + index for index in extra_examples]
            },
            values=root_values + list(extra_entity_values.values()),
        ),
        "orders": _table(
            examples=root_examples,
            ids={
                "order_id": [300 + index for index in root_examples],
                "owner_id": [100 + index for index in root_examples],
                "referred_id": [200 + index for index in root_examples],
            },
            values=[10.0 + index for index in root_examples],
        ),
    }
    relationships = [
        Relationship(
            left_table="orders",
            left_columns=(EXAMPLE_ID, "owner_id"),
            right_table="entity",
            right_columns=(EXAMPLE_ID, "entity_id"),
        ),
        Relationship(
            left_table="orders",
            left_columns=(EXAMPLE_ID, "referred_id"),
            right_table="entity",
            right_columns=(EXAMPLE_ID, "entity_id"),
        ),
    ]
    if item_values is not None:
        item_examples = list(item_values)
        tables["items"] = _table(
            examples=item_examples,
            ids={
                "item_id": [400 + index for index in item_examples],
                "order_id": [300 + index for index in item_examples],
            },
            values=list(item_values.values()),
        )
        relationships.append(
            Relationship(
                left_table="items",
                left_columns=(EXAMPLE_ID, "order_id"),
                right_table="orders",
                right_columns=(EXAMPLE_ID, "order_id"),
            )
        )

    related_tables = RelatedTables(
        tables=tables,
        relationships=relationships,
        task_links=(
            {
                "task_columns": (EXAMPLE_ID, "entity_id"),
                "table": "entity",
                "table_columns": (EXAMPLE_ID, "entity_id"),
            },
        ),
    )
    return with_full_sample(
        related_tables,
        num_task_rows=num_roots,
    )


@pytest.mark.parametrize("with_related_table", [False, True])
@pytest.mark.parametrize("regression", [False, True])
def test_fit_predict_matches_joint_forward(
    with_related_table: bool,
    regression: bool,
) -> None:
    torch.manual_seed(0)
    model = _SmallKumoRFM()
    train_x = torch.tensor([[0.0, 1.0], [1.0, 3.0], [3.0, 2.0]])
    query_x = torch.tensor([[2.0, 4.0], [5.0, 1.0]])
    train_y = (
        torch.tensor([0.5, -1.0, 2.0])
        if regression
        else torch.tensor([0, 1, 2])
    )
    train_entity = [1.0, 4.0, 9.0]
    query_entity = [6.0, 12.0]
    train_events = {"events": [2.0, 5.0, 8.0]}
    query_events = {"events": [7.0, 11.0]}
    if not with_related_table:
        train_events = {}
        query_events = {}

    expected = model(
        torch.cat((train_x, query_x)),
        train_y,
        _related_tables(
            train_entity + query_entity,
            {
                name: train_events[name] + query_events[name]
                for name in train_events
            },
        ),
    )
    model.fit(
        train_x,
        train_y,
        _related_tables(train_entity, train_events),
    )
    actual = model.predict(
        query_x,
        _related_tables(query_entity, query_events),
    )
    torch.testing.assert_close(actual, expected, atol=1e-5, rtol=1e-5)
    if with_related_table and not regression:
        repeated = model.predict(
            query_x,
            _related_tables(query_entity, query_events),
        )
        torch.testing.assert_close(repeated, actual)


def test_omitted_relation_keeps_later_relation_stable() -> None:
    torch.manual_seed(0)
    model = _SmallKumoRFM()
    train_x = torch.tensor([[0.0], [1.0], [2.0]])
    query_x = torch.tensor([[3.0], [4.0]])
    train_y = torch.tensor([0, 1, 2])
    train_events = {
        "first_events": [1.0, 2.0, 3.0],
        "second_events": [4.0, 5.0, 6.0],
        "query_only": [],
    }
    query_events = {
        "second_events": [7.0, 8.0],
        "query_only": [9.0, 10.0],
    }
    joint_events = {
        "first_events": [1.0, 2.0, 3.0],
        "second_events": [4.0, 5.0, 6.0, 7.0, 8.0],
        "query_only": [9.0, 10.0],
    }

    expected = model(
        torch.cat((train_x, query_x)),
        train_y,
        _related_tables(
            [1.0, 2.0, 3.0, 4.0, 5.0],
            joint_events,
            event_examples={"query_only": [3, 4]},
        ),
    )
    model.fit(
        train_x,
        train_y,
        _related_tables([1.0, 2.0, 3.0], train_events),
    )

    actual = model.predict(
        query_x,
        _related_tables([4.0, 5.0], query_events),
    )

    torch.testing.assert_close(actual, expected, atol=1e-5, rtol=1e-5)


def test_query_only_hop_uses_fallback_without_kv_cache() -> None:
    torch.manual_seed(0)
    model = _SmallKumoRFM()
    handle = model.cls_model.table_hop_encoder.register_forward_pre_hook(
        lambda _module, args, kwargs: (
            args,
            {**kwargs, "max_keys": 1},
        ),
        with_kwargs=True,
    )
    train_x = torch.tensor([[0.0], [1.0], [2.0]])
    query_x = torch.tensor([[3.0], [4.0]])
    train_y = torch.tensor([0, 1, 2])
    train_tables = _two_hop_tables(
        [1.0, 2.0, 3.0],
        extra_entity_values={},
        item_values={0: 4.0, 1: 5.0, 2: 6.0},
    )
    query_tables = _two_hop_tables(
        [7.0, 8.0],
        extra_entity_values={0: 9.0, 1: 10.0},
        item_values=None,
    )
    joint_tables = _two_hop_tables(
        [1.0, 2.0, 3.0, 7.0, 8.0],
        extra_entity_values={3: 9.0, 4: 10.0},
        item_values={0: 4.0, 1: 5.0, 2: 6.0},
    )

    expected = model(
        torch.cat((train_x, query_x)),
        train_y,
        joint_tables,
    )
    model.fit(train_x, train_y, train_tables)

    assert model._caches is not None
    table_hop_cache = cast(Cache, model._caches[0]["table_hop_encoder"])
    tables_cache = cast(Cache, table_hop_cache["tables"])
    entity_cache = cast(Cache, tables_cache["entity"])
    query_only_hop = cast(Cache, entity_cache["hop2"])
    assert query_only_hop["has_context"] is False
    assert "row_embedding" not in query_only_hop

    actual = model.predict(query_x, query_tables)
    handle.remove()

    torch.testing.assert_close(actual, expected, atol=1e-5, rtol=1e-5)
    assert "row_embedding" not in query_only_hop


def test_cache_rejects_incompatible_query_schema() -> None:
    torch.manual_seed(0)
    model = _SmallKumoRFM()
    train_x = torch.tensor([[0.0], [1.0], [2.0]])
    train_y = torch.tensor([0, 1, 2])
    model.fit(
        train_x,
        train_y,
        _related_tables(
            [1.0, 2.0, 3.0],
            {"events": [4.0, 5.0, 6.0]},
        ),
    )
    query_tables = _related_tables(
        [7.0, 8.0],
        {"events": [9.0, 10.0]},
    )

    entity = query_tables.tables["entity"]
    renamed_entity = TableTensor(
        columns={
            "numerical": ("renamed_value",),
            "id": (EXAMPLE_ID, "entity_id"),
        },
        numerical=entity.numerical,
        id=entity.id,
    )
    with pytest.raises(ValueError, match=r"table 'entity'.*fitted schema"):
        model.predict(
            torch.tensor([[3.0], [4.0]]),
            RelatedTables(
                tables={**query_tables.tables, "entity": renamed_entity},
                relationships=query_tables.relationships,
                task_links=query_tables.task_links,
                sample=query_tables.sample,
            ),
        )

    incompatible_relationship = Relationship(
        left_table="events",
        left_columns=(EXAMPLE_ID,),
        right_table="entity",
        right_columns=(EXAMPLE_ID,),
    )
    with pytest.raises(ValueError, match=r"relationship.*incompatible"):
        model.predict(
            torch.tensor([[3.0], [4.0]]),
            RelatedTables(
                tables=query_tables.tables,
                relationships=(incompatible_relationship,),
                task_links=query_tables.task_links,
                sample=query_tables.sample,
            ),
        )

    with pytest.raises(ValueError, match="source feature width"):
        model.predict(torch.ones(2, 2), query_tables)


def test_cache_rejects_model_dtype_change() -> None:
    model = _SmallKumoRFM()
    model.fit(
        torch.tensor([[0.0], [1.0]]),
        torch.tensor([0, 1]),
        _related_tables([1.0, 2.0]),
    )
    model.double()

    with pytest.raises(ValueError, match="model contract"):
        model.predict(
            torch.tensor([[2.0]], dtype=torch.double),
            _related_tables([3.0]),
        )


def test_cache_rejects_changed_sampling_policy() -> None:
    model = _SmallKumoRFM()
    model.fit(
        torch.tensor([[0.0], [1.0]]),
        torch.tensor([0, 1]),
        _related_tables([1.0, 2.0], {"events": [3.0, 4.0]}),
    )
    query_tables = _related_tables([5.0], {"events": [6.0]})
    assert query_tables.sample is not None
    query_tables = RelatedTables(
        tables=query_tables.tables,
        relationships=query_tables.relationships,
        task_links=query_tables.task_links,
        sample=replace(query_tables.sample, temporal=True),
    )

    with pytest.raises(ValueError, match="sampling policy"):
        model.predict(torch.tensor([[2.0]]), query_tables)
