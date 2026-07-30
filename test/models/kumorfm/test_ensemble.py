from typing import Any

import torch

from sdm import Stype
from sdm.models import ICLModel, KumoRFM
from sdm.models.kumorfm.model import _KumoRFM
from sdm.testing import withCUDA
from sdm.testing.datasets import canonical_rfm_data


def _small_model(device: torch.device) -> KumoRFM:
    model = KumoRFM.__new__(KumoRFM)
    ICLModel.__init__(model)
    common: dict[str, Any] = {
        "channels": 8,
        "num_embedding_layers": 1,
        "num_embedding_heads": 2,
        "num_inducing_points": 2,
        "group_size": 2,
        "num_readout_tokens": 2,
        "num_icl_layers": 1,
        "num_icl_heads": 2,
        "norm_bias": True,
        "device": device,
    }
    model.cls_model = _KumoRFM(
        num_classes=10,
        num_quantiles=0,
        **common,
    )
    model.reg_model = _KumoRFM(
        num_classes=0,
        num_quantiles=999,
        **common,
    )
    return model.eval()


def test_canonical_rfm_data_exercises_processor_edge_cases() -> None:
    data = canonical_rfm_data()

    assert tuple(data.related_context.tables) == (
        "customers",
        "orders",
        "products",
    )
    assert len(data.related_context.relationships) == 2
    assert len(data.related_context.task_links) == 1

    for table in data.related_context.tables.values():
        assert table.id.size(-1) > 0
        assert table.datetime.size(-1) > 0
        assert table.numerical.size(-1) > 0
        assert table.categorical.size(-1) > 0
        assert torch.isnan(table.numerical).any()

    customers = data.related_context.tables["customers"]
    orders = data.related_context.tables["orders"]
    products = data.related_context.tables["products"]
    assert customers[:, "constant_customer"].numerical.unique().numel() == 1
    assert orders[:, "constant_order"].numerical.unique().numel() == 1
    assert products[:, "constant_product"].numerical.unique().numel() == 1
    assert customers[:, "lifetime_spend"].numerical.max() == 1_000_000
    assert orders[:, "amount"].numerical.nan_to_num().max() == 5_000_000
    assert products[:, "price"].numerical.nan_to_num().max() == 250_000

    query_categories = {
        value
        for table in data.related_query.tables.values()
        for categories in table.categorical.categories
        for value in categories.tolist()
    }
    assert {
        "unseen-segment",
        "unseen-channel",
        "unseen-product-category",
    } <= query_categories
    assert data.classification_target.columns[Stype.categorical] == ("target",)
    assert data.regression_target.columns[Stype.numerical] == ("target",)


@withCUDA
def test_rfm_ensemble_matches_schedules_and_cached_execution(
    device: torch.device,
) -> None:
    torch.manual_seed(0)
    data = canonical_rfm_data(
        num_context_rows=6,
        num_query_rows=3,
        num_products=6,
        device=device,
    )
    model = _small_model(device)

    for task in ("classification", "regression"):
        arguments = {
            "x_context": data.x_context,
            "y_context": data.target(task),
            "x_query": data.x_query,
            "related_context_tables": data.related_context,
            "related_query_tables": data.related_query,
            "num_estimators": 2,
            "num_hops": 1,
        }
        parallel = model(
            **arguments,
            ensemble_mode="parallel",
            generator=torch.Generator(device=device).manual_seed(42),
        )
        sequential = model(
            **arguments,
            ensemble_mode="sequential",
            generator=torch.Generator(device=device).manual_seed(42),
        )
        assert parallel.columns == sequential.columns
        torch.testing.assert_close(
            parallel.numerical,
            sequential.numerical,
        )

        model.fit(
            data.x_context,
            data.target(task),
            data.related_context,
            num_estimators=2,
            ensemble_mode="parallel",
            generator=torch.Generator(device=device).manual_seed(42),
            num_hops=1,
        )
        cached = model.predict(data.x_query, data.related_query)
        assert parallel.columns == cached.columns
        torch.testing.assert_close(
            parallel.numerical,
            cached.numerical,
            atol=2e-5,
            rtol=2e-5,
        )
        model.clear()
