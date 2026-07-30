from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, cast

import numpy as np
import pandas as pd  # noqa: TID253
import torch

from sdm import RelatedTables, Stype, TableTensor

Task = Literal["classification", "regression"]


@dataclass(frozen=True)
class CanonicalRFMData:
    """Canonical context/query graph used by RFM parity and benchmarks."""

    x_context: TableTensor
    classification_target: TableTensor
    regression_target: TableTensor
    x_query: TableTensor
    related_context: RelatedTables
    related_query: RelatedTables

    def target(self, task: Task) -> TableTensor:
        """Return the context target for ``task``."""
        if task == "classification":
            return self.classification_target
        return self.regression_target


def _customer_frame(
    ids: np.ndarray,
    *,
    query: bool,
) -> pd.DataFrame:

    count = len(ids)
    age = 18.0 + np.remainder(ids, 55).astype(np.float64)
    age[::7] = np.nan
    spend = np.log1p(ids).astype(np.float64) * 10
    if count > 0:
        spend[-1] = 1_000_000.0
    segments = np.asarray(["consumer", "business", "education"], dtype=object)[
        np.remainder(ids, 3)
    ]
    if query and count > 0:
        segments[-1] = "unseen-segment"
    segments[::11] = None
    return pd.DataFrame(
        {
            "customer_id": ids,
            "age": age,
            "lifetime_spend": spend,
            "segment": segments,
            "signup_at": pd.Timestamp("2020-01-01")
            + pd.to_timedelta(ids, unit="D"),
            "constant_customer": np.ones(count),
        }
    )


def _order_frame(
    customer_ids: np.ndarray,
    *,
    num_products: int,
    query: bool,
) -> pd.DataFrame:

    order_customer = np.repeat(customer_ids, 2)
    order_ids = order_customer * 2 + np.tile([0, 1], len(customer_ids))
    amount = (np.remainder(order_ids, 97) + 1).astype(np.float64)
    amount[::13] = np.nan
    if len(amount) > 0:
        amount[0] = 5_000_000.0
    channels = np.asarray(["web", "store", "partner"], dtype=object)[
        np.remainder(order_ids, 3)
    ]
    if query and len(channels) > 0:
        channels[-1] = "unseen-channel"
    channels[::17] = None
    return pd.DataFrame(
        {
            "order_id": order_ids,
            "customer_id": order_customer,
            "product_id": 10_000 + np.remainder(order_ids, num_products),
            "amount": amount,
            "channel": channels,
            "ordered_at": pd.Timestamp("2022-01-01")
            + pd.to_timedelta(order_ids, unit="h"),
            "constant_order": np.full(len(order_ids), 7.0),
        }
    )


def _product_frame(num_products: int, *, query: bool) -> pd.DataFrame:

    ids = np.arange(10_000, 10_000 + num_products)
    categories = np.asarray(["hardware", "software", "service"], dtype=object)[
        np.remainder(ids, 3)
    ]
    if query and num_products > 0:
        categories[-1] = "unseen-product-category"
    prices = (np.remainder(ids, 31) + 1).astype(np.float64)
    prices[::9] = np.nan
    if num_products > 0:
        prices[-1] = 250_000.0
    return pd.DataFrame(
        {
            "product_id": ids,
            "category": categories,
            "price": prices,
            "released_at": pd.Timestamp("2019-01-01")
            + pd.to_timedelta(np.arange(num_products), unit="W"),
            "constant_product": np.zeros(num_products),
        }
    )


def canonical_rfm_data(
    *,
    num_context_rows: int = 8,
    num_query_rows: int = 4,
    num_products: int = 12,
    device: torch.device | str = "cpu",
) -> CanonicalRFMData:
    """Build the canonical processor-covering relational dataset.

    Dataset construction is intentionally outside benchmark timing. The graph
    includes primary/foreign IDs, datetimes, numerical and categorical data,
    missing values, constants, outliers, and query-only categories.

    Args:
        num_context_rows: Number of context task/customer rows.
        num_query_rows: Number of query task/customer rows.
        num_products: Number of product rows in each graph.
        device: Destination device for all tables.
    """
    if num_context_rows < 1 or num_query_rows < 1 or num_products < 1:
        raise ValueError("Canonical RFM row counts must be positive.")
    context_ids = np.arange(num_context_rows)
    query_ids = np.arange(
        num_context_rows,
        num_context_rows + num_query_rows,
    )

    def table(df: pd.DataFrame, stypes: dict[str, Stype]) -> TableTensor:
        return cast(
            TableTensor,
            TableTensor.from_pandas(df=df, stypes=stypes).to(device),
        )

    customer_stypes = {
        "customer_id": Stype.id,
        "age": Stype.numerical,
        "lifetime_spend": Stype.numerical,
        "segment": Stype.categorical,
        "signup_at": Stype.datetime,
        "constant_customer": Stype.numerical,
    }
    order_stypes = {
        "order_id": Stype.id,
        "customer_id": Stype.id,
        "product_id": Stype.id,
        "amount": Stype.numerical,
        "channel": Stype.categorical,
        "ordered_at": Stype.datetime,
        "constant_order": Stype.numerical,
    }
    product_stypes = {
        "product_id": Stype.id,
        "category": Stype.categorical,
        "price": Stype.numerical,
        "released_at": Stype.datetime,
        "constant_product": Stype.numerical,
    }

    relationships = (
        {
            "left_table": "orders",
            "left_column": "customer_id",
            "right_table": "customers",
            "right_column": "customer_id",
        },
        {
            "left_table": "orders",
            "left_column": "product_id",
            "right_table": "products",
            "right_column": "product_id",
        },
    )
    task_links = (
        {
            "task_column": "customer_id",
            "table": "customers",
            "table_column": "customer_id",
        },
    )

    def related(ids: np.ndarray, *, query: bool) -> RelatedTables:
        return RelatedTables(
            tables={
                "customers": table(
                    _customer_frame(ids, query=query),
                    customer_stypes,
                ),
                "orders": table(
                    _order_frame(
                        ids,
                        num_products=num_products,
                        query=query,
                    ),
                    order_stypes,
                ),
                "products": table(
                    _product_frame(num_products, query=query),
                    product_stypes,
                ),
            },
            relationships=relationships,
            task_links=task_links,
        )

    def task_table(ids: np.ndarray, *, query: bool) -> TableTensor:
        frame = pd.DataFrame(
            {
                "customer_id": ids,
                "task_score": np.sqrt(ids + 1),
                "task_group": np.where(
                    np.remainder(ids, 2) == 0,
                    "even",
                    "odd",
                ),
                "task_time": pd.Timestamp("2024-01-01")
                + pd.to_timedelta(ids, unit="D"),
            }
        )
        if query and len(frame) > 0:
            frame.loc[frame.index[-1], "task_group"] = "unseen-task-group"
        return table(
            frame,
            {
                "customer_id": Stype.id,
                "task_score": Stype.numerical,
                "task_group": Stype.categorical,
                "task_time": Stype.datetime,
            },
        )

    classification_target = table(
        pd.DataFrame(
            {
                "target": np.asarray(["low", "mid", "high"], dtype=object)[
                    np.remainder(context_ids, 3)
                ]
            }
        ),
        {"target": Stype.categorical},
    )
    regression_target = table(
        pd.DataFrame(
            {
                "target": np.log1p(context_ids).astype(np.float32)
                + np.remainder(context_ids, 5)
            }
        ),
        {"target": Stype.numerical},
    )
    return CanonicalRFMData(
        x_context=task_table(context_ids, query=False),
        classification_target=classification_target,
        regression_target=regression_target,
        x_query=task_table(query_ids, query=True),
        related_context=related(context_ids, query=False),
        related_query=related(query_ids, query=True),
    )
