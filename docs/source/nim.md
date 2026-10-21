# NIM Inference

NVIDIA NIM runs Kumo Tabular and Kumo Relational as inference services on a GPU host.
This guide shows how to call a deployed NIM from Python and use data connectors to read input tables from a database or warehouse.
Your Python process prepares the data and receives predictions; it does not need model weights or a local GPU.

Start with installation and a NIM connection, then follow the example for your model.
The later sections cover data connectors, Databricks Model Serving, and reuse of context across prediction calls.
For models that run directly in your Python process, see [In-Context Learning](icl).

## Installation

Install the lightweight client package from PyPI.
It runs independently of the PyTorch library described in [Installation](install); that library is not needed to call a NIM:

```bash
python -m pip install "structured-data-models-client"
```

For Kumo Relational, include the `relational` extra:

```bash
python -m pip install "structured-data-models-client[relational]"
```

Use Python 3.10 through 3.14 for Tabular inference.
The relational-engine wheels are built for CPython 3.10 through 3.13 on Linux with glibc 2.28 or newer, on x86-64 and aarch64.
Tabular inference can be called from Linux, macOS, or Windows; the relational-engine wheels require Linux.
The GPU is needed on the NIM host.

(nim-databricks-notebooks)=

### Databricks notebooks

Use `%pip` to install into a Databricks notebook while preserving the runtime's NumPy version.
Run each of these steps in its own cell, before running the prediction examples.
The extras below cover both warehouse access and Databricks Model Serving.

First, capture the installed NumPy version:

```python
import numpy

keep_numpy = f"numpy=={numpy.__version__}"
```

Then install the Python package and platform dependencies:

```text
%pip install --only-binary=:all: "structured-data-models-client[relational,databricks,databricks-serving]" $keep_numpy
```

Finally, restart Python so the notebook loads the installed packages:

```python
dbutils.library.restartPython()
```

## Connect to a NIM

Choose a model and follow its deployment instructions on a Linux GPU host with Docker and the NVIDIA Container Toolkit:

| Model ID          | Data                                         | Deployment instructions                                                                                                                                                          |
| ----------------- | -------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `kumo-tabular`    | One table; classification and regression     | [Kumo Tabular on NVIDIA Build](https://build.nvidia.com/nvidia/kumo-tabular)                                                                                                     |
| `kumo-relational` | Related tables and temporal prediction tasks | [Kumo Relational tags on NGC](https://catalog.ngc.nvidia.com/orgs/nim/nvidia/containers/kumo-relational/-/tags), [NVIDIA Build](https://build.nvidia.com/nvidia/kumo-relational) |

Check the deployment pages for available tags, supported hardware, access requirements, and the terms for the selected artifact.
For example, authenticate to the registry with an NGC API key as the password:

```bash
docker login nvcr.io --username '$oauthtoken'
```

Then run the Relational NIM in one terminal, mapping host port 8001 to the container's port 8000:

```bash
docker run --rm --gpus all --shm-size=16GB -p 8001:8000 \
    nvcr.io/nim/nvidia/kumo-relational:1.0.1
```

For the Tabular example, follow its deployment page and expose the service on host port 8000.
The Python examples use `http://localhost:8000` for Tabular and `http://localhost:8001` for Relational.
When the service runs on another machine, replace these addresses with the deployment's reachable base URLs.

Wait for the service to finish starting, then check readiness and the model IDs it advertises:

```python
from sdm_client import SDMClient

with SDMClient(url="http://localhost:8001") as client:
    ready = client.health_ready()
    print(ready)
    if ready:
        print(client.models())
```

`health_ready()` returns `False` for a non-200 response and raises `StructuredDataError` with code `TRANSPORT_ERROR` if the service cannot be reached yet.
Retry the check while the container starts, and call `models()` only after readiness is `True`.
Continue when the endpoint advertises the model you intend to use.
Pass the **base URL**, such as `http://localhost:8001`; `SDMClient` adds the `/v1/...` routes.
`KumoTabular()` and `KumoRelational()` select the model to call on that endpoint.
These selectors are distinct from the in-process model implementations in `sdm.models`.
They do not start a model or connect to a different deployment.

(nim-authentication)=

### Authentication

Registry authentication grants access to the container image.
Inference authentication is configured separately by the deployment operator.
If an authenticating gateway accepts `X-API-Key`, supply its key explicitly with `SDMClient(url="https://your-nim-host", api_key=...)`.
`SDMClient` sends that value as `X-API-Key` and requires HTTPS when sending credentials to a non-local host.

NVIDIA's hosted API Catalog uses model-specific URLs and Bearer authentication.
`SDMClient` cannot call that hosted API.
The connection shown here is for a deployed NIM.
To call NVIDIA's hosted API, follow the model's API example on NVIDIA Build.

## Predict with Kumo Tabular

Supply labeled context rows and query rows with the same feature columns.
Leave the target out of the query rows:

```python
import pandas as pd
from sdm_client import KumoTabular, Request, SDMClient

context = pd.DataFrame(
    {
        "age": [31.0, 44.5, 28.0, 51.0, 39.0, 57.0],
        "spend": [120.0, 430.0, 90.0, 510.0, 260.0, 600.0],
        "churn": [0, 1, 0, 1, 0, 1],
    }
)
to_score = pd.DataFrame({"age": [33.0, 47.0], "spend": [150.0, 480.0]})
tabular_request = Request(
    context=context,
    predict=to_score,
    target="churn",
    task="classification",
)

with SDMClient(url="http://localhost:8000") as client:
    result = client.predict(tabular_request, model=KumoTabular())
    print(result.to_pandas())
```

The context teaches the prediction task at inference time, without updating model weights.
Classification returns predictions and class probabilities by default.
For a numerical target, use `task="regression"`.

## Predict with Kumo Relational

Describe the tables, their keys, their relationships, and their time columns.
Then express the prediction with the Predictive Query Language (PQL).
This example asks for each user's spending over the next 30 days:

```python
import pandas as pd
from sdm_client import (
    KumoRelational,
    PredictiveQuery,
    Relationship,
    Schema,
    SDMClient,
)

users = pd.DataFrame({"user_id": range(20)})
orders = pd.DataFrame(
    {
        "order_id": range(240),
        "user_id": [i % 20 for i in range(240)],
        "amount": [10.0 + i % 17 for i in range(240)],
        "ordered_at": pd.date_range("2025-01-01", periods=240, freq="D"),
    }
)
schema = Schema.from_data(
    {"users": users, "orders": orders},
    primary_keys={"users": "user_id", "orders": "order_id"},
    time_columns={"users": None, "orders": "ordered_at"},
    relationships=[Relationship("orders", "user_id", "users", "user_id")],
)
model = KumoRelational()
query = PredictiveQuery(
    "PREDICT SUM(orders.amount, 0, 30, days) FOR EACH users.user_id",
    schema,
)
relational_request = query.materialize(
    indices=[1, 2, 3],
    model=model,
    num_context_samples=64,
)

with SDMClient(url="http://localhost:8001") as client:
    result = client.predict(relational_request, model=model)
    print(result.to_pandas())
```

`materialize()` resolves the query, constructs historical labeled examples, and samples the related records needed for prediction.
The sampled data and schema are included in the request.
The NIM does not connect to your database.

The **prediction anchor** is the timestamp from which the next 30 days are measured.
Features come from records at or before that timestamp, and labels come from the later target window.
`0, 30, days` denotes `(anchor, anchor + 30 days]`.
Use data spanning comfortably more than the target window so historical labels can be constructed.
`materialize()` reports when no eligible context can be built.

## Data Connectors

Data connectors (`sdm_connectors`) read input data using your database connection, warehouse session, or file path.
You choose the data source independently of where the model runs: for example, read tables from Snowflake and send a prediction request to a NIM on another host.
Add the platform extras below to the client installation command to install the required connectors and drivers.

| Data source             | Python API                                    | Installation extra                           |
| ----------------------- | --------------------------------------------- | -------------------------------------------- |
| Databricks              | `DatabricksSource`                            | `databricks`                                 |
| Snowflake               | `SnowflakeSource`                             | `snowflake`                                  |
| PostgreSQL              | `PostgresSource`                              | `postgres`                                   |
| DuckDB                  | `DuckDBSource`                                | `duckdb`                                     |
| SQLite                  | `SQLiteSource`                                | `sqlite`                                     |
| Databricks Lakebase     | `lakebase()` connection with `PostgresSource` | `lakebase`                                   |
| S3 CSV/Parquet files    | `read(path)`                                  | `s3`                                         |
| Local CSV/Parquet files | `read(path)`                                  | No driver extra; connectors package required |

On Databricks, install `lakebase` instead of `postgres` or `all`, even for PostgreSQL outside Lakebase.
This avoids the bundled OpenSSL in `psycopg[binary]`, which fails its FIPS self-test on Databricks aarch64 hosts.

The `relational` extra also installs the connectors package.
If you only installed the base package for Tabular inference and want to use local file readers, add the connectors package:

```bash
python -m pip install structured-data-models-connectors
```

The Lakebase connection helper is imported from `sdm_connectors.databricks`.

Database sources with `Schema.from_source()` are for Kumo Relational.
`materialize()` samples the records needed for a prediction instead of loading every table into memory.
For Kumo Tabular, use `read(path)` for files or `read_table(connection, table=...)` for database tables to load rows into pandas DataFrames, then pass those DataFrames to `Request`.
`read_table()` requires a DB-API connection, such as one from `databricks.sql.connect(...)`, `warehouse(...)`, or `snowflake.connector.connect(...)`.
For a Snowpark session, pass its `.connection`; a Spark or Snowpark session itself is not a DB-API connection.

### Databricks and Snowflake

For Databricks:

```bash
python -m pip install "structured-data-models-client[relational,databricks]"
```

For Snowflake:

```bash
python -m pip install "structured-data-models-client[relational,snowflake]"
```

For installation in a Databricks notebook, follow {ref}`nim-databricks-notebooks`.

The following source examples assume that tables equivalent to `users` and `orders` above already exist in the selected warehouse.
In a Databricks notebook with an existing `spark` session:

```python
from sdm_connectors import DatabricksSource

source = DatabricksSource(spark, catalog="main", schema="sales")
```

This Spark path requires Databricks Runtime 17.3 LTS or serverless environment 4 and later.
For a SQL warehouse, open a connection with `databricks.sql.connect(...)` or `warehouse(warehouse_id)` from `sdm_connectors.databricks`.
The `warehouse()` helper uses the Databricks SDK's configured workspace identity.
Pass that connection to `DatabricksSource(connection, catalog=..., schema=...)`.

With an existing Snowpark session or Snowflake connector connection named `snowflake_session`:

```python
from sdm_connectors import SnowflakeSource

source = SnowflakeSource(snowflake_session, database="SALES", schema="PUBLIC")
```

Column names in schema arguments and PQL must match the names reported by the warehouse exactly.
The shared example below assumes the lowercase column names from the DataFrames above.
In Snowflake, those names require columns created with quoted lowercase identifiers.
For ordinary unquoted Snowflake columns, use `USER_ID`, `ORDER_ID`, `ORDERED_AT`, and `AMOUNT` in the schema arguments and PQL, such as `users.USER_ID` and `orders.AMOUNT`.

After choosing one source, build the schema with the same keys and relationships:

```python
from sdm_client import (
    KumoRelational,
    PredictiveQuery,
    Relationship,
    Schema,
    SDMClient,
)

model = KumoRelational()
warehouse_schema = Schema.from_source(
    source,
    {"users": "users", "orders": "orders"},
    primary_keys={"users": "user_id", "orders": "order_id"},
    time_columns={"users": None, "orders": "ordered_at"},
    relationships=[Relationship("orders", "user_id", "users", "user_id")],
)
warehouse_request = PredictiveQuery(
    "PREDICT SUM(orders.amount, 0, 30, days) FOR EACH users.user_id",
    warehouse_schema,
).materialize(indices=[1, 2, 3], model=model, num_context_samples=64)

with SDMClient(url="https://your-nim-host") as client:
    print(client.predict(warehouse_request, model=model).to_pandas())
```

Schema construction reads catalog metadata, previews up to 1,000 rows per table, and checks primary keys with a full scan of each eligible table.
Pass `check_keys=False` to skip the key scans when you have already validated those keys.
`materialize()` samples records with SQL and includes them in the prediction request.
The source uses your warehouse permissions.
It does not close your connection; close it yourself when you are done.
The notebook must be able to reach the NIM host over the network; supply its base URL and, if required by the gateway, its API key as described under {ref}`nim-authentication`.

## Databricks Model Serving

To call a model on Databricks Model Serving, connect by its serving endpoint name.
The endpoint must serve the Kumo Relational MLflow package with the `request_json` input and `response_json` output fields.
Obtain that model package through your deployment provider, or ask your workspace operator to register it and create the serving endpoint.
This section covers invocation of an existing compatible endpoint.
Install the `relational` and `databricks-serving` extras:

```bash
python -m pip install "structured-data-models-client[relational,databricks-serving]"
```

For notebook setup, see {ref}`nim-databricks-notebooks`.

Use the **serving endpoint name**, and the Relational request prepared above:

```python
with SDMClient.for_databricks_serving("kumo-relational") as client:
    result = client.predict(relational_request, model=KumoRelational())
    print(result.to_pandas())
```

Authentication comes from the Databricks SDK's configured workspace identity, including the notebook's identity when supported by its environment.
For another workspace or credential, pass `workspace_client=WorkspaceClient(...)` from `databricks.sdk`.
Each call sends one `request_json` row and reads one `response_json` row.
The endpoint must support these fields; an MLflow model with a different input signature will not work.
This path supports one-shot `predict()`; `fit()` and `models()` are unavailable.
`predict()` checks Databricks' 16 MB limit against the wrapped request; reduce the number of context samples or entities per call when a request exceeds it.
See [Databricks custom-model endpoint documentation](https://docs.databricks.com/aws/en/machine-learning/model-serving/score-custom-model-endpoints) for the platform's invocation interfaces.

## Snowflake Model Serving

The `SnowflakeSource` connector reads tables; it does not call a model hosted by Snowflake.
`SDMClient` currently has no public connection for Snowflake Model Serving.
To use Snowflake data with the Python workflow in this guide, sample tables as above and send the request to a deployed NIM.
For a model hosted through the Snowflake Model Registry on Snowpark Container Services, use Snowflake's serving interfaces.
See [Snowflake model serving](https://docs.snowflake.com/en/developer-guide/snowflake-ml/inference/real-time-inference-rest-api) for the platform's own serving interfaces.

## Reuse Context with a Session

A NIM that supports sessions can retain context across prediction calls for the same task and schema.
Using the schema, model, and query from the Relational example:

```python
more_request = query.materialize(
    indices=[4, 5],
    model=model,
    num_context_samples=64,
)

with SDMClient(url="http://localhost:8001") as client:
    with client.fit(
        relational_request,
        model=model,
        holdout=[1, 2, 3, 4, 5],
    ) as fitted:
        first = fitted.predict(relational_request)
        second = fitted.predict(more_request)
        print(first.to_pandas())
        print(second.to_pandas())
```

`fit()` uploads and pins context; it does not train the model.
The fitted handle sends query data against that context, and its context manager releases the session.
Declare every entity the session may score in `holdout` before pinning context, so those entities cannot become their own labeled examples.
`fit()` emits an `SDMWarning` when it removes holdout rows from the context; this is expected and reduces the context size.
For temporal tasks, this excludes those entities at every anchor; use one-shot prediction when historical examples for the same entity are needed.
If the deployment does not support sessions, use `client.predict()`.
Sessions live in the serving process and can expire, be evicted, or disappear after a restart.
Route session calls to the same serving process; create a new session or use one-shot prediction if the session is gone.

## Prediction Results

`client.predict()` and the fitted handle return a `PredictionResult`.
`result.to_pandas()` exposes fields such as `prediction`, `probability_<class>`, and `quantile_<level>`, depending on the task and requested outputs.
Use the returned entity `id` when present, or `row_index` for a positional request, to join predictions to the input.
Do not assume the endpoint preserves row order.

Endpoint capabilities depend on the deployed model and version.
`client.models()` reports the model IDs advertised by a NIM, while `client.capabilities(model)` describes locally known capabilities rather than querying the deployment.
