# NIM Inference

`structured-data-models` (SDM) provides Python building blocks for developing and evaluating structured data models.
[NVIDIA NIM](https://developer.nvidia.com/nim) packages selected models as GPU inference microservices with an HTTP API for enterprise applications.

The SDM [in-context learning interface](icl) runs models in the application's Python process.
A NIM runs model execution in a separate GPU service, and applications send structured prediction requests to it over HTTP.
Applications can call the model without loading its weights or managing the GPU runtime.
Application capacity and GPU service capacity can be provisioned separately.

## From SDM to a NIM

The Kumo models in SDM are also published as NIMs.
Get the NIMs and their deployment instructions from NVIDIA Build:

| Model ID          | SDM model                              | Prediction inputs                                                | Deployment instructions                                                            |
| ----------------- | -------------------------------------- | ---------------------------------------------------------------- | ---------------------------------------------------------------------------------- |
| `kumo-tabular`    | {py:class}`~sdm.models.KumoTabular`    | One table; classification and regression                         | [Kumo Tabular on NVIDIA Build](https://build.nvidia.com/nvidia/kumo-tabular)       |
| `kumo-relational` | {py:class}`~sdm.models.KumoRelational` | Instance table and related tables; classification and regression | [Kumo Relational on NVIDIA Build](https://build.nvidia.com/nvidia/kumo-relational) |

Start by evaluating the model and prediction task on representative data with SDM.
When moving inference into an application, deploy the NIM for that model.
The NIM artifact defines the served model and supported API; the SDM Python interface and the NIM HTTP interface are separate contracts.
Local SDM customizations, such as changes to a {py:class}`~sdm.processing.recipe.Recipe` or fine-tuned weights, are not automatically carried into a published NIM.
Validate prediction quality on the deployed NIM with representative requests.

Follow the selected NIM's deployment instructions to run its container on supported NVIDIA GPU infrastructure.
NIMs can run in a data center or cloud environment managed by your organization.
Check the model's deployment page for hardware requirements, the container image, access requirements, and license terms.

Once deployed, applications call the service at its base URL.
The examples below assume a NIM at `http://localhost:8000`; replace this address with your deployment's URL.
Applications can use any HTTP client, regardless of the language or data platform they run in.

## The NIM API

Both Kumo NIMs provide these endpoints:

| Method   | Path                                    | Purpose                                              |
| -------- | --------------------------------------- | ---------------------------------------------------- |
| `GET`    | `/v1/health/live`                       | Check whether the service is running.                |
| `GET`    | `/v1/health/ready`                      | Check whether the model is ready to serve inference. |
| `GET`    | `/v1/models`                            | List the model IDs served by the deployment.         |
| `POST`   | `/v1/predictions`                       | Send context and query data for one prediction call. |
| `POST`   | `/v1/sessions`                          | Retain context for subsequent prediction calls.      |
| `POST`   | `/v1/sessions/{session_id}/predictions` | Predict using context retained in a session.         |
| `DELETE` | `/v1/sessions/{session_id}`             | Release a session.                                   |

Check readiness before sending predictions or listing models:

```bash
NIM_URL=http://localhost:8000
curl -fsS "$NIM_URL/v1/health/ready" && \
    curl -fsS "$NIM_URL/v1/models"
```

During startup, the service may be unreachable or return a non-success readiness response.
Retry until readiness returns HTTP 200, then use a model ID advertised by that deployment.

### Prediction Requests

`POST /v1/predictions` accepts a JSON request that describes the task and carries the input data.
The main fields are:

| Field     | Meaning                                                                       |
| --------- | ----------------------------------------------------------------------------- |
| `model`   | The served model ID, such as `kumo-tabular` or `kumo-relational`.             |
| `task`    | The prediction kind and target, with entity and time information when needed. |
| `schema`  | Column types, keys, and relationships that describe the tables.               |
| `context` | Labeled examples that define the task at inference time.                      |
| `predict` | Query rows to score, with their features and any required related records.    |
| `output`  | Requested results, such as predictions, class probabilities, or quantiles.    |

Optional `inference` and `metadata` fields carry supported execution options and request metadata.
Within `context` and `predict`, the `instance_table` holds the labeled examples or query rows.
The model's API reference defines the accepted tasks, table encodings, and output fields.
See the [Kumo Tabular](https://docs.api.nvidia.com/nim/reference/nvidia-kumo-tabular) and [Kumo Relational](https://docs.api.nvidia.com/nim/reference/nvidia-kumo-relational) references and the complete request examples on their NVIDIA Build pages.

As with SDM, labeled context rows define the prediction task at inference time without updating model weights.
For Kumo Tabular, context rows contain features and the target; query rows contain the same features without the target.
For Kumo Relational, the request also includes related tables and the relationships that link them to the instance table.

For time-dependent Kumo Relational predictions, set `task.anchor_time_column` to the instance table's timestamp column and `inference.use_prediction_time` to `true`.
Include only feature values and related records available at each row's anchor time, and use only context labels that were already known at the query rows' anchor times.
Sessions run with prediction time turned off, and Kumo Tabular does not support anchor times.

The NIM receives these prepared records; it does not connect to the source database or execute a predictive query against it.
Save a complete request from the selected model's deployment example as `request.json`, adapt it to your data, and submit it to the running NIM:

```bash
curl -fsS "$NIM_URL/v1/predictions" \
    -H 'Content-Type: application/json' \
    --data-binary @request.json
```

The call returns prediction results synchronously.
Outputs depend on the task and requested fields.
Match prediction records to input rows using `row_index`, the zero-based position of the query row, and the record's `id` when present.
Kumo Relational returns the query row's `instance_id` as that record's `id`.
The response-level `id` identifies the call, not an input row.

### Reusing Context

For repeated predictions with the same model, task, and schema, a session can retain context in the serving process.
Sessions are the HTTP counterpart of the SDM {py:meth}`~sdm.models.ICLModel.fit`+{py:meth}`~sdm.models.ICLModel.predict` flow.
Create it with `POST /v1/sessions`, use the returned session ID for subsequent predictions, and release it with `DELETE /v1/sessions/{session_id}`.
Session creation retains context; it does not train the model.

Sessions can expire, be evicted, or disappear after a restart.
Route follow-up calls to the same serving process and recreate the session if it is lost.
Use one-shot prediction when retaining context is unnecessary.

### Python Client

Applications can call deployed Kumo NIMs from Python with [`structured-data-models-client`](https://github.com/NVIDIA/structured-data-models-client).
The client prepares structured prediction requests and handles the results, while inference runs in the deployed NIM.
With Kumo Relational, it also answers ranking and multi-step forecasting questions by combining several classification or regression calls.

## Enterprise Integration

Data preparation may run in an application, a Databricks notebook, or a job reading Snowflake or another database.
Use database drivers or data connectors to assemble the input tables under your existing data permissions, then send those records to a reachable NIM endpoint.
Data access and model deployment can be configured independently.
For example, an application can read Snowflake tables and call a NIM running in your data center.

The deployment operator manages GPU capacity, image updates, network access, authentication, and monitoring.
The Kumo NIMs do not enforce application authentication on their prediction and session routes.
Deploy the service within the required network boundary and add TLS and authentication at a gateway, ingress, or service mesh in front of it.
Registry credentials only pull the container image; callers authenticate with the scheme configured at that gateway.

Use readiness checks when admitting traffic, and measure latency and throughput with representative requests before choosing capacity.
Request size, context size, and relational neighborhoods affect inference cost.
Batch work within the deployment's configured request limits, as described in the model's documentation.
For deployments with multiple serving processes, account for session affinity when routing requests.

NVIDIA's hosted API Catalog provides a separate way to prototype with available models.
Its model-specific URLs and Bearer authentication differ from a self-hosted NIM's routes and deployment authentication.
Follow the API example on the model's NVIDIA Build page when using the hosted service.
For broader deployment information, see the [NVIDIA NIM documentation](https://docs.nvidia.com/nim/).

## See Also

- [`structured-data-models-client` documentation](https://docs.nvidia.com/sdm/client/overview) for installing the Python client, building prediction requests, and running them against a deployed NIM.
- [Kumo Tabular NIM reference](https://docs.api.nvidia.com/nim/reference/nvidia-kumo-tabular) and [Kumo Relational NIM reference](https://docs.api.nvidia.com/nim/reference/nvidia-kumo-relational) for the full request and response schemas.
- [NVIDIA NIM documentation](https://docs.nvidia.com/nim/) for deployment, infrastructure, and operations guidance that applies across NIMs.
