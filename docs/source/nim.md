# NIM Inference

[NVIDIA Inference Microservices (NIMs)](https://developer.nvidia.com/nim) let us package and deploy structured data models behind an HTTP API for enterprise applications.
While the [in-context learning interface](icl) of this package runs models in the application's Python process, a NIM runs model execution in a separate GPU service, and applications send structured prediction requests to it over HTTP.
Thus, an application can call the model without loading its weights or managing a GPU runtime.

## From SDM to NIM

NIMs for structured data models follow SDM's unified prediction interface.
We have published the following NIMs:

| Model ID          | Model                                  | Deployment Instructions                                                            |
| ----------------- | -------------------------------------- | ---------------------------------------------------------------------------------- |
| `kumo-tabular`    | {py:class}`~sdm.models.KumoTabular`    | [Kumo Tabular on NVIDIA Build](https://build.nvidia.com/nvidia/kumo-tabular)       |
| `kumo-relational` | {py:class}`~sdm.models.KumoRelational` | [Kumo Relational on NVIDIA Build](https://build.nvidia.com/nvidia/kumo-relational) |

## The NIM API

NIMs for structured data models provide the following endpoints:

| Method   | Path                                    | Purpose                                              |
| -------- | --------------------------------------- | ---------------------------------------------------- |
| `GET`    | `/v1/health/live`                       | Check whether the service is running.                |
| `GET`    | `/v1/health/ready`                      | Check whether the model is ready to serve.           |
| `GET`    | `/v1/models`                            | List the model IDs served by the deployment.         |
| `POST`   | `/v1/predictions`                       | Send context and query data for one prediction call. |
| `POST`   | `/v1/sessions`                          | Retain context for subsequent prediction calls.      |
| `POST`   | `/v1/sessions/{session_id}/predictions` | Predict using context retained in a session.         |
| `DELETE` | `/v1/sessions/{session_id}`             | Release a session.                                   |

The prediction API mirrors {py:meth}`sdm.models.ICLModel.forward`, with inputs expressed as JSON payloads.
The main request fields are:

| Field     | Meaning                                                                    |
| --------- | -------------------------------------------------------------------------- |
| `model`   | The served model ID, such as `kumo-tabular` or `kumo-relational`.          |
| `task`    | The prediction task.                                                       |
| `schema`  | Column types and relationships between tables.                             |
| `context` | Context data.                                                              |
| `predict` | Query data.                                                                |
| `output`  | Requested results, such as predictions, class probabilities, or quantiles. |

NIM sessions support workflows similar to {py:meth}`~sdm.models.ICLModel.fit` and {py:meth}`~sdm.models.ICLModel.predict`, caching context examples and model state for repeated prediction calls.
Separate sessions allow multiple tenants to reuse the same deployment while maintaining their own prediction contexts.
See the [Kumo Tabular NIM reference](https://docs.api.nvidia.com/nim/reference/nvidia-kumo-tabular) and [Kumo Relational NIM reference](https://docs.api.nvidia.com/nim/reference/nvidia-kumo-relational) for the full request and response schemas.

## Python Client

Applications can call deployed NIMs for structured data models from Python using the [`structured-data-models-client`](https://github.com/NVIDIA/structured-data-models-client) package.
The client prepares structured prediction requests and handles the results, while inference runs in the deployed NIM.
See its [documentation](https://docs.nvidia.com/sdgm/client/overview) for installation instructions and usage examples.

Data preparation may run in an application, a Databricks notebook, or a job reading Snowflake or another database.
The [`structured-data-models-client`](https://github.com/NVIDIA/structured-data-models-client) uses data connectors to assemble the input tables under your existing data permissions, then sends those records to a reachable NIM endpoint.
Data access and model deployment can be configured independently.
For example, an application can read Snowflake tables and call a NIM running in your data center.
