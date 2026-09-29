# Kumo Relational on SALT

This example evaluates `KumoRelational` on the [SALT (Sales Autocompletion Linked Business Tables Dataset)](https://huggingface.co/datasets/SAP/SALT) benchmark.

## Run

```bash
python main.py --task=SALESOFFICE
python main.py --task=SALESGROUP
python main.py --task=CUSTOMERPAYMENTTERMS
python main.py --task=SHIPPINGCONDITION
python main.py --task=HEADERINCOTERMSCLASSIFICATION
python main.py --task=PLANT
python main.py --task=SHIPPINGPOINT
python main.py --task=ITEMINCOTERMSCLASSIFICATION
```

## Results

| Task               | KumoRFM-1 | KumoRFM-2 | 1 estimator<br>50K context | 8 estimators<br>50K context | 1 estimator<br>100K context | 8 estimators<br>100K context |
| ------------------ | --------: | --------: | -------------------------: | --------------------------: | --------------------------: | ---------------------------: |
| Plant              |  **0.99** |  **0.99** |                 **0.9898** |                  **0.9913** |                  **0.9901** |                   **0.9932** |
| ShippingPoint      |  **0.99** |      0.98 |                     0.9780 |                      0.9816 |                      0.9799 |                       0.9838 |
| Item Incoterm      |      0.79 |      0.78 |                     0.8057 |                      0.8090 |                      0.8078 |                   **0.8101** |
| Header Incoterm    |      0.81 |      0.81 |                     0.8405 |                      0.8504 |                      0.8422 |                   **0.8479** |
| Sales Office       |  **1.00** |  **1.00** |                 **0.9993** |                  **0.9994** |                  **0.9994** |                   **0.9994** |
| Sales Group        |      0.38 |      0.61 |                     0.5983 |                  **0.6273** |                      0.6124 |                       0.6186 |
| Payment Terms      |      0.66 |      0.68 |                     0.6722 |                      0.6896 |                      0.6985 |                   **0.7067** |
| Shipping Condition |      0.78 |      0.79 |                     0.8121 |                      0.8233 |                      0.8175 |                   **0.8263** |
| **Average**        |      0.80 |      0.83 |                     0.8370 |                      0.8465 |                      0.8435 |                   **0.8483** |

## cuGraph neighborhood sampling

[`cugraph_sampling.py`](cugraph_sampling.py) compares the CPU and CUDA samplers on the same SALT `SHIPPINGCONDITION` task, using the public [RelBench SALT release](https://huggingface.co/datasets/stanford-star/relbench-v2-extra) and its `sales-shipcond` split. Moving `RelationalData` to CUDA selects the cuGraph backend of `data.sampler()`. The example keeps 2,000 train/validation rows as context and evaluates the first 2,048 test rows with the same pretrained `KumoRelational` checkpoint. Both samplers use uniform temporal sampling and every eligible neighbor for two hops (`--num-neighbors -1 -1`). This is a sampler comparison on the RelBench split; its accuracy is not directly comparable with the MRR values above.

Install the project test dependencies and RelBench, then run:

```bash
uv sync --group test
uv pip install --python .venv/bin/python relbench
.venv/bin/python examples/relational/salt/cugraph_sampling.py
```

On an NVIDIA L4 with seed 0, the source tables contained 4,748,942 rows. The measured model inputs were:

| Sampler        | Context related rows | Query related rows | Test accuracy |
| -------------- | -------------------: | -----------------: | ------------: |
| PyG (CPU)      |               15,009 |             14,102 |        0.5552 |
| cuGraph (CUDA) |               15,009 |             14,102 |        0.5552 |

The cuGraph and CPU runs agreed on 99.71% of the 2,048 predicted classes and had the same measured accuracy. The context and query inputs were 316 and 337 times smaller than the full source tables by row count. These counts include repeated records in separate task neighborhoods. Sampling reduces what is passed to the model; the full source tables and graph still have to fit in memory before sampling, including GPU memory for cuGraph. The result measures this task slice and sampling configuration, not every SALT task or seed.
