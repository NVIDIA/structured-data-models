# Kumo Relational on SALT

This benchmark evaluates `KumoRelational` on the [SALT (Sales Autocompletion Linked Business Tables Dataset)](https://huggingface.co/datasets/SAP/SALT) benchmark.

## Run

```bash
python -m benchmark.relational.salt.kumo --task=SALESOFFICE
python -m benchmark.relational.salt.kumo --task=SALESGROUP
python -m benchmark.relational.salt.kumo --task=CUSTOMERPAYMENTTERMS
python -m benchmark.relational.salt.kumo --task=SHIPPINGCONDITION
python -m benchmark.relational.salt.kumo --task=HEADERINCOTERMSCLASSIFICATION
python -m benchmark.relational.salt.kumo --task=PLANT
python -m benchmark.relational.salt.kumo --task=SHIPPINGPOINT
python -m benchmark.relational.salt.kumo --task=ITEMINCOTERMSCLASSIFICATION
```
