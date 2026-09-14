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

The current implementation generally improves results with more in-context examples and more estimators.

| Task | KumoRFM-1 | KumoRFM-2 | SDM 1E / 100K | SDM 8E / 100K | SDM 1E / 50K | SDM 8E / 50K |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Plant | 0.99 | 0.99 | 0.9901 | 0.9932 | 0.9898 | 0.9913 |
| ShippingPoint | 0.99 | 0.98 | 0.9799 | 0.9838 | 0.978 | 0.9816 |
| Item Incoterm | 0.79 | 0.78 | 0.8078 | 0.8101 | 0.8057 | 0.809 |
| Header Incoterm | 0.81 | 0.81 | 0.8422 | 0.8479 | 0.8405 | 0.8504 |
| Sales Office | 1 | 1 | 0.9994 | 0.9994 | 0.9993 | 0.9994 |
| Sales Group | 0.38 | 0.61 | 0.6124 | 0.6186 | 0.5983 | 0.6273 |
| Payment Terms | 0.66 | 0.68 | 0.6985 | 0.7067 | 0.6722 | 0.6896 |
| Shipping Condition | 0.78 | 0.79 | 0.8175 | 0.8263 | 0.8121 | 0.8233 |
| Avg | 0.8 | 0.83 | 0.843475 | 0.84825 | 0.8369875 | 0.8464875 |
