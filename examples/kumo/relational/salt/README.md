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

| Task               | KumoRFM-1 | KumoRFM-2 | 1 estimator<br>50K context | 8 estimators<br>50K context | 1 estimator<br>100K context | 8 estimators<br>100K context |
| ------------------ | --------: | --------: | -------------------------: | --------------------------: | --------------------------: | ---------------------------: |
| Plant              |      0.99 |      0.99 |                     0.9898 |                      0.9913 |                      0.9901 |                       0.9932 |
| ShippingPoint      |      0.99 |      0.98 |                     0.9780 |                      0.9816 |                      0.9799 |                       0.9838 |
| Item Incoterm      |      0.79 |      0.78 |                     0.8057 |                      0.8090 |                      0.8078 |                       0.8101 |
| Header Incoterm    |      0.81 |      0.81 |                     0.8405 |                      0.8504 |                      0.8422 |                       0.8479 |
| Sales Office       |      1.00 |      1.00 |                     0.9993 |                      0.9994 |                      0.9994 |                       0.9994 |
| Sales Group        |      0.38 |      0.61 |                     0.5983 |                      0.6273 |                      0.6124 |                       0.6186 |
| Payment Terms      |      0.66 |      0.68 |                     0.6722 |                      0.6896 |                      0.6985 |                       0.7067 |
| Shipping Condition |      0.78 |      0.79 |                     0.8121 |                      0.8233 |                      0.8175 |                       0.8263 |
| Avg                |       0.8 |      0.83 |                     0.8370 |                      0.8465 |                      0.8435 |                       0.8483 |
