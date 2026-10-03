# Kaggle zero-shot evaluation examples

Zero-shot (in-context, no fine-tuning) evaluation of `KumoTabular` (both
`"small"` and `"large"`) and `TabICLv2` on several Kaggle competitions:

- [`santander_customer_transaction_prediction.py`](santander_customer_transaction_prediction.py) — [Santander Customer Transaction Prediction](https://www.kaggle.com/competitions/santander-customer-transaction-prediction) (binary classification)
- [`bioresponse.py`](bioresponse.py) — [Predicting a Biological Response](https://www.kaggle.com/competitions/bioresponse) (binary classification)
- [`itk_fall_2018.py`](itk_fall_2018.py) — [ITK Fall 2018](https://www.kaggle.com/competitions/itk-fall-2018) (binary classification; the APS Failure at Scania Trucks dataset)
- [`otto_group_product_classification_challenge.py`](otto_group_product_classification_challenge.py) — [Otto Group Product Classification Challenge](https://www.kaggle.com/competitions/otto-group-product-classification-challenge) (9-class classification)
- [`santander_value_prediction_challenge.py`](santander_value_prediction_challenge.py) — [Santander Value Prediction Challenge](https://www.kaggle.com/competitions/santander-value-prediction-challenge) (regression, ~4991 features against ~4459 rows)
- [`restaurant_revenue_prediction.py`](restaurant_revenue_prediction.py) — [Restaurant Revenue Prediction](https://www.kaggle.com/competitions/restaurant-revenue-prediction) (regression, only 137 training rows)
- [`mercari_price_suggestion_challenge.py`](mercari_price_suggestion_challenge.py) — [Mercari Price Suggestion Challenge](https://www.kaggle.com/competitions/mercari-price-suggestion-challenge) (regression with free-text features, ~1.48M training rows)

Each script cross-validates a local metric (repeated stratified 80/20
splits), then fits on the full labeled training data and writes a
`submission.csv` in the competition's required format. None of the scripts
call the Kaggle submission API — that upload is a manual step below.

## Setup

**Hugging Face**: pretrained checkpoints are downloaded from
`nvidia/Kumo-Tabular` via `huggingface_hub`, which reads its configured
login or the `HF_TOKEN` environment variable:

```bash
export HF_TOKEN=<your token>
# Optional: redirect the download cache, e.g. onto scratch storage.
export HF_HOME=/path/to/cache
```

**Kaggle**: install the CLI and set credentials. Recent versions of the
`kaggle` package (2.x) use `KAGGLE_API_TOKEN` rather than the older
`KAGGLE_USERNAME`/`KAGGLE_KEY` pair — generate a token at
[kaggle.com/settings/api](https://www.kaggle.com/settings/api):

```bash
pip install kaggle pandas scikit-learn
export KAGGLE_API_TOKEN=<your token>
```

You must also **join each competition on its Kaggle web page** (accept its
rules) before the API is allowed to download its data — this can't be done
through the API itself:

```bash
kaggle competitions download -c santander-customer-transaction-prediction -p data/santander
kaggle competitions download -c bioresponse -p data/bioresponse
kaggle competitions download -c itk-fall-2018 -p data/itk
kaggle competitions download -c otto-group-product-classification-challenge -p data/otto
kaggle competitions download -c santander-value-prediction-challenge -p data/santander-value
kaggle competitions download -c restaurant-revenue-prediction -p data/restaurant
kaggle competitions download -c mercari-price-suggestion-challenge -p data/mercari
# unzip each into its directory (mercari's train.tsv.7z needs `7z x`, not `unzip`)
```

## Running

```bash
python examples/kaggle/santander_customer_transaction_prediction.py \
  --data-dir data/santander --model kumo-large --output santander_submission.csv

python examples/kaggle/bioresponse.py \
  --data-dir data/bioresponse --model kumo-large --output bioresponse_submission.csv

python examples/kaggle/itk_fall_2018.py \
  --data-dir data/itk --model kumo-large --output itk_submission.csv
```

`--model` accepts `kumo-small`, `kumo-large`, or `tabiclv2` (run all three
to compare Kumo against the `TabICLv2` baseline on identical splits). If
Hugging Face Hub access to `nvidia/Kumo-Tabular` isn't available, pass
`--local-checkpoint /path/to/checkpoint.pt` to load a `KumoTabular`
checkpoint file directly instead. See each script's `--help` for the rest
of the knobs (`--num-splits`, `--val-fraction`, `--num-estimators`,
`--max-context-size`, `--seed`).

### On the NVIDIA dl cluster

```bash
sbatch --partition=general --gres=gpu:H100:1 \
  --constraint="x86_64&[ipp2-1|ipp2-2]" \
  scripts/submit.sh <conda-env> examples/kaggle/santander_customer_transaction_prediction.py \
  --data-dir /path/to/data/santander --model kumo-large --output /path/to/results/santander_submission.csv
```

(`scripts/submit.sh` is the SBATCH template from the `submit-dlcluster-job`
skill; see that skill for GPU-availability checks, scratch-pod constraints,
and job monitoring.)

## Results

Zero-shot, default settings (5 repeated stratified 80/20 splits,
`--num-estimators 8`, full unsubsampled training context, seed 0):

| Competition | Metric | kumo-small | kumo-large | tabiclv2 |
|---|---|---|---|---|
| ITK Fall 2018 | AUC | 0.9922 ± 0.0008 | 0.9921 ± 0.0007 | 0.9912 ± 0.0017 |
| ITK Fall 2018 | log loss (hard labels) | 0.1688 ± 0.0236 | 0.1622 ± 0.0275 | 0.1976 ± 0.0219 |
| ITK Fall 2018 | cost (10×FP + 500×FN) | 20554 ± 2890 | 20238 ± 3889 | 22414 ± 1807 |
| Santander Customer Transaction Prediction | AUC | 0.8881 ± 0.0031 | 0.8817 ± 0.0041 | 0.8907 ± 0.0017 |
| Bioresponse | log loss | 0.4215 ± 0.0130 | 0.4170 ± 0.0147 | 0.4533 ± 0.0188 |

`kumo-large` was run from a local checkpoint (`--local-checkpoint`) rather
than the Hugging Face Hub: the account's HF token(s) only ever resolved
`small/classifier.pt` in `nvidia/Kumo-Tabular` (404s on everything else,
including plain repo listing), which traced to the account not being a
member of whichever HF **resource group** that repo lives under in
NVIDIA's org — not fixable by regenerating a token. `--local-checkpoint`
loads a checkpoint file directly (mirroring
`KumoTabular._load_from_pretrained`'s remapping), bypassing the Hub
download entirely for either size.

Neither `kumo-small`/`kumo-large` nor `tabiclv2` wins outright:
`kumo-{small,large}` outperform `tabiclv2` on ITK and Bioresponse (lower
log loss/cost, higher AUC on ITK), but `tabiclv2` edges out both Kumo
sizes on Santander (0.8907 AUC vs. 0.8881 and 0.8817). `kumo-large` beats
`kumo-small` on ITK and Bioresponse by a small margin but is slightly
worse on Santander — scaling up isn't a clean win here either.

For external context, ITK Fall 2018's public leaderboard top score is
`-0.46375` (`kaggle competitions leaderboard -c itk-fall-2018 --show`).
That score's range (roughly -0.46 to -0.7 across the leaderboard) looks
like negated log loss of hard true/false predictions, consistent with the
values above, rather than the classic APS Failure cost metric — the exact
metric couldn't be confirmed from the competition's (JS-rendered) page.

`otto_group_product_classification_challenge.py`, `santander_value_prediction_challenge.py`,
`restaurant_revenue_prediction.py`, and `mercari_price_suggestion_challenge.py` follow the
same pattern for multi-class classification and regression respectively;
see each script's module docstring for competition-specific notes (e.g.
Mercari's free-text columns and its scale requiring
`--max-context-size`/`--query-batch-size` by default, Santander Value's
`--max-columns` column-selection override, Restaurant Revenue's optional
`--drop-city`/`--pca-components` feature engineering).

## Submitting to Kaggle

The scripts only ever write a local `submission.csv` — review it, then
submit it yourself:

```bash
kaggle competitions submit -c santander-customer-transaction-prediction \
  -f santander_submission.csv -m "<message>"
```
