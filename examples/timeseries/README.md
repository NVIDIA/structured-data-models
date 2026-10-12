# Time-series examples

[`compare_timesfm3.py`](compare_timesfm3.py) compares SDM and official TimesFM3 forecasts for one target without covariates. It prints forecast differences and a median forecast preview, then checks all nine sorted quantiles.

Install SDM and the reference implementation from the source revision used for checkpoint mapping:

```bash
uv pip install -e .
uv pip install "timesfm[torch] @ git+https://github.com/google-research/timesfm.git@e51928e27119cb17bebc005be2696b75e0a9e688"
python examples/timeseries/compare_timesfm3.py --device cpu
```

Both implementations load checkpoint revision `43046b85ec22d584a13f8098c2ed39c889e129c2`. The first download prompts for acceptance of the [TimesFM Non-Commercial License v1.0](https://huggingface.co/google/timesfm-3.0-pytorch/blob/43046b85ec22d584a13f8098c2ed39c889e129c2/LICENSE); use `--accept-license` if you have already accepted it. Use `--context-length` and `--horizon` to change the example, or `--device cuda` to run on a GPU.

The comparison disables upstream linear detrending and iterative CPM RevIN refinement to match SDM's current scope. It sorts SDM's quantiles in the script to match upstream's default sorting; the SDM model remains unchanged.
