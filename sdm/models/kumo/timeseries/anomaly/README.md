# Kumo-Anomaly components

First iteration of the native Kumo-Anomaly port from [NVIDIA/Kumo-TS](https://github.com/NVIDIA/Kumo-TS/blob/af74798bbf7d020d3298f545fb9be383560a18ce/Kumo-Anomaly/models/diff_models.py).

Implemented:

- `DiffusionEmbedding`: the discrete sinusoidal lookup and learned projections.
- `ResidualBlock`: temporal and feature-axis attention, diffusion-step and mask-strategy conditioning, and gated residual/skip outputs.
- `DiffusionDenoiser`: input/output projections and normalized skip aggregation for one noise-prediction step.

The components use PyTorch modules directly, with explicit constructor arguments and device/dtype support. The native post-norm `TransformerEncoderLayer` preserves the released normalization order and training-time dropout. Singleton time or feature axes bypass their attention and normalization, as in Kumo-TS.

The denoiser consumes normalized inputs of shape `[batch, input_channels, features, time]`, precomputed side information of shape `[batch, side_channels, features, time]`, and integer diffusion-step and masking-strategy tensors of shape `[batch]` or `[1]`. It returns predicted noise of shape `[batch, features, time]`, not reconstructed observations or anomaly scores.

Later iterations will add the pretrained checkpoint loader, side-information construction, diffusion schedule and sampling, reconstruction-based scores, preprocessing recipe, and public SDM model wrapper. Thresholding and SDK-level orchestration are outside this first PR. No public `sdm.models` detector is exported yet.
