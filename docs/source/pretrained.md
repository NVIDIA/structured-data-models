# Pretrained checkpoints

Pretrained model checkpoints are downloaded and cached by
[`huggingface_hub`](https://huggingface.co/docs/huggingface_hub/en/guides/download).
Public checkpoints do not require authentication. To use a private checkpoint,
such as the KumoRFM-2 weights, authenticate with a Hugging Face account that
has access to the repository:

```bash
hf auth login
```

Automated environments can instead provide a read token through `HF_TOKEN`:

```bash
HF_TOKEN=hf_... python -c "from sdm.models import KumoRFM; KumoRFM()"
```

Do not store Hugging Face tokens in source control. Use the secret store of the
CI or deployment environment.
