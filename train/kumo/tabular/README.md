# Kumo Tabular training

[NVIDIA Kumo Tabular](https://arxiv.org/abs/2611.23001) is a pretrained tabular foundation model for classification and regression. Given labeled train set rows, it predicts test set labels in a single forward pass. It comes in three sizes (28M to 215M parameters) and was pretrained only on artificial tables sampled from structural causal models. Classification and regression are trained as separate models in three stages that increase the maximum dataset length from 1,024 to 60,000 rows. This repository contains the structural causal model generators ([`sdm.prior`](https://nvidia.github.io/structured-data-models/api/prior)) and the training recipe. Pretrained weights are released under the permissive OpenMDW-1.1 license on [Hugging Face](https://huggingface.co/nvidia/Kumo-Tabular).

## Setup

```bash
# The Muon optimizer requires PyTorch >=2.9.
pip install -e ".[prior]" "torch>=2.9"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export TORCHINDUCTOR_CACHE_DIR=/tmp/sdm-inductor-cache
export PYTORCH_ALLOC_CONF=expandable_segments:True
```

Run these commands from the repository root on CUDA GPUs. For distributed training, replace `python` with `torchrun --standalone --nproc-per-node=<gpus>`; the global batch size is shared across ranks and must be divisible by GPU count × group size.

Each stage starts fresh optimizers from the previous stage's training checkpoint. To recover an interrupted stage, repeat its command with `--resume checkpoints/<stage>/step-<step>.pt` and remove `--init-checkpoint`, retaining the original total `--max-steps`. Recovery retains the five newest checkpoints.

The expandable allocator is required for Medium S3 on the tested H100 80 GB setup, where the column ladder stays wider at long context.

## Classification

### Small

```bash
python -m train.kumo.tabular.train --task classification --size small --output ckpt-kumo-small-classification-stage1.pt --max-steps 500000 --lr 3e-4 --min-rows 1024 --max-rows 1024 --min-train-size 0.3 --max-train-size 0.9 --global-batch-size 64 --group-size 8 --mem-preserve-threshold 0 --warmup-proportion 0.02 --lr-end-factor 0 --grad-clip 10 --seed 42 --precision bf16 --workers -1 --compile --checkpoint-dir checkpoints/small-classification-stage1 --relabel-prob 0 --coarsen-prob 0 --high-cardinality-prob 0 --near-empty-missingness-prob 0 --latent-driver-missingness-prob 0 --softmax-argmax-prob 0 --keep-trivial-prob 0

python -m train.kumo.tabular.train --task classification --size small --init-checkpoint ckpt-kumo-small-classification-stage1.pt --output ckpt-kumo-small-classification-stage2.pt --max-steps 40000 --lr 4e-5 --min-rows 400 --max-rows 10240 --min-train-size 0.79 --max-train-size 0.81 --global-batch-size 128 --group-size 4 --mem-preserve-threshold 2048 --warmup-proportion 0.01 --lr-end-factor 0.001 --grad-clip 10 --seed 42 --precision bf16 --workers -1 --checkpoint-dir checkpoints/small-classification-stage2

python -m train.kumo.tabular.train --task classification --size small --init-checkpoint ckpt-kumo-small-classification-stage2.pt --output ckpt-kumo-small-classification-stage3.pt --max-steps 12000 --lr 1e-5 --min-rows 400 --max-rows 60000 --min-train-size 0.79 --max-train-size 0.81 --global-batch-size 128 --group-size 2 --mem-preserve-threshold 512 --warmup-proportion 0.01 --lr-end-factor 0.01 --grad-clip 1 --seed 42 --replay-small --precision bf16 --workers -1 --checkpoint-dir checkpoints/small-classification-stage3 --max-columns-ladder '[[null,100]]'
```

### Medium

```bash
python -m train.kumo.tabular.train --task classification --size medium --output ckpt-kumo-medium-classification-stage1.pt --max-steps 500000 --lr 3e-4 --min-rows 1024 --max-rows 1024 --min-train-size 0.3 --max-train-size 0.9 --global-batch-size 128 --group-size 2 --mem-preserve-threshold 0 --warmup-proportion 0.02 --lr-end-factor 0 --grad-clip 10 --seed 42 --precision bf16 --workers -1 --compile --checkpoint-dir checkpoints/medium-classification-stage1 --relabel-prob 0 --coarsen-prob 0 --high-cardinality-prob 0 --near-empty-missingness-prob 0 --latent-driver-missingness-prob 0 --softmax-argmax-prob 0 --keep-trivial-prob 0

python -m train.kumo.tabular.train --task classification --size medium --init-checkpoint ckpt-kumo-medium-classification-stage1.pt --output ckpt-kumo-medium-classification-stage2.pt --max-steps 40000 --lr 4e-5 --min-rows 400 --max-rows 10240 --min-train-size 0.79 --max-train-size 0.81 --global-batch-size 128 --group-size 1 --mem-preserve-threshold 2048 --warmup-proportion 0.01 --lr-end-factor 0.001 --grad-clip 10 --seed 42 --precision bf16 --workers -1 --checkpoint-dir checkpoints/medium-classification-stage2

python -m train.kumo.tabular.train --task classification --size medium --init-checkpoint ckpt-kumo-medium-classification-stage2.pt --output ckpt-kumo-medium-classification-stage3.pt --max-steps 12000 --lr 1e-5 --min-rows 400 --max-rows 60000 --min-train-size 0.79 --max-train-size 0.81 --global-batch-size 128 --group-size 1 --mem-preserve-threshold 512 --warmup-proportion 0.01 --lr-end-factor 0.01 --grad-clip 1 --seed 42 --replay-small --precision bf16 --workers -1 --checkpoint-dir checkpoints/medium-classification-stage3 --max-columns-ladder '[[16000,100],[20000,100],[25000,100],[32000,100],[36000,88],[40000,80],[45000,72],[50000,64],[55000,58],[60000,53],[null,48]]'
```

### Large

```bash
python -m train.kumo.tabular.train --task classification --size large --output ckpt-kumo-large-classification-stage1.pt --max-steps 1000000 --lr 3e-4 --min-rows 1024 --max-rows 1024 --min-train-size 0.3 --max-train-size 0.9 --global-batch-size 128 --group-size 2 --mem-preserve-threshold 0 --warmup-proportion 0.02 --lr-end-factor 0 --grad-clip 10 --seed 42 --precision bf16 --workers -1 --compile --checkpoint-dir checkpoints/large-classification-stage1 --relabel-prob 0 --coarsen-prob 0 --high-cardinality-prob 0 --near-empty-missingness-prob 0 --latent-driver-missingness-prob 0 --softmax-argmax-prob 0 --keep-trivial-prob 0

python -m train.kumo.tabular.train --task classification --size large --init-checkpoint ckpt-kumo-large-classification-stage1.pt --output ckpt-kumo-large-classification-stage2.pt --max-steps 60000 --lr 4e-5 --min-rows 400 --max-rows 10240 --min-train-size 0.79 --max-train-size 0.81 --global-batch-size 128 --group-size 1 --mem-preserve-threshold 2048 --warmup-proportion 0.01 --lr-end-factor 0.001 --grad-clip 10 --seed 42 --precision bf16 --workers -1 --checkpoint-dir checkpoints/large-classification-stage2

python -m train.kumo.tabular.train --task classification --size large --init-checkpoint ckpt-kumo-large-classification-stage2.pt --output ckpt-kumo-large-classification-stage3.pt --max-steps 18000 --lr 1e-5 --min-rows 400 --max-rows 60000 --min-train-size 0.79 --max-train-size 0.81 --global-batch-size 128 --group-size 1 --mem-preserve-threshold 512 --warmup-proportion 0.01 --lr-end-factor 0.01 --grad-clip 1 --seed 42 --replay-small --precision bf16 --workers -1 --checkpoint-dir checkpoints/large-classification-stage3 --max-columns-ladder '[[16000,100],[20000,100],[25000,100],[32000,78],[36000,69],[40000,61],[45000,55],[50000,48],[55000,44],[60000,40],[null,40]]'
```

## Regression

### Small

```bash
python -m train.kumo.tabular.train --task regression --size small --output ckpt-kumo-small-regression-stage1.pt --max-steps 500000 --lr 3e-4 --min-rows 1024 --max-rows 1024 --min-train-size 0.3 --max-train-size 0.9 --global-batch-size 64 --group-size 8 --mem-preserve-threshold 0 --warmup-proportion 0.02 --lr-end-factor 0 --grad-clip 10 --seed 42 --precision fp16 --workers -1 --compile --y-heavy-tail-prob 0 --checkpoint-dir checkpoints/small-regression-stage1 --min-regression-train-std 0.1 --relabel-prob 0 --coarsen-prob 0 --high-cardinality-prob 0 --near-empty-missingness-prob 0 --latent-driver-missingness-prob 0 --softmax-argmax-prob 0 --keep-trivial-prob 0

python -m train.kumo.tabular.train --task regression --size small --init-checkpoint ckpt-kumo-small-regression-stage1.pt --output ckpt-kumo-small-regression-stage2.pt --max-steps 40000 --lr 4e-5 --min-rows 400 --max-rows 10240 --min-train-size 0.79 --max-train-size 0.81 --global-batch-size 128 --group-size 4 --mem-preserve-threshold 2048 --warmup-proportion 0.01 --lr-end-factor 0.001 --grad-clip 10 --seed 42 --precision bf16 --workers -1 --y-heavy-tail-prob 0 --checkpoint-dir checkpoints/small-regression-stage2 --min-regression-train-std 0.1

python -m train.kumo.tabular.train --task regression --size small --init-checkpoint ckpt-kumo-small-regression-stage2.pt --output ckpt-kumo-small-regression-stage3.pt --max-steps 12000 --lr 1e-5 --min-rows 400 --max-rows 60000 --min-train-size 0.79 --max-train-size 0.81 --global-batch-size 128 --group-size 2 --mem-preserve-threshold 512 --warmup-proportion 0.01 --lr-end-factor 0.01 --grad-clip 1 --seed 42 --replay-small --precision bf16 --workers -1 --y-heavy-tail-prob 0.2 --checkpoint-dir checkpoints/small-regression-stage3 --max-columns-ladder '[[null,100]]' --min-regression-train-std 0.1
```

### Medium

```bash
python -m train.kumo.tabular.train --task regression --size medium --output ckpt-kumo-medium-regression-stage1.pt --max-steps 500000 --lr 3e-4 --min-rows 1024 --max-rows 1024 --min-train-size 0.3 --max-train-size 0.9 --global-batch-size 128 --group-size 2 --mem-preserve-threshold 0 --warmup-proportion 0.02 --lr-end-factor 0 --grad-clip 10 --seed 42 --precision fp16 --workers -1 --compile --y-heavy-tail-prob 0 --checkpoint-dir checkpoints/medium-regression-stage1 --min-regression-train-std 0.1 --relabel-prob 0 --coarsen-prob 0 --high-cardinality-prob 0 --near-empty-missingness-prob 0 --latent-driver-missingness-prob 0 --softmax-argmax-prob 0 --keep-trivial-prob 0

python -m train.kumo.tabular.train --task regression --size medium --init-checkpoint ckpt-kumo-medium-regression-stage1.pt --output ckpt-kumo-medium-regression-stage2.pt --max-steps 40000 --lr 4e-5 --min-rows 400 --max-rows 10240 --min-train-size 0.79 --max-train-size 0.81 --global-batch-size 128 --group-size 1 --mem-preserve-threshold 2048 --warmup-proportion 0.01 --lr-end-factor 0.001 --grad-clip 10 --seed 42 --precision bf16 --workers -1 --y-heavy-tail-prob 0 --checkpoint-dir checkpoints/medium-regression-stage2 --min-regression-train-std 0.1

python -m train.kumo.tabular.train --task regression --size medium --init-checkpoint ckpt-kumo-medium-regression-stage2.pt --output ckpt-kumo-medium-regression-stage3.pt --max-steps 12000 --lr 1e-5 --min-rows 400 --max-rows 60000 --min-train-size 0.79 --max-train-size 0.81 --global-batch-size 128 --group-size 1 --mem-preserve-threshold 512 --warmup-proportion 0.01 --lr-end-factor 0.01 --grad-clip 1 --seed 42 --replay-small --precision bf16 --workers -1 --y-heavy-tail-prob 0.2 --checkpoint-dir checkpoints/medium-regression-stage3 --max-columns-ladder '[[16000,100],[20000,100],[25000,100],[32000,100],[36000,88],[40000,80],[45000,72],[50000,64],[55000,58],[60000,53],[null,48]]' --min-regression-train-std 0.1
```

### Large

```bash
python -m train.kumo.tabular.train --task regression --size large --output ckpt-kumo-large-regression-stage1.pt --max-steps 1000000 --lr 3e-4 --min-rows 1024 --max-rows 1024 --min-train-size 0.3 --max-train-size 0.9 --global-batch-size 128 --group-size 2 --mem-preserve-threshold 0 --warmup-proportion 0.02 --lr-end-factor 0 --grad-clip 10 --seed 42 --precision fp16 --workers -1 --compile --y-heavy-tail-prob 0 --checkpoint-dir checkpoints/large-regression-stage1 --min-regression-train-std 0.1 --relabel-prob 0 --coarsen-prob 0 --high-cardinality-prob 0 --near-empty-missingness-prob 0 --latent-driver-missingness-prob 0 --softmax-argmax-prob 0 --keep-trivial-prob 0

python -m train.kumo.tabular.train --task regression --size large --init-checkpoint ckpt-kumo-large-regression-stage1.pt --output ckpt-kumo-large-regression-stage2.pt --max-steps 60000 --lr 4e-5 --min-rows 400 --max-rows 10240 --min-train-size 0.79 --max-train-size 0.81 --global-batch-size 128 --group-size 1 --mem-preserve-threshold 2048 --warmup-proportion 0.01 --lr-end-factor 0.001 --grad-clip 10 --seed 42 --precision bf16 --workers -1 --y-heavy-tail-prob 0 --checkpoint-dir checkpoints/large-regression-stage2 --min-regression-train-std 0.1

python -m train.kumo.tabular.train --task regression --size large --init-checkpoint ckpt-kumo-large-regression-stage2.pt --output ckpt-kumo-large-regression-stage3.pt --max-steps 18000 --lr 1e-5 --min-rows 400 --max-rows 60000 --min-train-size 0.79 --max-train-size 0.81 --global-batch-size 128 --group-size 1 --mem-preserve-threshold 512 --warmup-proportion 0.01 --lr-end-factor 0.01 --grad-clip 1 --seed 42 --replay-small --precision bf16 --workers -1 --y-heavy-tail-prob 0.2 --checkpoint-dir checkpoints/large-regression-stage3 --max-columns-ladder '[[16000,100],[20000,100],[25000,100],[32000,78],[36000,69],[40000,61],[45000,55],[50000,48],[55000,44],[60000,40],[null,40]]' --min-regression-train-std 0.1
```

## Flash attention

On Hopper GPUs, pass `--attention-backend fa3` with `--precision bf16` or `--precision fp16`, eager or with `--compile`. FlashAttention-3 is not an sdm dependency. Install the PyTorch-hosted `flash-attn-3` wheel for the CUDA version of your PyTorch build (`python -c "import torch; print(torch.version.cuda)"`), for example CUDA 13.0:

```bash
pip install "flash-attn-3>=3.0.0" --extra-index-url https://download.pytorch.org/whl/cu130
```
