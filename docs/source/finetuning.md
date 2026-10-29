# Fine-Tuning

Tabular foundation models (TFMs) such as {py:class}`~sdm.models.TabICLv2` and {py:class}`~sdm.models.KumoTabular` solve a new task through [in-context learning](icl): they read labeled context rows and predict query rows without updating their weights.
This works well out of the box, but the pretrained weights know nothing specific about your dataset.

**Full fine-tuning** trains every weight of the pretrained model on your dataset, and the result is still an in-context learner with the same interface.
On standard benchmarks, it helps on more datasets than it hurts: on TALENT, for example, it improved {py:class}`~sdm.models.TabICLv2` on 73 datasets and worsened it on 51.
A fine-tuned model can also use a smaller, cheaper inference context: on two large datasets, it came close to the pretrained model's full-context score with 20% to 47% of the training rows.
In exchange, each dataset needs its own GPU training run, its own copy of the weights, and a validation split to catch overfitting.

## How Full Fine-Tuning Works

Fine-tuning keeps the pretraining task, predicting query labels from labeled context rows, but samples each task from the training split of your dataset:

```{figure} images/finetuning_light.svg
:figclass: light-only
:width: 100%
```

```{figure} images/finetuning_dark.svg
:figclass: dark-only
:width: 100%
```

1. Sample rows from the training split and divide them into a **context** of *C* labeled rows and a **query** of *Q* rows.
2. Run the model on the labeled context and the query features.
3. Compute the loss of the query predictions against the held-out query labels.
4. Backpropagate and update every weight of the model.

```{note}
The code below for splitting, training, and keeping the best weights is condensed from [`examples/tabular/finetune.py`](https://github.com/NVIDIA/structured-data-models/blob/main/examples/tabular/finetune.py), which fine-tunes {py:class}`~sdm.models.TabICLv2` or {py:class}`~sdm.models.KumoTabular` on a classification or regression dataset.
```

## Data Splits

Fine-tuning splits the rows of a {py:class}`~sdm.tensor.TableTensor` (see [Table Semantics](tensor)) three ways:

- **Training** rows are sampled into context and query rows during fine-tuning, and serve as the context at inference.
- **Validation** rows are used to choose the weights and settings, and never appear in a training step.
- **Test** rows are used once, for the final evaluation.

```python
import torch
from sklearn.datasets import load_digits

import sdm

df = load_digits(as_frame=True).frame

table = sdm.TableTensor.from_pandas(
    df=df,
    stypes=sdm.infer_stypes(df, overrides={"target": "categorical"}),
    device="cuda",
)

# Split rows into training/validation/test (60/20/20):
index = torch.randperm(table.size(0), device="cuda")
num_train, num_val = int(0.6 * index.numel()), int(0.2 * index.numel())
train_table = table[index[:num_train]]
val_table = table[index[num_train:num_train + num_val]]
test_table = table[index[num_train + num_val:]]
```

## Training

Since {py:meth}`~sdm.models.ICLModel.forward` enables gradients in train mode (see [In-Context Learning](icl)), fine-tuning is an ordinary PyTorch loop that runs the model with a training recipe.
Every model ships a default {py:class}`~sdm.processing.recipe.Recipe` (see [Data Processing](processing)) whose output pipeline is built for inference: it combines the outputs of the ensemble members (*estimators*) and turns classification logits into probabilities.
For training, replace it with {py:class}`~sdm.processing.common.Identity` to compute the loss directly on the model outputs:

```python
import sdm.processing as sp

task = "classification"  # Use "regression" for a numerical target.
model = sdm.models.TabICLv2(task=task, device="cuda")

train_recipe = model.default_recipe()
train_recipe.output = sp.Identity()  # No post-processing.
```

The feature and target pipelines stay unchanged: they turn raw columns into the inputs that the pretrained model expects, and evaluation and inference use them too.
Some of their processors are random: {py:class}`~sdm.processing.common.ShuffleColumns` reorders the feature columns, and {py:class}`~sdm.processing.categorical.ShuffleCategories` relabels the classes.
At inference, this gives each estimator a different view of the same task (see Ensembling in [In-Context Learning](icl)).
During training, each step draws a new view, so the model cannot rely on a column's position or a class's label; this acts as data augmentation, like random crops and flips for images.

For regression with {py:class}`~sdm.models.KumoTabular`, remove the random sign flip ({py:class}`~sdm.processing.numerical.FlipSign`) from the target pipeline.
A flipped target reverses the predicted quantiles, which the default output pipeline re-sorts via {py:class}`~sdm.processing.output.SortQuantiles`, but the identity output does not:

```python
train_recipe.target = sp.StypeDispatch(  # Default target pipeline without FlipSign.
    categorical=[
        sp.AlignCategories(sort_by="value"),
        sp.ShuffleCategories(method="shift"),
    ],
    numerical=sp.Standardize(),
)
```

Each step runs a single estimator on a newly sampled context and query, and a fixed number of steps forms an *epoch*:

```python
import torch.nn.functional as F

context_size, query_size, steps_per_epoch = 256, 128, 30
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)

model.train()
for step in range(steps_per_epoch):
    index = torch.randperm(train_table.size(0), device="cuda")
    context = train_table[index[:context_size]]
    query = train_table[index[context_size:context_size + query_size]]

    out = model(
        x_context=context.drop_columns("target"),
        y_context=context["target"],
        x_query=query.drop_columns("target"),
        num_estimators=1,
        recipe=train_recipe,
    )[0]

    logits, y = sdm.evaluation.to_class_indices(
        pred=out,
        target=query["target"],
        missing_score=-torch.inf,
    )
    loss = F.cross_entropy(logits, y)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
```

The identity output keeps a leading estimator dimension, so `[0]` selects the single estimator.
For classification, the output has one logit column per class in the sampled context, which {py:func}`~sdm.evaluation.to_class_indices` aligns with the query targets.
`missing_score=-torch.inf` gives classes absent from the context zero probability, so a query row of an absent class gets infinite loss.
Make sure that every class appears in the context, or compute the loss on probabilities with a finite `missing_score`, as the benchmark harness does.

For regression, both models predict 999 quantiles (see [In-Context Learning](icl)), and the pinball (quantile) loss trains each output column toward its quantile level:

```python
diff = query["target"].numerical - out.numerical  # [Q, 999]
level = torch.linspace(0.001, 0.999, 999, device="cuda")
loss = torch.maximum(level * diff, (level - 1) * diff).mean()
```

## Choosing Hyperparameters

Validation scores decide both which weights to keep and which settings to train with.

### Keeping the Best Weights

Fine-tuning can overfit, especially on small datasets, so the example scores the model on the validation rows after every epoch and keeps the best weights, which picks the number of epochs automatically.
Evaluation matches inference: eval mode, the default recipe, and estimator ensembling.
For classification, maximize accuracy; for regression, average the predicted quantiles into a point prediction and minimize root mean squared error (RMSE):

```python
import copy

def evaluate(context: sdm.TableTensor, query: sdm.TableTensor) -> float:
    model.eval()
    out = model(
        x_context=context.drop_columns("target"),
        y_context=context["target"],
        x_query=query.drop_columns("target"),
        num_estimators=8,
    )
    if task == "classification":
        prob, y = sdm.evaluation.to_class_indices(
            pred=out,
            target=query["target"],
            missing_score=0.0,
        )
        return (prob.argmax(dim=-1) == y).float().mean().item()

    # Regression: average the predicted quantiles and compute RMSE.
    pred = out.numerical.mean(dim=-1)
    y = query["target"].numerical.squeeze(-1)
    return (pred - y).pow(2).mean().sqrt().item()

higher_is_better = task == "classification"
best_metric = evaluate(context=train_table, query=val_table)  # Pretrained model.
best_state = copy.deepcopy(model.state_dict())

max_epochs = 10
for epoch in range(max_epochs):
    model.train()
    ...  # Training steps from above.

    metric = evaluate(context=train_table, query=val_table)
    if metric > best_metric if higher_is_better else metric < best_metric:
        best_metric, best_state = metric, copy.deepcopy(model.state_dict())

model.load_state_dict(best_state)
test_metric = evaluate(context=train_table, query=test_table)
```

Since the first evaluation scores the pretrained model, selection never returns weights that score below it on the validation rows.

### Settings for Larger Datasets

The example's settings suit its dataset of about 1,000 training rows.
The benchmark harness in [`benchmark/tabular/finetune.py`](https://github.com/NVIDIA/structured-data-models/blob/main/benchmark/tabular/finetune.py) keeps the default recipe during training, and its defaults target larger datasets:

| Setting          | Example         | Harness                   |
| ---------------- | --------------- | ------------------------- |
| Learning rate    | `1e-5`          | `1e-6`                    |
| Context size *C* | 256             | up to 8,000               |
| Query size *Q*   | 128             | up to 2,000               |
| Epochs × steps   | 10 × 30         | 75 × 10                   |
| Validation rows  | 20% of all rows | 20% of the training split |

Starting from the harness defaults, we fine-tuned {py:class}`~sdm.models.TabICLv2` on two of the larger TALENT datasets and changed one setting at a time.
Airline (`customer_satisfaction_in_airline`) is a binary classification task with 83,123 training rows, and Diamonds is a regression task with 34,521 training rows.
The figure shows test accuracy and RMSE with the full training split as the inference context, averaged over 3 seeds; error bars show the standard deviation, and *zero-shot* marks the pretrained model:

```{figure} images/finetuning_sweep_light.svg
:figclass: light-only
:width: 100%
```

```{figure} images/finetuning_sweep_dark.svg
:figclass: dark-only
:width: 100%
```

The gains were small, since the pretrained model is already strong on both datasets.
Picked by validation score, the best settings improved test accuracy on Airline from 96.42% to 96.54% (learning rate `1e-5`) and test RMSE on Diamonds from 495.1 to 489.3 (*C* = 16,000).

- **Context size mattered most on Diamonds**, while on Airline, the three settings mattered about equally. *C* = 500 or 2,000 did not improve on the pretrained model, while *C* = 16,000, the largest size we tried, did best on both datasets, likely because larger training contexts are closer to inference, where the model reads the whole training split.
- **Learning rates between `1e-6` and `1e-5` worked best** within the range we tried (`1e-7` to `1e-5`); smaller rates helped less or not at all.
- **Query size had no consistent best value.** Airline accuracy rose steadily from *Q* = 128 to *Q* = 8,000, while on Diamonds, *Q* = 2,000 scored best.

For datasets of this size, start from the harness defaults, make *C* as large as memory allows, and tune the learning rate on a validation split.
With *Q* = 2,000 on Airline, fine-tuning peaked at 13 GB of GPU memory for *C* = 8,000 and 23 GB for *C* = 16,000, and each run took 46 to 60 minutes on an H100 GPU.

## Smaller Contexts After Fine-Tuning

A pretrained TFM knows your dataset only through its context, so more context rows typically improve accuracy but also raise inference cost.
Fine-tuning moves part of that knowledge into the weights, so a fine-tuned model can reach comparable accuracy with a smaller context, *e.g.*, a random subset of the training rows:

```python
model.eval()

index = torch.randperm(train_table.size(0), device="cuda")[:4096]
context = train_table[index]

model.fit(
    x=context.drop_columns("target"),
    y=context["target"],
    num_estimators=8,
)
out = model.predict(test_table.drop_columns("target"))
```

We compared the best fine-tuned model per dataset from the previous section with the pretrained model, giving both the same random subsets of training rows as context and averaging test scores over 3 seeds:

```{figure} images/finetuning_context_light.svg
:figclass: light-only
:width: 100%
```

```{figure} images/finetuning_context_dark.svg
:figclass: dark-only
:width: 100%
```

| Context rows | Airline, zero-shot | Airline, fine-tuned | Diamonds, zero-shot | Diamonds, fine-tuned |
| ------------ | ------------------ | ------------------- | ------------------- | -------------------- |
| 256          | 89.08%             | 89.88%              | 930.4               | 908.5                |
| 1,024        | 92.67%             | 93.82%              | 630.2               | 621.5                |
| 4,096        | 94.61%             | 95.62%              | 548.3               | 528.3                |
| 16,384       | 95.87%             | 96.30%              | 505.0               | 496.0                |
| All          | 96.42%             | 96.57%              | 495.1               | 488.8                |

- **The fine-tuned model scored better at every context size**, but at the smallest contexts (256 rows on both datasets and 1,024 rows on Diamonds), the difference was within the variation across seeds.
- **With 16,384 rows, the fine-tuned model came close to the pretrained model with the full training split**, using only 20% (Airline) and 47% (Diamonds) of the training split.
- **More context still helped.** Both models improved with every increase in context size.

The full-context results differ slightly from the previous section because estimator ensembles are randomized at inference.

## Benchmark Results

Beyond these two datasets, the benchmark harness runs this procedure on [TALENT](https://github.com/LAMDA-Tabular/TALENT), [TabArena](https://github.com/autogluon/tabarena), and [ScoringBench](https://scoringbench.com).
Wins, losses, and ties count the datasets on which fine-tuning improved, worsened, or matched the pretrained model's score on each benchmark's own metric:

| Benchmark    | Model                                       | Tasks                     | Wins | Losses | Ties |
| ------------ | ------------------------------------------- | ------------------------- | ---- | ------ | ---- |
| TALENT       | {py:class}`~sdm.models.TabICLv2`            | Classification            | 73   | 51     | 31   |
| TALENT       | {py:class}`~sdm.models.KumoTabular` (small) | Classification            | 51   | 39     | 38   |
| ScoringBench | {py:class}`~sdm.models.TabICLv2`            | Probabilistic regression  | 50   | 26     | 10   |
| TabArena     | {py:class}`~sdm.models.KumoTabular` (large) | Classification/regression | 35   | 12     | 0    |

On TALENT, fine-tuning {py:class}`~sdm.models.TabICLv2` won twice as often as it lost on multiclass datasets (36 *vs.* 18), but only slightly more often on binary ones (37 *vs.* 33).
{py:class}`~sdm.models.KumoTabular` covers fewer TALENT datasets because its fine-tuning variant supports at most 10 classes.

To reproduce these results, add `--finetune` to the TALENT and ScoringBench runners (with `--model kumo-tabular-small-ft` for {py:class}`~sdm.models.KumoTabular` on TALENT), or pick a fine-tuned `-ft` model on TabArena:

```bash
python -m benchmark.tabular.talent.main \
  --model tabiclv2 \
  --dataset-path /path/to/talent/data \
  --finetune

python -m benchmark.tabular.tabarena.main --model kumo-tabular-large-ft
```

The TALENT and ScoringBench runners expose the learning rate, *C*, and *Q* as `--finetune-lr`, `--finetune-train-size` (*C* + *Q*), and `--finetune-context-frac` (*C* / (*C* + *Q*)).
See the [tabular benchmark instructions](https://github.com/NVIDIA/structured-data-models/tree/main/benchmark/tabular) for setup and all fine-tuning flags.
