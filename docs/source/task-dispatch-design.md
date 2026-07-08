# TaskDispatch design

Status: implemented by this change.

## Goal

Add a task-aware output processor that selects a deterministic processing
route from the semantic type produced by the complete target pipeline:

- one numerical target column means regression;
- one categorical target column means classification;
- empty, multi-column, datetime, ID, or otherwise ambiguous targets raise.

`TaskDispatch` initially belongs only in `Recipe.output`. Target processing can
already use `StypeDispatch`, and the transformed target is the source of truth
for the task. The design must retain the `TableTensor -> TableTensor`
`Processor` contract and must not make users pass task metadata to every
processor call.

## Recommended API

```python
recipe = Recipe(
    features=[
        MeanImpute(),
        StandardScale(epsilon=1e-6),
    ],
    target=[Identity()],
    output=[
        TaskDispatch(
            classification=SoftmaxTemperature(temperature=0.9),
            regression=Identity(),
        ),
    ],
)

features, target = recipe.fit_transform(features, target)
prediction = model(features, target)
prediction = recipe.target.inverse_transform(prediction)
output = recipe.output.transform(prediction)
```

An invertible `StypeDispatch` could later apply different target transforms;
that inverse behavior is tracked in #202.

`Recipe.fit_transform(features, target)` is the task-resolution boundary. It:

1. fits and transforms the target first;
2. validates and infers its task from the final target `TableTensor`;
3. resolves every `TaskDispatch` in the output role;
4. fits and transforms the features;
5. returns transformed features and target in that order.

Calling it again refits the role processors and replaces the resolved task.
If refitting fails, output dispatchers remain unresolved instead of retaining
the route selected by an earlier fit.
`TaskDispatch.transform` before resolution raises an actionable error that
points to `Recipe.fit_transform`.

Existing direct role calls remain valid for recipes without task dispatch.
Task-aware recipes must use the recipe-level entry point because target output
and model output are different processor inputs.

## Routes

```python
TaskDispatch(
    *,
    classification: Processor | Iterable[Processor] | None = None,
    regression: Processor | Iterable[Processor] | None = None,
)
```

- At least one route is required.
- An iterable is normalized to `Sequential`, matching `StypeDispatch`.
- Initial routes must be stateless. Fitting an output processor would require
  predictions and potentially validation labels, which this lifecycle does
  not provide.
- A selected missing route raises instead of silently falling back.
- A supported no-op route uses `Identity()` explicitly.
- `TaskDispatch` has no inverse API.
- The selected task is shown by `repr` and replaced when the recipe is refit.

A model with one fixed output path does not need `TaskDispatch`. A
regression-only model may define only the regression route, so a categorical
target fails clearly. A model that deliberately treats every target as
regression converts categorical targets to numerical in `Recipe.target`; the
final numerical stype then resolves regression.

## Placement

The initial implementation supports `TaskDispatch` only in `Recipe.output`.
`Recipe` rejects instances in feature or target processor trees with an
actionable error. A direct check in `Recipe` is preferable to introducing a
general processor-role declaration before another processor needs it.

This restriction is intentionally extensible. TabPFN uses task-dependent
feature preprocessing: its official inference configuration describes
different classifier and regressor quantile, encoding, power-transform, and
outlier behavior. Processing the target first means a future design can lift
the placement restriction and resolve feature dispatch before fitting the
feature pipeline.

Evidence:
[`PriorLabs/TabPFN/src/tabpfn/inference_config.py`](https://github.com/PriorLabs/TabPFN/blob/3cdc664fdc595ce3f37830123585566550a00b4b/src/tabpfn/inference_config.py).

## Alternatives evaluated

| Design                                                | Result                                                                                                                                                                                                                  |
| ----------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Recipe-level `fit_transform(features, target)`        | **Recommended.** The cross-role operation is explicit at the object that owns all roles and naturally supports target-first ordering.                                                                                   |
| Hidden wrapper around `recipe.target`                 | Allows direct `recipe.target.fit_transform(target)`, but silently mutates a sibling pipeline through callbacks or aliases. It has no measured speed advantage and complicates deepcopy, replacement, and serialization. |
| Explicit target `ResolveTask` plus shared `TaskState` | Direct and slightly faster, but adds two concepts, shared mutable state, and a target step whose only purpose is configuring output.                                                                                    |
| Pass target/task to `TaskDispatch.transform`          | Rejected because it changes the common one-`TableTensor` processor signature and prevents ordinary `Sequential` composition.                                                                                            |

The hidden wrapper is especially unfriendly to code-reading agents: the local
target call appears pure but changes later output behavior. The recipe method
makes the side effect, ordering, and required action discoverable from one
public API.

## Prototype evidence

In-memory prototypes against the current processor API verified:

- numerical target resolves regression;
- categorical target resolves classification;
- categorical-to-numerical target conversion deliberately resolves
  regression;
- a missing selected route and an unresolved dispatch raise;
- repeated fitting changes the selected task;
- deepcopy and standalone `state_dict` retain the selected task.

Repeated timing medians on PyTorch 2.12.1 with two CPU threads and an NVIDIA
L4 were:

| Operation                                    |  Median |
| -------------------------------------------- | ------: |
| Recipe-level target transform and resolution | 4.19 us |
| Hidden target wrapper                        | 4.21 us |
| Explicit resolver step                       | 3.04 us |

The approximately 1 us resolver difference does not justify its additional
public concepts. Steady-state dispatch added about 1.5 us around `Identity`,
3.7-7.9 us around CPU softmax, and 4.2-7.2 us around CUDA softmax. Dispatch is
not the material cost in a real output pipeline.

## Persistence

The resolved task belongs to `TaskDispatch` state and survives `state_dict`
through PyTorch extra state. `Sequential` retains its immutable public `steps`
tuple and also registers those processors as child modules. Consequently,
ordinary `modules()`, `state_dict()`, and device movement include nested
processors, and `Recipe` can use ordinary module traversal to validate
placement and resolve nested output dispatchers.

Recipe-level serialization is not currently a public contract; the
persistence guarantee applies to `TaskDispatch` and its registered owning
processor tree.

## Non-goals

- Inferring task from model output. Classification logits and regression
  predictions are both numerical.
- Routing target processors. `StypeDispatch` already handles the target's
  semantic type before task inference.
- Stateful output calibration.
- Multitask or multi-output targets.
- Task-dependent features in the first implementation.
- Inverse transformation through `TaskDispatch`.

## Implementation scope

The change registers `Sequential` steps as child modules, adds target-first
`Recipe.fit_transform`, and implements stateless named `TaskDispatch` routes
with strict one-column inference, output-only placement, actionable errors,
representation, and persisted task state. Focused tests cover both task routes,
refitting, persistence, invalid usage, and CPU/CUDA execution.

Related work:

- #208 provides the closest processor-delegation precedent through `Choice`.
- #200 introduced stype-based routing and iterable normalization.
- #172 adds TabICLv2 multiclass behavior needed for an end-to-end
  classification recipe.
- #215 defines the non-finite-value behavior expected from processor routes.
