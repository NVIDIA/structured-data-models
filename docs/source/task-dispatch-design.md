# TaskDispatch design

Status: proposed revision. The current runtime implementation still uses
recipe-level fitting and must be aligned separately.

## Goal

Add a task-aware processor that selects a deterministic output route from the
semantic type produced by the complete target pipeline:

- one numerical target column means regression;
- one categorical target column means classification;
- empty, multi-column, datetime, ID, or otherwise ambiguous targets raise.

The transformed target remains the source of truth. Every public processor
method keeps the `TableTensor -> TableTensor` contract, and users do not pass a
task to each output transformation.

The revised goal is also to preserve the existing role-level processor API.
Users fit and transform `recipe.target` and `recipe.features` directly instead
of learning additional recipe-level preprocessing and postprocessing methods.

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

target = recipe.target.fit_transform(target)
features = recipe.features.fit_transform(features)

prediction = model(features, target)
prediction = recipe.target.inverse_transform(prediction)
prediction = recipe.output.transform(prediction)
```

Fit-only workflows remain ordinary processor calls:

```python
recipe.target.fit(target)
recipe.features.fit(features)
```

Validation or query features use the fitted feature processor directly:

```python
features = recipe.features.transform(features)
```

There is no `Recipe.fit`, `Recipe.fit_transform`, `Recipe.preprocess`, or
`Recipe.postprocess` lifecycle in this design. `Recipe` remains an inspectable
container for the three role processors.

## Target wrapper

For a task-aware recipe, `Recipe` wraps the complete normalized target
processor in an internal processor, called `_TaskResolver` here. The wrapper
is returned as `recipe.target`, so the normal processor calls are also the task
resolution boundary.

Conceptually, recipe construction performs:

```python
target = Sequential(*target)
output = Sequential(*output)

dispatchers = tuple(
    module
    for module in output.modules()
    if isinstance(module, TaskDispatch)
)

if dispatchers:
    target = _TaskResolver(
        processor=target,
        dispatchers=dispatchers,
    )
```

The wrapper delegates target transformation and inverse transformation to the
wrapped processor. During fitting it uses the final transformed target to
resolve every associated output dispatcher.

```python
class _TaskResolver(Processor, InvertibleMixin):
    def fit(self, input: TableTensor) -> Self:
        self._fit_transform_and_resolve(input)
        return self

    def fit_transform(self, input: TableTensor) -> TableTensor:
        return self._fit_transform_and_resolve(input)

    def _fit_transform_and_resolve(
        self,
        input: TableTensor,
    ) -> TableTensor:
        for dispatcher in self._dispatchers:
            dispatcher._reset()

        succeeded = False
        try:
            output = self.processor.fit_transform(input)
            for dispatcher in self._dispatchers:
                dispatcher._resolve(output)
            self._fitted = True
            succeeded = True
            return output
        finally:
            if not succeeded:
                for dispatcher in self._dispatchers:
                    dispatcher._reset()

    def _transform(self, input: TableTensor) -> TableTensor:
        return self.processor.transform(input)

    def _inverse_transform(self, input: TableTensor) -> TableTensor:
        return self.processor.inverse_transform(input)
```

This is pseudocode. The implementation must retain the usual supported-stype
and fitted-state checks.

The wrapper always requires one fit before output dispatch, even when the
wrapped target processor is stateless, because task resolution itself is
fitted state.

## Why the wrapper is coherent

The cross-role relationship cannot be removed: the output route depends on
the result of the target pipeline. The wrapper places that relationship at the
operation that produces the required information:

```text
recipe.target.fit(target)
-> complete target transformation
-> task inference
-> output dispatchers become usable
```

This retains the established processor vocabulary. A reader does not need to
know whether the enclosing recipe has a second fitting API, and a model can fit
feature and target roles independently.

The tradeoff is that fitting `recipe.target` changes later output behavior.
That side effect must be made discoverable through:

- the `_TaskResolver` representation around the target pipeline;
- `TaskDispatch` documentation that points to `recipe.target.fit()`;
- an actionable unresolved error;
- focused tests for refitting, failure, copying, and persistence.

The wrapper must not use callbacks or search for sibling processors during
each fit. `Recipe.__init__` discovers the output dispatchers once and wires the
immutable relationship.

## Dispatcher ownership and module registration

`TaskDispatch` remains registered only under `recipe.output`. The target
wrapper keeps non-owning Python references to those same dispatcher objects;
it must not register them as target child modules. Registering the dispatchers
under both roles would make traversal, placement validation, state dict keys,
and representation misleading.

The wrapped target processor is a normal registered child of `_TaskResolver`.
`Sequential` continues to register its steps, so nested output dispatchers are
found once during recipe construction and remain visible to ordinary PyTorch
module traversal.

The selected task continues to belong to each `TaskDispatch` and persists via
its PyTorch extra state. The wrapper coordinates fitting but does not introduce
a second persisted `TaskState` concept.

## Copying one recipe per estimator

An ensemble may use one independently fitted recipe copy per estimator:

```python
member_recipe = deepcopy(recipe)
member_target = member_recipe.target.fit_transform(target)
member_features = member_recipe.features.fit_transform(features)
```

The whole `Recipe` must be copied, not its roles independently. Python's
`deepcopy` memo preserves the alias between the wrapper's dispatcher references
and the dispatchers registered in the copied output tree. Copying target and
output separately would break that relationship. An in-memory prototype against
the current processor classes verified that whole-container `deepcopy` preserves
the alias and that two copies resolve independently without mutating the
template.

This requirement needs a focused test that proves:

- copied recipes resolve independently;
- the copied target wrapper controls only the copied output dispatchers;
- refitting one member does not change another member or the template;
- output `state_dict` round-tripping retains each selected route.

If models later make fitted recipes part of model state, their target, feature,
and output processors must be registered in `ModuleList` or `ModuleDict`
containers so `.to(device)` and model checkpoints include them.

## Fitting order and failure behavior

The recommended order is target first, then features:

```python
recipe.target.fit(target)
recipe.features.fit(features)
```

The initial implementation allows `TaskDispatch` only in `Recipe.output`, so
feature fitting is not task-dependent yet. Target-first ordering leaves a
straightforward extension for task-dependent feature routes without adding a
new recipe lifecycle.

Every target refit resets associated dispatchers before doing work. If target
fitting, transformation, task inference, or route validation fails, all
associated dispatchers remain unresolved. A failed refit must never leave a
route selected by an earlier successful fit.

Calling `TaskDispatch.transform` before target fitting raises an error such as:

```text
'TaskDispatch' has no resolved task; call 'recipe.target.fit()' first.
```

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
- Initial routes must be stateless. Fitting output calibration would require
  predictions and potentially validation labels, which target fitting does not
  provide.
- A selected missing route raises during target fitting.
- A supported no-op route uses `Identity()` explicitly.
- `TaskDispatch` has no inverse API.
- The selected task is shown by `repr` and replaced when the target is refit.

A fixed-task model does not need `TaskDispatch`. A regression-only model may
define only the regression route, so fitting a categorical target fails
clearly. A model that deliberately treats every case as regression converts
its target to numerical before the wrapper infers the task.

## Placement

The first implementation supports `TaskDispatch` only in `Recipe.output`.
Instances in feature or target processor trees raise during recipe
construction. This keeps the initial change narrow and prevents recursive or
self-dependent task resolution.

Task-dependent feature processing is a plausible later use. Once required,
`Recipe.__init__` can wire feature dispatchers to the same target wrapper and
document that target fitting must precede feature fitting. No processor method
signature needs to change.

## Alternatives evaluated

| Design                                                                     | Result                                                                                                                                              |
| -------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------- |
| Target wrapper wired by `Recipe`                                           | **Recommended.** Preserves direct role-level processor calls and moves existing resolution behavior to the target operation that produces the task. |
| Recipe-level `fit(features, target)` and `fit_transform(features, target)` | Explicitly owns cross-role ordering, but duplicates processor lifecycle methods and requires users and models to learn a second fitting API.        |
| Public shared `TaskState` passed to target and dispatchers                 | Makes sharing explicit but adds a user-visible coordination object with no independent domain behavior.                                             |
| Pass target or task to `TaskDispatch.transform`                            | Changes the common one-input processor signature and prevents ordinary `Sequential` composition.                                                    |
| Infer task from model output                                               | Invalid because both classification logits and regression predictions are numerical.                                                                |

The target wrapper was rejected in the original design because it mutates a
sibling role. This revision accepts that localized tradeoff to retain the
ordinary processor API. It avoids the most problematic version of the wrapper:
there are no runtime callbacks, no dynamic sibling lookup, and no second shared
state object. The relationship is constructed once, represented in the target
tree, and tested as part of whole-recipe copying.

## Evidence

Earlier prototypes measured no meaningful performance difference between a
recipe-level resolver and a target wrapper:

| Operation                                    |  Median |
| -------------------------------------------- | ------: |
| Recipe-level target transform and resolution | 4.19 us |
| Target wrapper                               | 4.21 us |
| Explicit resolver step                       | 3.04 us |

The approximately one-microsecond difference does not decide the API. The
wrapper is selected for lifecycle consistency, not speed.

Existing prototypes and tests already establish the core task behavior:

- numerical targets resolve regression;
- categorical targets resolve classification;
- categorical-to-numerical target conversion resolves regression;
- missing and unresolved routes raise;
- refitting may change the selected task;
- the selected task persists in `TaskDispatch` extra state;
- registered nested processors participate in traversal and device movement.

The revised implementation additionally needs whole-recipe `deepcopy` tests
because the wrapper introduces a deliberate alias from target coordination to
output dispatchers.

## Non-goals

- Recipe-level preprocessing or postprocessing convenience methods.
- Inferring task from model output.
- Stateful output calibration.
- Multitask or multi-output targets.
- Task-dependent features in the first implementation.
- Inverse transformation through `TaskDispatch`.
- General ensemble aggregation or model-output decoding.

## Implementation consequences

Compared with the current PR implementation, the revised design requires:

1. Add the internal target wrapper and wire it during `Recipe` construction.
2. Move dispatcher reset and resolution from `Recipe.fit_transform` into the
   target wrapper's `fit` and `fit_transform` paths.
3. Remove recipe-level `fit`, `fit_transform`, and `preprocess` methods.
4. Update unresolved errors to point to `recipe.target.fit()`.
5. Keep `Sequential` child-module registration and `TaskDispatch` extra state.
6. Rewrite focused tests around direct target and feature processor calls.
7. Add whole-recipe copying tests before using one recipe per estimator.

Related work:

- #200 provides stype-based routing and iterable normalization.
- #202 defines inverse routing required by richer target pipelines.
- #208 provides the closest processor-delegation precedent through `Choice`.
- #215 defines non-finite-value behavior expected from processor routes.
