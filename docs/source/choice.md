# Choice Processor Design

`Choice` is a processing container that selects one processor from a finite
set for each resolved recipe. Its main use is estimator diversity: every
ensemble member receives an independent, unfitted processing graph with one
stable normalization policy.

The target recipe should eventually support composition such as:

```python
from sdm.processing import (
    Choice,
    ConstantFilter,
    FeaturePermute,
    Identity,
    MeanImpute,
    Quantile,
    Recipe,
    SigmaClip,
    StandardScale,
    StypeDispatch,
    ToNumerical,
)

recipe = Recipe(
    features=[
        StypeDispatch({"categorical": ToNumerical()}),
        MeanImpute(),
        ConstantFilter(),
        StandardScale(epsilon=1e-6),
        Choice(
            [
                Identity(),
                Quantile(output_distribution="normal"),
            ]
        ),
        SigmaClip(threshold=4.0),
        FeaturePermute(method="latin"),
    ],
)
```

This note proposes the public behavior and lifecycle. The first change should
land the design only; implementation can follow in the same draft PR after the
API is agreed.

## Related Work

The proposal follows the current processing architecture and the related open
PRs as of July 2026:

- [#200](https://github.com/NVIDIA/structured-data-models/pull/200) adds
  `StypeDispatch`, another processor container. It establishes that containers
  should compose through `Processor`, register children as PyTorch modules,
  and preserve the table-in/table-out contract.
- [#202](https://github.com/NVIDIA/structured-data-models/pull/202) designs
  `StypeDispatch` inverse routing. It provides evidence that concrete routes
  need explicit ownership and schema metadata, although it does not define a
  recipe-resolution phase.
- [#149](https://github.com/NVIDIA/structured-data-models/pull/149) proposes
  `Processor.resolve(estimator=..., generator=...)` and uses it to materialize
  estimator-specific `FeaturePermute` policies. Its default implementation
  returns `self` and its containers do not resolve recursively, so independent
  recipe cloning requires a coordinated revision rather than direct reuse.
- [#150](https://github.com/NVIDIA/structured-data-models/pull/150) applies the
  same initial resolution contract to `LabelShuffle` and adds
  `ConstantFilter`.
- [#109](https://github.com/NVIDIA/structured-data-models/pull/109) keeps
  `Quantile` fitting state in registered buffers and makes its transform path
  friendly to `torch.compile`. A choice must therefore be fixed outside the
  transform path, before fitting and compilation.

PRs #193, #194, #197, #199, and #155 have since landed on `main`. They provide
the `TableTensor` helpers, table-level processor lifecycle, supported-stype
validation, and `ToNumerical` behavior used by the example.

The inference recipe in #179 was inspected as well. It concerns model runtime
precision and compilation rather than processing-route selection, so it does
not add a competing `Choice` contract.

## Requirements

- Select exactly one option for each concrete recipe.
- Keep the selection stable across `fit`, `transform`, and inverse operations.
- Fit only the selected option. Unselected alternatives must not observe
  training data, allocate fitted state, move devices, or appear in the
  concrete module graph.
- Produce independent processor instances for different estimators so fitting
  one recipe cannot mutate another.
- Make random selection reproducible with a `torch.Generator`.
- Preserve direct Python composition and avoid a mandatory config system.
- Keep stochastic orchestration outside tensor transforms and compiled model
  execution.
- Leave a compatible path to deterministic policies such as round-robin and
  to a richer resolution context.

## Evidence

### Recipe copies currently share fitted state

`Recipe` contains mutable `torch.nn.Module` processor graphs. A local check on
current `main` shows:

```text
copy.copy(recipe).features is recipe.features                   True
copy.copy(recipe).features.steps[0] is recipe.features.steps[0] True
```

Consequently, this is unsafe today:

```python
recipes = [copy.copy(recipe) for _ in range(num_estimators)]
```

Fitting one shallow copy fits the same `Quantile`, `StandardScale`, or other
stateful module referenced by every copy. This follows
[Python's documented shallow-copy semantics](https://docs.python.org/3/library/copy.html):
a new compound object contains references to its nested objects.

Python permits custom `__copy__` behavior, but making `copy.copy(recipe)` both
deep-clone a module graph and consume randomness would be surprising. Copying
an object should not silently select a model policy.

### Fitted and template state need different operations

An ensemble member needs the constructor configuration of every processor but
not the fitted buffers of another member. This is closer to an estimator
*clone* than to a general deep copy. Scikit-learn's
[`clone`](https://scikit-learn.org/stable/modules/generated/sklearn.base.clone.html)
provides a useful precedent: it creates a new unfitted estimator with the same
parameters rather than copying learned data.

The existing `_fitted` flag is not a valid template-safety check. PR #150 uses
`_fitted = True` for `LabelShuffle(n_classes=...)` because constructor state is
already complete, even though no data has been observed. The lifecycle needs a
separate recursive `_fit_called` signal. It must be set before invoking any
data-consuming fit logic, including container overrides of `fit` and
`fit_transform`. If fitting mutates state and then raises, the source remains
marked as consumed and cannot later be resolved as a pristine template.

Resolution validates the complete source graph before consuming RNG, then
deep-copies it once and resolves policies on that independent copy. This keeps
constructor-ready state while rejecting templates that have observed data. A
future constructor-based `clone_unfitted` protocol can replace deep copy if
processor state becomes too broad.

### Selection belongs before fit

`Quantile` learns buffers during `fit`; `Identity` is stateless. Selecting on
every `transform` could therefore apply a processor that was never fitted.
Fitting every option would waste work and expose unused routes to training
data. Selecting during the first `fit` also leaves stateless pipelines with no
natural resolution point and makes configuration depend on data lifecycle.

The open `FeaturePermute` and `LabelShuffle` PRs already establish an explicit
resolution phase. `Choice` should use that phase rather than create another
lifecycle.

### Randomness should remain explicit

[`torch.Generator`](https://docs.pytorch.org/docs/stable/generated/torch.Generator.html)
carries explicit RNG state and can be passed to random tensor operations.
Resolving with a caller-owned generator makes repeated ensemble construction
reproducible without changing global RNG state. Sampling is orchestration over
a handful of options, so using a scalar choice during resolution does not
affect the tensor-centric model execution path.

## Considered Designs

### 1. Select lazily during `fit`

`Choice.fit` would sample an option and fit it. The selection would then be
reused by `transform`.

This is superficially simple, but it couples configuration to training data
lifecycle. A choice containing only stateless options would not otherwise need
`fit`, and `transform` before `fit` would have unclear behavior. It also does
not solve independent recipe cloning. This option is rejected.

### 2. Select as a side effect of `copy.copy`

`Recipe.__copy__` could recursively clone all processors and ask every
`Choice` to sample.

This makes the suggested list comprehension work, but gives shallow copy
non-standard deep and stochastic semantics. It also obscures where RNG state
is consumed and makes accidental copies alter ensemble composition. This
option is rejected.

### 3. Fit every option and select afterward

`Choice` could fit all alternatives, then select one for transformation.

This has a simple fitted-state story, but it performs unnecessary fitting,
stores unused buffers, and lets alternatives learn from data even though they
are absent from the concrete recipe. It scales poorly for expensive routes and
weakens recipe auditability. This option is rejected.

### 4. Resolve an unfitted template into concrete recipes

An unresolved recipe is a reusable policy template. `resolve` clones its
processor tree, selects each `Choice`, and returns a concrete unfitted recipe.
Only the concrete recipe may be fitted.

This matches #149 and #150, gives randomness one explicit boundary, and lets
the same mechanism resolve feature permutation and label shuffling. It is the
recommended design.

## Proposed API

The first implementation should expose a random choice template:

```python
Choice(options: Sequence[Processor])
```

`options` must be non-empty. Each option is a `Processor`; a multi-step
option can be expressed explicitly with `Sequential`. The unresolved choice
registers its alternatives in a `torch.nn.ModuleList`, but it is a policy
template and cannot process data directly.

Recipe resolution is the public ensemble boundary:

```python
recipe.resolve(
    *,
    estimator: int = 0,
    generator: torch.Generator | None = None,
) -> Recipe
```

This signature aligns with the direction in #149 and #150, but the ownership
contract is stricter:

1. `Recipe.resolve` validates the complete source graph before consuming RNG.
2. It deep-copies the three role graphs once so every estimator owns independent
   processor instances.
3. A private recipe-owned visitor recursively resolves copied `Sequential`,
   `StypeDispatch`, and processor nodes.
4. The visitor handles `Choice` before generic processor dispatch and replaces
   it with the selected processor.
5. Direct `Choice.resolve()` raises an actionable error. Its inherited
   `Processor.resolve() -> Self` contract cannot return an arbitrary option
   type safely.
6. Plain processors may keep the #149 behavior of returning `self` from
   `Processor.resolve`, because that `self` is already part of the private
   copy.
7. Containers are rebuilt after resolution so derived properties such as
   `requires_fit` reflect the concrete children.

This requires a coordinated update to #149 and #150. It is not compatible with
calling their current default `Processor.resolve` directly on a shared graph
and expecting an independent estimator.

The coordinated revision also changes estimator-view RNG ownership.
`FeaturePermute.resolve` and `LabelShuffle.resolve` must snapshot immutable
seed state rather than retain the caller's mutable generator. Reseeding the
original generator after resolution must not change a concrete recipe.
Shape-dependent random permutations are materialized during fit, after schema
or class width is known and before compiled transforms execute.

Random ensemble construction becomes:

```python
generator = torch.Generator().manual_seed(123)
recipes = [
    recipe.resolve(estimator=i, generator=generator)
    for i in range(num_estimators)
]

for member in recipes:
    member.features.fit(features)
```

Random selection is uniform over option indices and consumes one draw from a
CPU generator per visited `Choice`. Before drawing, resolution validates every
alternative without consuming RNG. After drawing, it recursively resolves only
the selected branch, so dead alternatives cannot affect later random choices.

The same seed, template, estimator order, and traversal order reproduce the
same concrete recipes. Without an explicit generator, resolution uses the
global CPU generator. CUDA generators are rejected in the first implementation
because this is CPU orchestration and no model data participates in selection.

The current `Recipe` API exposes role processors rather than a top-level
`Recipe.fit`. The example therefore calls `member.features.fit(...)`. A
top-level fitting convenience is separate because it must define how feature
and target inputs are supplied.

## Choice Lifecycle

An unresolved `Choice` cannot be fitted or transformed. Calling a data method
before resolution raises an actionable error.

Resolution does not create a runtime wrapper. It replaces `Choice` with an
independent copy of the selected processor. Consequently:

- only the selected processor appears in the concrete module graph;
- unselected options are not fitted, moved between devices, checkpointed, or
  exposed to an optimizer;
- `fit`, `fit_transform`, `transform`, and `inverse_transform` use the
  selected processor's existing implementation;
- `repr` shows the concrete processor rather than an unresolved policy;
- the transform path contains no random operation, selected-index lookup, or
  tensor-to-Python conversion.

This structural replacement is simpler and more compile-friendly than retaining
all alternatives in a resolved `ModuleList`.

An unresolved `Choice` does not claim a static `supported_stypes`
intersection. Container options can hide stricter children, for example
`Sequential(StandardScale())`, even when the container itself advertises all
stypes. The selected concrete processor performs normal supported-stype
validation after resolution.

Target recipes need structural inverse validation before random selection.
Checking only `InvertibleMixin` is insufficient because `Sequential` currently
inherits it even when an inner step is non-invertible. The implementation should
add a recursive `supports_inverse` capability for containers and require all
`Choice` alternatives in the target role to support inverse. If that capability
does not land with the first implementation, `Choice` must initially be
rejected in `Recipe.target` rather than fail after model execution.

## Resolution And Cloning

The template and concrete recipe are separate objects:

```text
template Recipe
    -> resolve(estimator=0, generator=g)
    -> concrete, independent, unfitted Recipe
    -> member.features.fit(...)
    -> fitted estimator-0 Recipe
```

The normative algorithm is:

1. Walk every processor reachable from the template, including alternatives
   that may not be selected.
2. Reject the source if any processor has previously consumed data. A dedicated
   `_fit_called` signal distinguishes this from constructor-ready state such as
   `LabelShuffle(n_classes=...)`.
3. Validate container structure, target invertibility, options, and generator
   device before consuming RNG.
4. Deep-copy each role graph once.
5. Traverse the copy in stable feature, target, output, then child order.
6. At each `Choice`, draw once and recurse only into the selected option.
7. Rebuild containers with registered concrete children and recompute derived
   lifecycle properties.

`Sequential.steps` is currently a plain tuple, so its children are not
registered PyTorch modules. Recursive resolution should change it to
`torch.nn.ModuleList`. `StypeDispatch` should retain its `ModuleDict`
registration while resolving each route. `Recipe` can remain a dataclass
because each role is itself a registered processor graph.

The initial implementation should test that:

- resolved processors do not share identity, parameters, or buffers;
- fitting one resolved recipe leaves the template and peers untouched;
- constructor-ready processors remain resolvable before observing data;
- resolving any graph that has consumed fit data raises before RNG advances;
- a processor that mutates state and then fails fitting remains consumed;
- a partially failed container pipeline remains consumed recursively;
- nested choices in unselected branches consume no RNG;
- reseeding a caller-owned generator cannot alter a resolved recipe;
- shape-dependent random policies contain no RNG in compiled transforms;
- nested `Sequential` and `StypeDispatch` containers resolve recursively;
- target alternatives are validated before sampling;
- a resolved identity choice and a fitted quantile choice both compile without
  graph breaks.

`copy.copy(recipe)` remains a normal shallow copy and is not an ensemble API.
Documentation should point users to `recipe.resolve(...)`.

## Checkpoint Scope

A fitted-recipe checkpoint format is not part of the first `Choice`
implementation. Current `Recipe` is not an `nn.Module`, `Sequential` does
not register its tuple children, `_fitted` is not persistent, and processors
such as `Quantile` replace empty buffers with fitted buffers of different
shapes. A `Choice` state-dict promise would therefore overstate the current
architecture.

Registering concrete container children is required now for normal PyTorch
module behavior. Reconstructing a fitted recipe from a fresh graph and a state
dict needs a separate design covering constructor configuration, fitted-state
persistence, and dynamic buffers. Full-object serialization continues to follow
the repository's existing behavior; this proposal adds no stronger guarantee.

## Future Resolution Context

The initial keyword API is sufficient for random choices and compatible with
the estimator-view work already in progress. The implementation may use a
private context internally to carry role and traversal path:

```python
_ResolveContext(
    estimator=i,
    generator=generator,
    role="features",
    path=("features", 4),
)
```

Role is needed for target inverse validation, and path gives later policies a
stable identity. Once a second public need exists, this can become:

```python
context = ResolveContext(
    estimator=i,
    generator=generator,
)
member = recipe.resolve(context=context)
```

Round-robin is intentionally deferred until that context exists:

```python
Choice(
    [Quantile(output_distribution="normal"), Identity()],
    sampling="round-robin",
)
```

A naive `estimator % len(options)` makes every same-sized choice select the
same ordinal for an estimator. The context design must explicitly choose
whether that lockstep behavior is desired or derive a stable per-choice offset
from `path`. Deferring the public option avoids freezing correlated behavior
accidentally. The first implementation supports random sampling only.

The context can later support deterministic child RNG streams, sampling without
replacement, distributed rank information, and additional sampling policies
without adding unrelated keyword arguments to every processor.

## Error Handling

- Empty `options` raises `ValueError` at construction.
- A non-`Processor` option raises `TypeError` at construction.
- Data methods on an unresolved choice raise `RuntimeError`.
- Direct `Choice.resolve()` raises and points callers to `Recipe.resolve()`.
- A negative estimator index raises `ValueError` during resolution.
- A non-CPU generator raises `ValueError` before sampling.
- A source graph that has consumed fit data raises `RuntimeError` before RNG
  advances.
- A target choice with an option that lacks structural inverse support raises
  during resolution before sampling.

## `torch.compile` And Device Behavior

Sampling and Python branching occur only during recipe resolution. Because the
resolved graph contains the selected processor directly, transform calls do not
index a `ModuleList`, call `.item()`, or branch on selected-index tensors.

Resolution is device-independent and happens before fitting. The concrete
processor follows normal `torch.nn.Module.to(...)` behavior. No table or model
tensor is moved to CPU for selection.

## Shipped First Version

This draft ships an interim `Choice` ahead of the recipe-resolution phase.
The shipped processor selects one option uniformly at random **at
construction** and keeps only the selected option as a registered submodule.
Unselected options are discarded immediately, so they are never fitted,
moved between devices, or checkpointed:

```python
def make_member() -> Recipe:
    return Recipe(
        features=[
            MeanImpute(),
            Choice([Identity(), Quantile(output_distribution="normal")]),
            StandardScale(epsilon=1e-6),
        ],
    )

torch.manual_seed(123)
members = [make_member() for _ in range(8)]
```

The recipe factory is the template: constructing the recipe again draws a
new selection, so ensemble members are independent by construction and no
copying or resolution step is required. Selection draws from the global
CPU generator and `torch.manual_seed` makes it reproducible. Copying a
built recipe (for example with `copy.deepcopy`) duplicates the
already-made selection; copies are for reuse, not for diversity.

The construction-time draw is interim behavior, and two extensions are
planned. First, a keyword-only `generator` argument can be added without
breaking callers, isolating selection from unrelated global RNG use.
Second, once recipe resolution lands, `Choice` becomes the unresolved
template described above: construction stops consuming RNG and
`Recipe.resolve` performs the draw instead.

## Implementation Plan

1. Add recursive fit-consumption tracking and change `Sequential.steps` to a
   registered `ModuleList`.
2. Reconcile #149 and #150 around one recipe-owned clone boundary, recursive
   container resolution, immutable seed snapshots, and fit-time permutation
   materialization.
3. Add `Choice` as an unresolved random policy that resolves to only the
   selected processor.
4. Add independence, generator, selected-branch traversal, target-capability,
   stype, nested-container, and compile tests.
5. Replace the commented `Choice` placeholder in the TabICLv2 recipe only
   after `ConstantFilter`, `FeaturePermute`, and `StypeDispatch` are
   available on `main`.
6. Design the public `ResolveContext`, round-robin semantics, and checkpoint
   reconstruction separately when those requirements become active.

## Decision

Use explicit template resolution. The first implementation selects uniformly at
random during `recipe.resolve`, clones the complete processor graph before
selection, and keeps only selected branches in each concrete recipe.

Do not overload `copy.copy`, select during fitting, retain dead alternatives,
or promise state-dict reconstruction that the current processing stack cannot
provide. Defer round-robin until a stable context path can define its behavior.

This preserves direct Python composition, aligns with the estimator-resolution
direction in related PRs, keeps stochastic orchestration outside compiled
transforms, and gives every estimator an independent processor graph.
