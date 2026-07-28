# Reusing fitted preprocessing state across estimators

Status: proposal, not implemented.

This document proposes a way to stop recomputing identical preprocessing work when a model runs with more than one estimator.

______________________________________________________________________

## 1. Background

### 1.1 What a step is

A step is an object that transforms a table. In this repository a step is a subclass of `Processor` (`sdm/processing/base.py`). Examples are `Standardize`, `ImputeMean`, and `DropConstantColumns`.

A step has two phases.

**Fitting.** Calling `fit(table)` on a step makes the step read the table and store numbers on itself. `Standardize.fit` computes the mean and the standard deviation of every column, and stores both on the step object (`sdm/processing/numerical/standardize.py`).

**Transforming.** Calling `transform(table)` on a fitted step applies the stored numbers. `Standardize.transform` subtracts the stored mean and divides by the stored scale.

The word "fitted" describes the step, not the table. A fitted step is one that has learned its numbers and is ready to transform.

Some steps learn nothing. `Clip` clamps values to a fixed interval that came from its constructor, so `Clip` sets `requires_fit = False` (`sdm/processing/numerical/clip.py`).

### 1.2 What a recipe is

A recipe bundles three pipelines of steps (`sdm/processing/recipe.py`):

- `features` — applied to the input columns before the model runs.
- `target` — applied to the labels before the model runs, and inverted on the way out.
- `output` — applied to the model's raw output.

Each pipeline is a tree. The outer container is a `Sequential`, which runs its children in order. A `Sequential` may contain a `StypeDispatch`, which routes columns by semantic type to further `Sequential` objects. So a pipeline is a nested structure, not a flat list.

### 1.3 What an estimator is

`num_estimators` is an argument to `ICLModel.forward` and `ICLModel.fit` (`sdm/models/base.py`). Setting `num_estimators=8` runs the model 8 times and averages the 8 results.

There is only one set of model weights. The 8 runs differ because parts of the recipe are deliberately random. In TabICLv2's default recipe, one step picks one of two transformations at random, and two other steps draw random permutations. Averaging over those random draws is what the ensemble buys.

Because each estimator needs its own random draws and its own learned numbers, the recipe is deep-copied once per estimator:

```python
recipes = [copy.deepcopy(recipe) for _ in range(num_estimators)]
```

That line appears twice, in `forward` and in `fit` (`sdm/models/base.py`).

### 1.4 What randomness control looks like today

A `torch.Generator` is PyTorch's random-number state object. The caller passes one as `generator=` to `forward` or `fit`. That single generator is threaded down into every step's fit call. `CLAUDE.md` requires stochastic transformations to expose generator control, so a step that draws randomly is expected to draw through the generator handed to it.

### 1.5 What a host-device synchronization is

When Python calls a CUDA operation, the work is placed on a queue for the GPU rather than run on the spot. The CPU moves straight on to the next line. That gap is deliberate. It lets the CPU stay far enough ahead that the GPU always has its next piece of work waiting.

A synchronization is anything that forces the CPU to learn a value living in GPU memory: `.item()`, `.tolist()`, `.cpu()`, `float(t)`, `if t > 0`, or printing a CUDA tensor. The CPU cannot answer such a question until the queued work has finished, so it stops and waits.

The wait itself is short. The lasting cost is that the queue empties. Once the CPU resumes, the GPU has nothing to do until the CPU has built its lead back up. Repeated in a loop, this makes the two take turns instead of working at the same time. `CLAUDE.md` forbids these calls in processor hot paths for that reason.

______________________________________________________________________

## 2. The problem

Take the numerical part of TabICLv2's default feature pipeline (`sdm/models/tabiclv2/recipe.py`), in order:

1. `ImputeMean` — learns each column's mean, replaces missing values with it.
2. `DropConstantColumns` — learns which columns carry no information, drops them.
3. `Standardize` — learns each column's mean and scale, centers and divides.
4. `Clip` — clamps to a fixed interval. Learns nothing.
5. `Choice(Identity, PowerTransform)` — draws one of two options at random, applies only the drawn one.
6. `ClipSigma` — learns outlier bounds from the data, clips to them.
7. `ShuffleColumns` — draws a random column permutation, reorders columns.

Two steps run before this list, handling categorical columns: `AlignCategories` builds a category vocabulary, and `ToNumerical` converts the resulting codes into numbers.

With `num_estimators=8`, the estimator loop runs this entire chain 8 times, on the same context table.

Steps 1 through 4 contain no randomness. Given the same input table, all 8 runs compute the same means, drop the same columns, compute the same scales, and produce the same output table. Seven of those eight runs are pure waste. The same applies to the two categorical steps that run before them.

Step 5 is where the 8 runs stop being identical, because `Choice` draws. Steps 6 and 7 receive different input depending on that draw, so those two steps cannot be shared across all 8 estimators.

**The waste has two parts, and the second is larger.** Recomputing a column mean is cheap. Re-running the transform over a large context table 8 times is not. The transform cost scales with the number of rows.

The waste grows in three directions:

- With `num_estimators`. Eight estimators waste seven copies of the work.
- With expensive fits. An open pull request adds a TF-IDF step for text columns whose fit takes several seconds on large inputs. That step is deterministic and would be recomputed once per estimator.
- With related tables. KumoRFM attaches extra tables to the input, and the feature pipeline is fitted separately on each one (`sdm/models/base.py`). The number of fitted pipelines is `num_estimators × (1 + number of related tables)`.

______________________________________________________________________

## 3. Prerequisite: measure the problem

The size of the problem is unverified. Nothing below should be built before measuring whether preprocessing is a meaningful fraction of total runtime next to the model forward pass.

At the scale used in `examples/tabiclv2.py`, which is 300 context rows, it will not be. The case for this mechanism rests on large context tables, high estimator counts, expensive steps such as the proposed TF-IDF processor, or KumoRFM's related-table multiplier. The measurement should cover those cases specifically, using CUDA events or explicit synchronization, since plain wall-clock timing of asynchronous GPU work is not valid.

If preprocessing turns out to be a small share of runtime in all of those cases, this proposal should not be built.

______________________________________________________________________

## 4. Goal and non-goals

**Goal.** Compute each distinct preprocessing result once. Several estimators often produce identical fitted numbers and identical output tables. The first estimator does that work, and the rest reuse the result.

**Non-goals.**

- Reducing memory. See section 8; this proposal spends memory to save time.
- Reuse across separate calls to `forward` or `fit`. Scope is one call.
- Reuse across related tables. Different tables hold different data, so their fitted numbers legitimately differ.
- Hard-coding knowledge of specific steps. The mechanism must work for steps written by users, and must not contain a list of known-deterministic step names.

______________________________________________________________________

## 5. The proposal, in three parts

### Part 1 — Detect randomness by watching the generator

A container records the generator's state before it calls a child step's fit, and compares it against the state after the call returns.

An unchanged state means that fit consumed no randomness. The result was fully determined by the input table, so another estimator receiving the same input table will learn the same numbers.

This needs no cooperation from the step. It works for steps that did not exist when the mechanism was written. It also handles steps that are random only under some conditions: `QuantileTransform` draws a row subsample only when the configured subsample size is smaller than the number of rows (`sdm/processing/numerical/quantile.py`). The condition is evaluated while the fit runs, so the comparison reports what actually happened rather than what might have happened.

**How the state is read.** `torch.Generator` is a C extension type. It cannot be subclassed, and a Python object that merely imitates it will be rejected by the tensor operations that type-check their `generator=` argument, so a proxy that counts calls is not an option. The state is read with `generator.get_state()` instead.

For a CUDA generator, that state is a seed and an offset which PyTorch keeps in host memory, so reading it should not touch the GPU queue. This is the one claim in this document that must be verified against the installed PyTorch before anything is built on top of it. If it turns out to synchronize, Part 1 is not viable in its current form, and steps would have to declare their own determinism instead.

For a CPU generator, the state is the full Mersenne Twister buffer of roughly 5 KB. Comparing two of them is a memory comparison that runs once per step, which is negligible.

### Part 2 — Give each step a key

A key is a label that identifies the input table a step is about to be fitted on. Two estimators that arrive at the same step with the same key will fit that step on the same table, so one of the two fits can be skipped and its result reused.

Two facts determine what a key must contain.

First, all `num_estimators` recipes in a call are deep copies of one original recipe. Their settings are therefore identical by construction. A key never needs to encode settings such as `epsilon=1e-6`.

Second, a key must not be computed from the values inside a table. Reading tensor values means a host-device synchronization (section 1.5), and it would happen once per step and per estimator.

What remains is a pair:

- **Position** — where the step sits in the recipe tree.
- **Draw history** — what every random draw before that step returned.

Position must be a path, not an index. A `Sequential` inside the categorical branch and a `Sequential` inside the numerical branch would both start counting their children at 0, so plain indices collide. A path such as `("features", "numerical", 3)` does not.

Applying this to TabICLv2 with 8 estimators:

| Point in the chain                | Draw history          | Distinct keys                 |
| --------------------------------- | --------------------- | ----------------------------- |
| Categorical steps, then steps 1–4 | empty                 | 1, shared by all 8 estimators |
| At `Choice`                       | the drawn index       | 2, one per branch             |
| At `ShuffleColumns` and after     | the drawn permutation | 8, one per estimator          |

The cached value stored under a key is a pair: the numbers the step learned, and the table the step produced.

### Part 3 — Let a random step report a comparable draw

Under Part 2 alone, any random step separates every estimator permanently. That loses a real opportunity. With 8 estimators and a `Choice` between two options, roughly half the estimators draw the same branch, and estimators that drew the same branch have identical tables afterwards.

To capture that, `Processor` gains one optional method. The method returns a small value describing what the step drew. The value must be readable on the host without touching device data.

- The default implementation returns nothing, meaning "my draw is not comparable". A step reporting nothing separates every estimator that reaches it.
- `Choice` overrides the method to return the index it drew. That index is already a plain Python integer (`sdm/processing/common/choice.py`).
- `ShuffleColumns` could report its permutation at no extra cost. Its `_transform` already calls `permutation.tolist()` in order to reorder the column names (`sdm/processing/common/shuffle.py`), so the permutation reaches the host on every transform in any case. It is the last step in TabICLv2's chain, so reporting buys nothing there, but it would matter in a recipe that continues past it.

This is the part that makes the mechanism degrade gracefully. A step that opts into nothing still works correctly. It simply yields less reuse.

______________________________________________________________________

## 6. Which layer invokes the caching

Only the container steps participate: `Sequential`, `StypeDispatch`, and `Choice`. Those classes walk their children, so those classes know the path and the ordering.

A leaf step such as `Standardize` or a user-written TF-IDF step never touches the cache and needs no changes. That is what keeps the mechanism general.

The caching logic itself lives in `sdm/processing`. The estimator loop in `sdm/models/base.py` only marks where a group of estimators begins and ends. `CLAUDE.md` requires model wrappers to stay thin and shared abstractions to live outside model implementations.

### 6.1 What a container does for each child

Today `Sequential` runs `out = child.fit_transform(out, generator=generator)` for each child in turn. With caching on, that call is routed through the scope, and the scope does four things.

1. **Build the key.** The current path plus this child's segment, together with the draw history so far. Both are ordinary Python values.
2. **Look it up.** On a hit, load the stored numbers into the child and return the stored table. No fit runs, no transform runs, no tensor work happens at all.
3. **On a miss, run the child and watch the generator.** Read the generator state, call `fit_transform`, read it again.
4. **Decide what to keep.** If the two readings match, the child drew nothing, so store its numbers and its output table under the key. If they differ, store nothing, and append what the child reports it drew to the draw history, so that every later sibling gets a different key.

Nesting needs no extra handling. A child that is itself a container pushes its own path segment and repeats the same four steps for its own children. A subtree whose first half is deterministic therefore caches at the inner level while missing at the outer one.

### 6.2 A trace through TabICLv2

The numerical route, with 8 estimators.

Estimator 0 runs `ImputeMean`, `DropConstantColumns`, `Standardize` and `Clip`. Each stores an entry, and each new entry's table replaces the one stored before it (section 8). Estimator 0 then reaches `Choice` and draws, so the draw history becomes the drawn index. `ClipSigma` and `ShuffleColumns` are keyed under that history.

Estimators 1 to 7 hit the cache for the first four steps and do no tensor work there. Each then draws its own branch at `Choice`. Those that drew the same branch as estimator 0 also hit the cache for `ClipSigma`.

### 6.3 Where the fitted numbers land

Each estimator keeps its own deep copy of the recipe, and it must. The copy is used again after the estimator loop, in two places: `forward` transforms the query rows through it, and `fit` stores it in a `Cache` for later `predict` calls.

So a cache hit does not hand back a shared step object. It copies the stored numbers into that estimator's own step, through `load_state_dict`. The copy is column-sized and therefore cheap, and it avoids two estimators holding one mutable object between them. `Choice` already exposes its drawn index through `get_extra_state` and `set_extra_state`, so a whole subtree restores correctly in a single call.

______________________________________________________________________

## 7. How the key travels through the call

Nothing is added to any public method signature. Adding a parameter to `fit` or `transform` would break every user-written step.

Instead the estimator loop opens a scope object before the loop and closes it after. The scope holds two things: the cache itself, and the key being built.

Each container pushes its own path segment onto the scope when it descends into a child, and pops the segment when the child returns. Each random draw appends its reported value, or a unique marker when the step reports nothing. So at any moment the scope holds the current path and the current draw history, which together form the key.

Because Part 1 reads the state of the generator itself, a draw made at any depth is visible to every enclosing container.

______________________________________________________________________

## 8. What is saved, and how memory is bounded

Only the first estimator ever produces intermediate tables. Estimator 0 runs the chain step by step, so a table exists after each step. Every later estimator produces none of them, because each one reads a cached table and continues from there.

So the only question is how many of estimator 0's intermediate tables the cache holds on to. The answer is one. Every estimator runs the same steps in the same order, so no estimator resumes from the middle of the shared portion, and only **the last table produced before the estimators separate** is ever read.

While estimator 0 is still running, the cache does not yet know which table that will be, because separation depends on draws that have not happened. So the cache keeps a rolling entry, where each new deterministic step replaces the table stored by the previous one. Peak cost is one table, not one per step.

The fitted numbers are a separate matter. Those must be kept for every step, not just the last one, because the query rows are later transformed through the whole chain using them. Keeping them is cheap: a few values per column.

Sizing the two:

- Fitted numbers scale with the number of columns.
- A cached table scales with rows times columns.

With a large context table, one cached table is far larger than all the fitted numbers of all estimators combined.

**On the relationship to `deepcopy`.** Removing the per-estimator `deepcopy` would not save meaningful memory. A deep copy duplicates what a step learned, which is column-sized. The mechanism proposed here spends row-sized memory to save compute. The two are not comparable, and this proposal makes memory use go up, not down.

______________________________________________________________________

## 9. What this solves and what it does not

### Solved

- Repeated fitting and repeated transforming of the deterministic portion of a pipeline across estimators, within one call.
- The same waste inside a `Choice` branch, for estimators that drew the same branch.
- The same waste in recipes never seen by this mechanism, including recipes built entirely from user-written steps, because nothing is hard-coded.

### Not solved

- **Memory.** Section 8. This proposal increases peak memory.
- **Reuse across calls.** Two separate `forward` calls share nothing. Adding that would require encoding step settings in the key, because the two calls may pass different recipe objects.
- **Reuse across related tables.** Each related table holds different data. Their fitted numbers differ legitimately.
- **The `deepcopy` itself.** Copies are still made. Only the recomputation is avoided.
- **Recipes with early randomness.** A recipe whose first step draws randomly gets no reuse at all. That is correct behavior, not a gap, but it means the benefit is recipe-dependent rather than universal.

______________________________________________________________________

## 10. Assumptions

**A1. All recipes within one call are copies of one original.** This is what allows the key to ignore step settings. It holds because `forward` and `fit` construct the copies themselves.

**A2. Random draws go through the passed generator.** `CLAUDE.md` already states this requirement. Part 1 depends on it. See section 11 for what happens when it is violated.

**A3. Steps do not modify their input table in place.** *Verified.* A search of `sdm/processing/` for in-place tensor operations found one result, `scale.fill_(1.0)` in `Standardize._fit`, and the tensor there is freshly allocated rather than part of the input. Every `transform` returns a new table through `TableTensor.replace_blocks`. This assumption must be preserved, since sharing a cached table between estimators makes any future in-place write corrupt every estimator holding that table.

**A4. Fitting is a meaningful fraction of runtime.** Unverified. See section 3.

______________________________________________________________________

## 11. Edge cases and hazards

### A step that hides its randomness

**What it is.** A step calls `torch.randperm(n)` without passing the generator it was given. PyTorch then uses its global random state, which the state comparison in Part 1 cannot observe. The step looks deterministic while actually being random.

**Can it happen.** Yes. `Processor` is a public base class and users write subclasses. No step inside this repository does this today; all four random steps thread the generator correctly.

**Consequence.** The cache treats the step as deterministic and gives every estimator the draw made by the first estimator. No error is raised, and each estimator remains internally consistent, so no prediction is wrong in itself. What is lost is ensemble diversity. The estimators become more alike than intended, so averaging them reduces less variance and accuracy degrades. If the offending step was the only source of randomness in the recipe, all estimators collapse to the same prediction, and `num_estimators` costs full compute while buying nothing.

**Response.** Do not guard at runtime. Detecting it means reading PyTorch's global random state around every fit, and on CUDA that costs a synchronization which `CLAUDE.md` forbids in these paths. Instead: state the contract in the `Processor` documentation, and provide a debug mode that runs the same input twice with caching on and off and compares the results. That check belongs in tests.

### Shifted random number consumption

**What it is.** Picture the generator as a single tape of random numbers read from front to back. All estimators share one tape. Estimator 0 reads the first few numbers, estimator 1 continues from there.

Caching lets an estimator skip work. If the skipped work would have read from the tape, that estimator reads fewer numbers than before, and every estimator after it now reads different numbers than before.

**Consequence.** Results are not wrong, but they differ from what the same code produced without caching. That breaks the most useful test of the whole mechanism, which is comparing a cached run against an uncached one.

**When it applies.** Only when a reusable cache entry covers work that drew from the tape. TabICLv2's default recipe never triggers it, because the reused steps draw nothing. It appears in shapes such as a `Choice` whose selected branch itself draws.

**Response.** Derive one seed per estimator once, at the start of the call, and give each estimator its own generator. What estimator 1 reads then no longer depends on how much estimator 0 read.

This is worth doing on its own, and before the rest. It is what makes the mechanism testable: run the same input twice, once with caching and once without, and assert that the outputs match. That test cannot be written while the estimators share one tape, because caching shifts the tape. Per-estimator generators also decouple the estimators from one another, which is a prerequisite for ever running them in parallel, and for reproducing a single estimator on its own.

The seeds are drawn from the generator the caller passed, so the call as a whole remains reproducible from that generator. Statistically nothing changes, since the estimators' draws are independent either way. Deriving the seeds costs one host-device synchronization at the start of the call, which is outside any hot path.

One consequence to declare: the same seed will produce different numbers than it does today. The distribution is unchanged, but the specific draws are not.

### A shared entry being refitted later

**What it is.** Two estimators point at one fitted step object. One of the two refits that object. The other estimator's numbers change without warning.

**Can it happen today.** No. `forward` never refits after the estimator loop. `fit` stores each recipe in a `Cache` and calls `freeze()`. `predict` only transforms. Nothing refits a fitted step.

**Response.** Freeze cache entries when they are inserted. This is inexpensive insurance against a future caller that holds a returned recipe and refits it.

### Steps that are random only sometimes

Covered by Part 1. `QuantileTransform` is deterministic below a configured row threshold and random above it. Because the comparison observes the actual fit, no static classification of the step is needed.

### Ties in deterministic ordering

`AlignCategories(sort_by="frequency")` orders categories by count. Equal counts are resolved by a stable sort (`sdm/processing/categorical/align.py`), so the result is reproducible. If a future step used an unstable sort, two runs on the same input could differ while consuming no randomness, and the state comparison would wrongly report the step as reusable. Deterministic ordering is therefore part of assumption A2's spirit and should be stated in the `Processor` contract.

______________________________________________________________________

## 12. Interface sketch

Names are proposals.

- `FitKey` — the path and draw history described in Part 2.
- `FitCache` — the store mapping a `FitKey` to a fitted state and a table.
- `Processor.draw_key()` — the optional method from Part 3. Returns `None` by default.
- One new keyword argument on `ICLModel.forward` and `ICLModel.fit` to switch the mechanism on or off.

The switch exists mainly for verification. When the contract in A2 holds, caching changes nothing observable, so no user has a reason to turn it off except while debugging. Whether it defaults to on or off should be decided by the measurement in section 3.
