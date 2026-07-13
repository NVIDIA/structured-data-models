# Prioritized TabICLv2 parity action plan

## Verified in this branch

All in-scope execution-parity tests pass after implementing the changes below
and rerunning processor, deterministic-model, semantic-stage, cache, and
pinned-checkpoint tests.

| Priority | Change                                                                                      | Scope                                                           | Failures resolved                                                            |
| -------- | ------------------------------------------------------------------------------------------- | --------------------------------------------------------------- | ---------------------------------------------------------------------------- |
| P0       | Keep one fitted deep-copied Recipe with each ensemble cache                                 | Generic pipeline correctness                                    | Member ordering, cache/non-cache replay, repeated prediction                 |
| P0       | Map every member to canonical class/original target space before averaging; run output once | Generic pipeline correctness                                    | Class direction, logit-vs-probability averaging, nonlinear inverse placement |
| P0       | Reconstruct classification output from fitted target categories; never call target inverse  | Current output design                                           | Obsolete TargetDispatch/class-score inverse coupling                         |
| P0       | Add fixed HardClip after feature scaling and use Power instead of Quantile                  | TabICLv2-specific Recipe configuration using generic Processors | Reference feature-stage mismatch                                             |
| P0       | Add configurable StypeDispatch route order and sorted CategoricalAlign vocabulary           | Generic pipeline capability, TabICLv2-specific configuration    | Mixed feature block order and sklearn vocabulary-code mismatch               |

These changes are dependency ordered: vocabulary/order alignment precedes
feature parity; member-local fitted state precedes correct output mapping;
canonical member mapping precedes aggregation; aggregation precedes the final
nonlinear output transform.

## Recommended follow-up

1. **P1 — Decide whether planning parity is required.** Implement a generic
   ensemble planner if exact TabICLv2 round-robin Identity/Power scheduling is
   desired. Current `Choice` sampling with replacement is an accepted planning
   limitation and does not fail explicit-Recipe execution parity. This could
   also serve future TabPFN/RFM planners without coupling `Model` to TabICLv2.
2. **P1 — Give stochastic Processors explicit generator ownership.** Choice,
   FeaturePermute, and ClassShuffle currently use the global torch CPU RNG.
   Explicit generators would improve independent plan reproduction and make
   cache metadata self-contained for future models.
3. **P2 — Add GPU Processor benchmarks when those paths are supported.** The
   current processing benchmark correctly reports CPU timings. Moving
   numerical preprocessing to GPU may reduce the 241 ms Large baseline, but
   host-backed StringTensor category metadata requires a separate design.
4. **P2 — Decide public user-output contracts.** SDM currently returns
   classification probabilities and all 999 regression coordinates. Label
   decoding and mean/median/quantile selection should be generic output
   processors if required, not hard-coded into TabICLv2 orchestration.
5. **P3 — Add non-perturbing memory telemetry in dedicated benchmark jobs.**
   Per-operation CPU peak memory is intentionally null. An external sampler
   can be added when memory, rather than short-operation latency, is the
   primary metric.

The first two follow-ups are generic pipeline/planning improvements. The
third is a cross-model performance opportunity. The fourth is an API decision.
None is required for the currently tested execution parity.
