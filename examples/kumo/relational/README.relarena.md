# RelArena model and system examples

These are native submission interfaces, not benchmark runners. Import the relevant module to register it, then let RelArena or your own caller handle task execution, hardware, results and whole-task timing.

| Interface                  | Module                                     | Selection and final fit                                                                                                                                       |
| -------------------------- | ------------------------------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Model (`sdm-kumo`)         | [rel_arena.py](rel_arena.py)               | RelArena tunes the declared shared search space on validation and performs its required default/selected final fits.                                          |
| System (`sdm-kumo-system`) | [rel_arena_system.py](rel_arena_system.py) | The system evaluates its candidate policy on full validation or a fixed subset, then refits the winner on full TRAIN+VAL and returns masked TEST predictions. |

Both use the shared [\_relarena/adapter.py](_relarena/adapter.py) predictor. RelArena owns hidden TEST scoring. The system does not register a model search space, even though its current internal policy reuses the same candidates. Neither interface imports a task-specific winner map or selects using TEST results.

Each module's opening docstring shows a one-task Python example. For the system, pass `validation_rows=10_000` to `KumoSystem.run` to cap tuning at 10,000 uniformly sampled rows, or omit it for full validation. The run seed selects one label-independent subset shared by every candidate; final TRAIN+VAL and TEST are not subsampled. `system.validation_rows` records the actual count. Tiny classification samples can lack a class and fail scoring. The standard `run_system_experiment` entrypoint does not forward this optional keyword; use the direct call shown in the docstring when setting a cap. The MODEL entrypoint continues to score full validation.

Use RelArena commit `4dece53d3cef84f3e2b18ceb3cea13075afefdab`, RelBench 2.1.2, SDM's CUDA environment and a matching CPU PyG sampler. Text features additionally require sentence-transformers. The integration uses V11's public `KumoRelational` v2.1.2 weights (`nvidia/Kumo-Relational`, revision `6115b706c266c81b386faa5dcd0f5ff22faea444`), not PR #676's older Nemotron checkpoint.

## Preserved behavior

- Eight independently sampled contexts, each capped at 10,000 rows; one query neighborhood shared across the eight estimators. Smaller training tables use all available rows.
- V11 neural precision and regression median/inverse-transform handling. Compilation is not enabled.
- Canonical table/relationship and neighbor tie ordering; corrected label-independent selection among equal-time rows when the latest-80k pool is enabled.
- Bounded text tensorization and frozen raw-Qwen vector caching before separately context-fitted PCA. The 32 GiB limit is disk vector payload, not RAM or VRAM; SQLite needs additional disk overhead. Supply a task-local cache directory through RelArena's native cache argument. Cold cache creation remains method work.
- Original ten-window raw-event lag features and generic mature-label history remain distinct options. They use only the supplied phase-censored database and eligible training labels, never query labels.

## Search and runtime status

[search_space.py](_relarena/search_space.py) carries the provisional 30-candidate design from stopped V12. It covers neighborhood/hops, lag mode, text OFF/PCA32, standard/quantile targets, recency and entity dates. It is common across tasks and is not a completed or optimized benchmark result. The model's trial budget belongs to the caller; the system currently evaluates all 30 candidates internally.

The caller must count preprocessing, cache creation, all selection work, final fitting and predictions toward RelArena's complete per-task budget. The native system `time_limit` covers its procedure after receiving the splits; it cannot account for earlier caller-side preparation. No custom launcher, deadline supervisor, result writer, Elo scorer, checksum package or cloud orchestration is included here. Full 24-hour compliance and GPU performance of this fresh-main integration have not been benchmarked.
