# Table dispatch in `Recipe`

Add `TableDispatch(task=..., related=...)` as a composable step in `Recipe.features`. Ordinary feature steps continue to apply to every table; users add `TableDispatch` only where task and related tables differ. An omitted route is an identity operation, and every related table gets its own fitted processor state.

`task` matches the established `task_table` and `task_links` vocabulary and remains correct for entity, event, and link-prediction tasks; `entity` would be too narrow. `related` matches `RelatedTables`. The API has no `all` route because an ordinary adjacent step already expresses “all tables”. Dispatch by table name is a separate concern and is not part of this proposal.

## User model

Start with a normal linear feature pipeline. Use `TableDispatch` only when the requirement contains “only on the task table”, “only on related tables”, or specifies different processing for the two. For example, this recipe expands related-table timestamps before applying the shared TabICL-style processing:

```python
recipe = Recipe(
    features=[
        TableDispatch(
            related=StypeDispatch(
                datetime=AddCalendarFields(
                    fields=("hour", "weekday", "month"),
                ),
            ),
        ),
        StypeDispatch(
            categorical=[AlignCategories(), ToNumerical()],
            numerical=[ImputeMean(), Standardize()],
        ),
        Choice(Identity(), ShuffleColumns(method="shift")),
    ],
    target=StypeDispatch(
        categorical=[AlignCategories(), ShuffleCategories()],
        numerical=Standardize(),
    ),
    output=[
        ReduceEstimators(method="mean"),
        TaskDispatch(classification=Softmax()),
    ],
)
```

Read this top to bottom: related tables execute the first route while the task table passes through; every table then executes the two shared steps. Each ensemble member follows the existing `Choice` semantics. Target and output processing are unchanged.

The dispatchers answer different questions: `StypeDispatch` selects columns by semantic type, `TaskDispatch` selects output processing for classification or regression, and `TableDispatch` selects feature processing for the task table or related tables.

## Semantics and implementation sketch

`TableDispatch` may appear anywhere inside the feature tree, including `Sequential`, `Choice`, and `StypeDispatch`. Before fitting, `Recipe.bind` recursively specializes that tree into ordinary task and related processor trees:


```
TableDispatch:
routes:
task = EnsembleProcessor.as_processor(task)
related = EnsembleProcessor.as_processor(related)

  selected_route = unresolved

  fit/transform:
      fail if selected_route is unresolved
      pass through if that route was omitted
      otherwise delegate to routes[selected_route]


\_RecipeExecution.\_bind(recipe, x, y, related_tables, ...):
\# Must happen first: resolves TaskDispatch as defined by PR #568.
y_out = recipe.target.fit_transform_ensemble(y)

  related_processors = {}
  related_outputs = {}

  for name, table in related_tables:
      features = deepcopy(recipe.features)

      for processor in features.modules():
          if processor is TableDispatch:
              processor.selected_route = "related"

      related_processors[name] = features
      related_outputs[name] = features.fit_transform_ensemble(table)

  for processor in recipe.features.modules():
      if processor is TableDispatch:
          processor.selected_route = "task"

  x_out = recipe.features.fit_transform_ensemble(x)

  return _RecipeExecution(
      recipe=recipe,
      task_context=x_out,
      target_context=y_out,
      related_context=related_outputs,
      related_processors=related_processors,
  )
```
