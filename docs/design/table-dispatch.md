# Table dispatch in `Recipe`

Status: Proposed

## Decision

Add `TableDispatch(task=..., related=...)` as a composable step in `Recipe.features`. Ordinary feature steps continue to apply to every table; users add `TableDispatch` only where task and related tables differ. An omitted route is an identity operation, and every related table gets its own fitted processor state.

`TableDispatch` is shorter and more specific than `TableRoleDispatch`: the route names already communicate the role. `task` matches the established `task_table` and `task_links` vocabulary and remains correct for entity, event, and link-prediction tasks; `entity` would be too narrow. `related` matches `RelatedTables`. The API has no `all` route because an ordinary adjacent step already expresses “all tables”. Dispatch by table name is a separate concern and is not part of this proposal.

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

The dispatchers answer different questions: `StypeDispatch` selects columns by semantic type, `TaskDispatch` selects output processing for classification or regression, and `TableDispatch` selects feature processing for the task table or related tables. `Choice` still represents ensemble alternatives. Users do not need a separate public “role” concept.

## Semantics and implementation sketch

`TableDispatch` may appear anywhere inside the feature tree, including `Sequential`, `Choice`, and `StypeDispatch`. Before fitting, `Recipe.bind` recursively specializes that tree into ordinary task and related processor trees:

```text
specialize(processor, table_kind):
    if processor is TableDispatch:
        route = processor[table_kind] or Identity()
        return specialize(deepcopy(route), table_kind)

    result = deepcopy(processor)
    for child_slot, child in processor.children():
        result[child_slot] = specialize(child, table_kind)
    return result

bind(recipe, task_context, related_context):
    task_features = specialize(recipe.features, "task")
    related_template = specialize(recipe.features, "related")

    task_features.fit_transform_ensemble(task_context)
    for name, table in related_context.tables:
        related_features[name] = deepcopy(related_template)
        related_features[name].fit_transform_ensemble(table)

    return Execution(task_features, related_features)
```

Query tables reuse their corresponding fitted processors. Specialization happens once per binding, so `TableDispatch` adds no branch to tensor execution and requires no change to `Processor`, ensemble, `Sequential`, `Choice`, or stype-dispatch semantics. Construction rejects an empty dispatcher and use outside `Recipe.features`. Because an unresolved scoped tree has no table context, it is executed through `Recipe.bind`; the specialized processors remain private execution state.

Do not add role-specific `Recipe` fields or `TableScope(all=..., task=..., related=...)`. Both split one ordered data flow across multiple locations, and a fixed `all` phase cannot express a related-only step that must run before a shared step without duplication or extra ordering rules.
