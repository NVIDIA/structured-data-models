# Table dispatch in `Recipe`

Add `TableDispatch(task=..., related=...)` as a composable step in `Recipe.features`. Ordinary feature steps continue to apply to every table; users add `TableDispatch` only where task and related tables differ. An omitted route is an identity operation, and every related table gets its own fitted processor state.

`task` matches the established `task_table` and `task_links` vocabulary and remains correct for entity, event, and link-prediction tasks; `entity` would be too narrow. `related` matches `RelatedTables`. The API has no `all` route because an ordinary adjacent step already expresses “all tables”. Dispatch by table name is a separate concern and is not part of this proposal.

## User model

Start with a normal linear feature pipeline. Use `TableDispatch` only when the requirement contains “only on the task table”, “only on related tables”, or specifies different processing for the two. For example, this recipe expands related-table timestamps before applying the shared TabICL-style processing:

```python
import sdm.processing as sp

recipe = sp.Recipe(
    features=[
        sp.TableDispatch(
            related=sp.StypeDispatch(
                datetime=sp.AddCalendarFields(
                    fields=("hour", "weekday", "month"),
                ),
            ),
        ),
        sp.StypeDispatch(
            categorical=[sp.AlignCategories(), sp.ToNumerical()],
            numerical=[sp.ImputeMean(), sp.Standardize()],
        ),
        sp.Choice(sp.Identity(), sp.ShuffleColumns(method="shift")),
    ],
    target=sp.StypeDispatch(
        categorical=[sp.AlignCategories(), sp.ShuffleCategories()],
        numerical=sp.Standardize(),
    ),
    output=[
        sp.ReduceEstimators(method="mean"),
        sp.TaskDispatch(classification=sp.Softmax()),
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


Test


  ### 1. Kernsemantik und unabhängiger Fit

  test_table_dispatch_scopes_and_fits_related_tables_independently

  recipe = sp.Recipe(
      features=sp.TableDispatch(
          related=sp.StypeDispatch(numerical=sp.Standardize()),
      ),
  )

  Über `_RecipeExecution._bind` / `.transform` mit Task-Tabelle sowie users und orders:

  - Task-Context und Task-Query bleiben unverändert.
  - Beide Related-Context-Tabellen werden standardisiert.
  - Related Queries verwenden den jeweils passenden Fit-Zustand.
  - users und orders bekommen unabhängig gefittete Processor-Kopien.

  Damit sind Related-only, fehlende Task-Route als Passthrough, mehrere Related Tables und
  zustandsbehaftete Verarbeitung abgedeckt.

  ### 2. Rekursive Komposition und Ensemble

  test_nested_table_dispatch_routes_ensemble_members

  recipe = sp.Recipe(
      features=sp.StypeDispatch(
          numerical=sp.Choice(
              sp.Sequential(
                  sp.TableDispatch(
                      task=Add(10),
                      related=Add(100),
                  ),
              ),
              Add(1),
              selection="round_robin",
          ),
      ),
  )

  Mit zwei Ensemble-Mitgliedern prüfen:

  member 0:
      task    += 10
      related += 100

  member 1:
      task    += 1
      related += 1

  Dieser eine Test validiert gemeinsam:

  - TableDispatch unter Sequential;
  - TableDispatch unter Choice;
  - TableDispatch unter StypeDispatch;
  - beide nichtleeren Routen;
  - Ensemble-Ausführung;
  - rekursive Modulauflösung.

  Kein Seed nötig, weil round_robin deterministisch ist.


  ### 3.  Ungültige Platzierung

  test_recipe_rejects_table_dispatch_outside_features

  @pytest.mark.parametrize("field", ["target", "output"])
  def test_recipe_rejects_table_dispatch_outside_features(field):
      with pytest.raises(ValueError, match="Recipe.features"):
          sp.Recipe(**{field: sp.TableDispatch(related=sp.Identity())})

  TableDispatch() ohne konfigurierte Route würde ich dagegen nicht zwingend ablehnen: Nach
  #568 darf auch TaskDispatch() leer sein, und AGENTS.md verlangt minimale Validierung. Ein
  leerer Dispatcher kann konsistent als Identity funktionieren.