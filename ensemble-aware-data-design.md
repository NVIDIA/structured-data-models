## Goal
Processors use the same code for single and ensemble inputs. Ensemble representation, alignment, sharing, and batching are handled by the data types.

## Types
```
EnsembleStorage[T]
    groups: list[T]
    locations: list[tuple[int, int]]

EnsembleData[T]
    storage: EnsembleStorage[T]

EnsembleTensor(EnsembleData[Tensor])
EnsembleTable(EnsembleData[TableTensor])
```

shared, batched & grouped are handled as before.

## Dispatch
EnsembleTensor uses __torch_dispatch__.
TableTensor exposes a central __table_dispatch__; EnsembleTable overrides it to lift existing TableTensor operations over ensemble storage.
```
Tensor operation      → __torch_dispatch__
TableTensor operation → __table_dispatch__
                              ↓
                    ensemble execution
```
This avoids reimplementing every TableTensor method/property in EnsembleTable.
EnsembleData provides the common execution layer required by EnsembleTensor and EnsembleTable operations. It translates logical ensemble semantics into efficient execution over the current physical representation, so dispatch implementations do not need to duplicate alignment, sharing, batching, and regrouping logic.
Execution must:

1. align ensemble operands by logical member; 
2. avoid unnecessary materialization of shared or batched inputs;
3. split only when member-specific computation requires it; 
4. reconstruct outputs in the most compact compatible representation.


Processor code never accesses storage, groups, or locations. E.g within Standardize we would have
```
mean = x.numerical.mean(dim=0)
return x.with_numerical(x.numerical - mean)
```

The physical ensemble representation is hidden from processor implementations.
