from collections.abc import Mapping

import torch
from sdm import TableTensor
from sdm.explain import (
    Explanation,
    ExplanationCallable,
    ExplanationInputs,
    ExplanationMethod,
    ExplanationMode,
    ExplanationRequirements,
    OutputIndex,
)


class FittedEndpointMethod(ExplanationMethod):
    requirements = ExplanationRequirements(
        supported_modes=frozenset({ExplanationMode.fitted})
    )

    def explain(
        self,
        *,
        evaluate: ExplanationCallable,
        inputs: ExplanationInputs,
        prediction: TableTensor,
        target: OutputIndex,
        mode: ExplanationMode,
    ) -> Explanation:
        assert mode is ExplanationMode.fitted
        assert all(site.split == "query" for site in inputs)
        site, table = next(iter(inputs.items()))
        evaluate({site: table.numerical + 0.01})
        repeated = evaluate(None)
        assert repeated.allclose(
            prediction,
            rtol=0.0,
            atol=0.0,
            equal_nan=True,
        )
        prediction = prediction.replace_blocks(
            numerical=prediction.numerical.detach()
        )
        return Explanation(
            prediction=prediction,
            target=target.resolve(prediction),
            method=self.name,
            mode=mode,
        )


def cache_tensors(value: object) -> tuple[torch.Tensor, ...]:
    if isinstance(value, torch.Tensor):
        return (value,)
    if isinstance(value, Mapping):
        return tuple(
            tensor for item in value.values() for tensor in cache_tensors(item)
        )
    if isinstance(value, list | tuple):
        return tuple(
            tensor for item in value for tensor in cache_tensors(item)
        )
    return ()
