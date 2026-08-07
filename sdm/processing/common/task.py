from typing import Literal, cast

from torch.nn import ModuleDict

from sdm import Stype
from sdm.processing import EnsembleProcessor, Processor
from sdm.tensor import EnsembleTable


class TaskDispatch(EnsembleProcessor):
    """Apply separate processors based on the semantic type of the target.

    Args:
        classification: Processor selected for a categorical target.
        regression: Processor selected for a numerical target.
    """

    supported_stypes = frozenset(Stype)

    def __init__(
        self,
        *,
        classification: object = None,
        regression: object = None,
    ) -> None:
        super().__init__()
        self.requires_fit = False
        self.processors: ModuleDict[EnsembleProcessor] = ModuleDict()
        for task, processor in (
            ("classification", classification),
            ("regression", regression),
        ):
            if processor is None:
                continue
            processor = EnsembleProcessor.as_processor(processor)
            if processor.requires_fit:
                self.requires_fit = True
            self.processors[task] = processor

        self._task: Literal["classification", "regression"] | None = None

    def _transform_ensemble(
        self,
        ensemble_table: EnsembleTable,
    ) -> EnsembleTable:
        if self._task is None:
            raise RuntimeError(
                f"{self.__class__.__name__!r} has no resolved task; call "
                "'recipe.target.fit()' before transforming model output."
            )
        if self._task not in self.processors:
            return ensemble_table
        return self.processors[self._task].transform_ensemble(ensemble_table)

    def get_extra_state(self) -> str | None:
        r""":meta private:"""  # noqa: D415
        return self._task

    def set_extra_state(self, state: str | None) -> None:
        r""":meta private:"""  # noqa: D415
        self._task = cast(
            Literal["classification", "regression"] | None,
            state,
        )

    def __repr__(self, *, indent: int = 0) -> str:
        reprs = []
        for task, processor in self.processors.items():
            processor = cast(Processor, processor)
            processor_repr = processor.__repr__(indent=indent + 2)
            processor_repr = processor_repr[indent + 2 :]
            reprs.append(f"{' ' * (indent + 2)}{task}: {processor_repr}")
        return (
            f"{' ' * indent}{self.__class__.__name__}(\n"
            + ",\n".join(reprs)
            + f",\n{' ' * indent})"
        )
