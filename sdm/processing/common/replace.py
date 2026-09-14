from collections.abc import Callable

from torch import Tensor

from sdm import (
    CategoricalTensor,
    ColumnarTensor,
    StringTensor,
    Stype,
    TableTensor,
)
from sdm.processing import Processor


class ReplaceBlocks(Processor):
    """Replace semantic blocks with the outputs of stateless functions.

    Each configured function receives only its semantic block. Its return value
    replaces that block while column names and unconfigured blocks are
    preserved.
    Functions are not called for empty blocks.

    Args:
        numerical: Function mapping the numerical block of shape
            ``[..., num_numerical]`` to a replacement with the same shape.
        categorical: Function mapping the categorical block of shape
            ``[..., num_categorical]`` to a replacement with the same shape.
        datetime: Function mapping the datetime block of shape
            ``[..., num_datetime]`` to a replacement with the same shape.
        text: Function mapping the text block of shape ``[..., num_text]`` to a
            replacement with the same shape.
        id: Function mapping the ID block of shape ``[..., num_id]`` to a
            replacement with the same shape.
    """

    requires_fit = False

    def __init__(
        self,
        *,
        numerical: Callable[[Tensor], Tensor] | None = None,
        categorical: Callable[[CategoricalTensor], CategoricalTensor]
        | None = None,
        datetime: Callable[[Tensor], Tensor] | None = None,
        text: Callable[[StringTensor], StringTensor] | None = None,
        id: Callable[[ColumnarTensor], ColumnarTensor] | None = None,
    ) -> None:
        super().__init__()
        self.numerical = numerical
        self.categorical = categorical
        self.datetime = datetime
        self.text = text
        self.id = id
        self.handles_stypes = frozenset(
            stype
            for stype, function in (
                (Stype.numerical, numerical),
                (Stype.categorical, categorical),
                (Stype.datetime, datetime),
                (Stype.text, text),
                (Stype.id, id),
            )
            if function is not None
        )

    def _transform(self, table: TableTensor) -> TableTensor:
        numerical = None
        if self.numerical is not None and table.numerical.size(-1) > 0:
            numerical = self.numerical(table.numerical)

        categorical = None
        if self.categorical is not None and table.categorical.size(-1) > 0:
            categorical = self.categorical(table.categorical)

        datetime = None
        if self.datetime is not None and table.datetime.size(-1) > 0:
            datetime = self.datetime(table.datetime)

        text = None
        if self.text is not None and table.text.size(-1) > 0:
            text = self.text(table.text)

        id = None
        if self.id is not None and table.id.size(-1) > 0:
            id = self.id(table.id)

        return table.replace_blocks(
            numerical=numerical,
            categorical=categorical,
            datetime=datetime,
            text=text,
            id=id,
        )
