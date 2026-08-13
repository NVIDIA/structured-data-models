import operator
from collections.abc import Iterable, Iterator
from typing import Any

import torch
from torch import Tensor


class BufferList(torch.nn.Module):
    """Store an indexed collection of registered buffers.

    Args:
        buffers: Tensors to register in order.
        persistent: Whether the buffers are included in :meth:`state_dict`.
    """

    def __init__(
        self,
        buffers: Iterable[Tensor] = (),
        *,
        persistent: bool = True,
    ) -> None:
        super().__init__()
        self._size = 0
        self._persistent = persistent
        for buffer in buffers:
            self.register_buffer(
                str(self._size),
                buffer,
                persistent=persistent,
            )
            self._size += 1

    def _load_from_state_dict(
        self,
        state_dict: dict[str, Any],
        prefix: str,
        local_metadata: dict[str, Any],
        strict: bool,
        missing_keys: list[str],
        unexpected_keys: list[str],
        error_msgs: list[str],
    ) -> None:
        if self._persistent:
            size = 0
            while isinstance(state_dict.get(f"{prefix}{size}"), Tensor):
                size += 1

            for index in range(len(self) - 1, size - 1, -1):
                delattr(self, str(index))
            self._size = size

            for index in range(size):
                name = str(index)
                state = state_dict[f"{prefix}{name}"]
                if name not in self._buffers:
                    self.register_buffer(
                        name,
                        torch.empty_like(state),
                        persistent=True,
                    )
                    continue

                current = self.get_buffer(name)
                if current.shape != state.shape:
                    setattr(
                        self,
                        name,
                        torch.empty(
                            state.shape,
                            dtype=current.dtype,
                            device=current.device,
                        ),
                    )

        super()._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )

    def __getitem__(self, index: int) -> Tensor:

        index = operator.index(index)
        if not -len(self) <= index < len(self):
            raise IndexError(f"index {index} is out of range")
        if index < 0:
            index += len(self)
        return self.get_buffer(str(index))

    def __iter__(self) -> Iterator[Tensor]:
        return (self[index] for index in range(len(self)))

    def __len__(self) -> int:
        return self._size
