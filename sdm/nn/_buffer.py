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
        if not self._persistent:
            super()._load_from_state_dict(
                state_dict,
                prefix,
                local_metadata,
                strict,
                missing_keys,
                unexpected_keys,
                error_msgs,
            )
            return

        loaded_keys = []
        for index in range(len(self)):
            key = f"{prefix}{index}"
            state = state_dict.get(key)
            if not isinstance(state, Tensor):
                continue
            current = self.get_buffer(str(index))
            setattr(
                self,
                str(index),
                state.to(device=current.device, dtype=current.dtype).clone(),
            )
            state_dict.pop(key)
            loaded_keys.append(key)

        key = f"{prefix}{len(self)}"
        while key in state_dict and isinstance(state_dict[key], Tensor):
            buffer = state_dict.pop(key)
            self.register_buffer(
                str(self._size),
                buffer.clone(),
                persistent=self._persistent,
            )
            self._size += 1
            loaded_keys.append(key)
            key = f"{prefix}{len(self)}"
        super()._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )
        for key in loaded_keys:
            if key in missing_keys:
                missing_keys.remove(key)

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
