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
        loaded_keys = []
        if self._persistent:
            size = 0
            while isinstance(state_dict.get(f"{prefix}{size}"), Tensor):
                size += 1

            for index in range(len(self) - 1, size - 1, -1):
                delattr(self, str(index))
            self._size = size

            assign = local_metadata.get("assign_to_params_buffers", False)
            for index in range(size):
                name = str(index)
                key = f"{prefix}{name}"
                state = state_dict[key]
                current = self._buffers.get(name)
                if type(state) is not Tensor:
                    if assign:
                        buffer = state
                    elif current is None:
                        buffer = state.clone()
                    else:
                        buffer = state.to(
                            device=current.device,
                            dtype=current.dtype,
                        ).clone()
                    if current is None:
                        self.register_buffer(name, buffer, persistent=True)
                    else:
                        setattr(self, name, buffer)
                    state_dict.pop(key)
                    loaded_keys.append(key)
                    continue

                if current is None:
                    self.register_buffer(
                        name,
                        torch.empty_like(state),
                        persistent=True,
                    )
                elif current.shape != state.shape:
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
