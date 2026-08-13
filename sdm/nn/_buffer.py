from collections.abc import Iterable, Iterator
from typing import Any

import torch
from torch import Tensor


class BufferList(torch.nn.Module):
    """Store tensors as indexed persistent PyTorch buffers.

    The tensors move with this module and are included in its
    :meth:`~torch.nn.Module.state_dict`.

    Args:
        buffers: Tensors to register in order.
    """

    def __init__(self, buffers: Iterable[Tensor] = ()) -> None:
        super().__init__()
        for index, buffer in enumerate(buffers):
            self.register_buffer(str(index), buffer)

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
        index = 0
        while isinstance(state_dict.get(f"{prefix}{index}"), Tensor):
            key = f"{prefix}{index}"
            self.register_buffer(str(index), state_dict.pop(key).clone())
            loaded_keys.append(key)
            index += 1

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
        return self.get_buffer(str(index))

    def __iter__(self) -> Iterator[Tensor]:
        return (self[index] for index in range(len(self)))

    def __len__(self) -> int:
        return len(self._buffers)
