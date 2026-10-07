"""Inference-only ICL placement experiments for Kumo models.

Install after loading weights and before fitting. Do not call model.to() or
compile the model after installing placement. The existing Cache owns tensors
on their producing layer's device; never migrate the fitted cache wholesale.
"""

from collections.abc import Sequence
from typing import Any, Literal

import torch
from torch import Tensor, nn

from sdm.cache import Cache


def _place_layer(device: torch.device):
    def hook(module: nn.Module, args: tuple, kwargs: dict) -> tuple:
        query = kwargs["query"]
        kwargs["query"] = query.to(device)
        key_value = kwargs["key_value"]
        kwargs["key_value"] = key_value.to(device)
        # A destination on the previous stage cannot be used as an in-place
        # output by this stage. Let the block allocate its local output.
        out = kwargs.get("out")
        if out is not None and out.device != device:
            kwargs["out"] = None
        return args, kwargs

    return hook


class ICLPlacement(nn.Module):
    """Place a complete ICL stage or contiguous ICL layers across devices.

    ``stage`` keeps front-end work on the input device and the entire ICL stack
    on the last device. ``layers`` distributes contiguous blocks evenly, with
    label embedding on the first device and the output head on the last.
    Execution is sequential; this measures capacity, not pipeline throughput.
    """

    def __init__(
        self,
        block: nn.Module,
        devices: Sequence[str | torch.device],
        mode: Literal["stage", "layers"] = "stage",
    ) -> None:
        super().__init__()
        self.block = block
        self.mode = mode
        self.devices = tuple(torch.device(device) for device in devices)
        self.entry = self.devices[-1] if mode == "stage" else self.devices[0]
        self.handles = []
        if mode == "stage":
            block.to(self.entry)
            self._finish_placement()
            return

        block.to(self.entry)
        layers = block.layers
        for index, layer in enumerate(layers):
            device = self.devices[index * len(self.devices) // len(layers)]
            layer.to(device)
            self.handles.append(
                layer.register_forward_pre_hook(
                    _place_layer(device),
                    with_kwargs=True,
                )
            )
        last = self.devices[-1]
        block.norm.to(last)
        block.head.to(last)
        self.handles.append(
            block.norm.register_forward_pre_hook(
                lambda module, args: (args[0].to(last),),
            )
        )
        # Hierarchical TabICLv2 classification combines head outputs with
        # class IDs and single-class leaves on the entry device.
        self.handles.append(
            block.head.register_forward_hook(
                lambda module, args, output: output.to(self.entry),
            )
        )
        self._finish_placement()

    def _finish_placement(self) -> None:
        # Setup may run on different streams from the inference executor.
        # Synchronize once after moving weights; never in the forward path.
        for device in set(self.devices):
            if device.type == "cuda":
                torch.cuda.synchronize(device)

    def forward(
        self,
        x: Tensor,
        y: Tensor,
        *,
        cache: Cache | None = None,
        **kwargs: Any,
    ) -> Tensor:
        """Run placed inference and return outputs on the caller's device."""
        if torch.is_grad_enabled():
            raise RuntimeError("Placement experiments require inference_mode")
        if (
            self.mode == "layers"
            and cache is None
            and getattr(self.block, "kv_heads", None) is not None
        ):
            raise ValueError(
                "Layer placement with reduced query KV heads requires a cache"
            )
        output_device = x.device
        output = self.block(
            x=x.to(self.entry),
            y=y.to(self.entry),
            cache=cache,
            **kwargs,
        ).to(output_device)
        if output_device.type == "cuda":
            # Fit can return an empty prediction, whose zero-byte copy need
            # not wait for remote cache writes. Join every stage explicitly.
            caller = torch.cuda.current_stream(output_device)
            for device in self.devices:
                if device != output_device:
                    done = torch.cuda.Event()
                    done.record(torch.cuda.current_stream(device))
                    caller.wait_event(done)
        return output


def install_icl_placement(
    model: nn.Module,
    devices: Sequence[str | torch.device],
    mode: Literal["stage", "layers"] = "stage",
) -> nn.Module:
    """Modify a freshly loaded Kumo inner model in place.

    Load on devices[0], install placement, then call the inner forward
    with an explicit Cache. A single device is the matched adapter baseline.
    Public fit/predict currently migrates the entire cache to the input device
    and requires a separate cache-placement integration before it can be used.
    """
    if hasattr(model, "models"):
        raise ValueError("Pass model.models[task] and use cached forwards")
    model.icl_block = ICLPlacement(model.icl_block, devices, mode)
    return model


def cache_bytes_by_device(cache: Cache) -> dict[str, int]:
    """Count unique cache storage bytes by device, including nested caches."""
    seen: set[tuple[str, int]] = set()
    totals: dict[str, int] = {}
    for tensor in cache._tensors():
        storage = tensor.untyped_storage()
        device = str(tensor.device)
        key = (device, storage.data_ptr())
        if key not in seen:
            seen.add(key)
            totals[device] = totals.get(device, 0) + storage.nbytes()
    return totals
