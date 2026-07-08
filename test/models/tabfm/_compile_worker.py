import sys

import torch
from sdm.cache import Cache
from sdm.models.tabfm.attention import MultiheadAttentionBlock
from sdm.models.tabfm.model import TabFMCore


def _core(*, is_classifier: bool, chunk_columns: bool) -> TabFMCore:
    model = TabFMCore(
        embed_dim=8,
        max_classes=4,
        col_num_blocks=1,
        col_nhead=2,
        col_num_inds=4,
        row_num_blocks=1,
        row_nhead=2,
        row_num_cls=2,
        icl_num_blocks=1,
        icl_nhead=2,
        ff_factor=2,
        feature_group_size=3,
        num_freq=4,
        decoder_hidden=16,
        is_classifier=is_classifier,
    ).eval()
    model.cell_embedder.row_chunk_size = None
    model.row_interactor.row_chunk_size = None
    model.row_interactor_2.row_chunk_size = None
    model.col_embedder.col_chunk_size = 3 if chunk_columns else None
    model.col_embedder_2.col_chunk_size = 3 if chunk_columns else None
    for block in model.modules():
        if isinstance(block, MultiheadAttentionBlock):
            block.ffn_chunk_size = None
    return model


def _target(
    *,
    is_classifier: bool,
    batch_size: int,
    num_rows: int,
) -> torch.Tensor:
    if is_classifier:
        return torch.randint(0, 4, (batch_size, num_rows))
    return torch.randn(batch_size, num_rows)


def _check_core_compilation(
    *,
    is_classifier: bool,
    chunk_columns: bool,
) -> None:
    model = _core(
        is_classifier=is_classifier,
        chunk_columns=chunk_columns,
    )
    compiled = torch.compile(
        model,
        backend="eager",
        fullgraph=True,
        dynamic=True,
    )
    train_size = torch.tensor([4, 3])

    for num_rows in (6, 7):
        input = torch.randn(2, num_rows, 5)
        target = _target(
            is_classifier=is_classifier,
            batch_size=2,
            num_rows=num_rows,
        )
        expected = model(input, target, train_size)
        output = compiled(input, target, train_size)

        torch.testing.assert_close(output, expected)


def _check_cached_replay_compilation(
    *,
    is_classifier: bool,
    chunk_columns: bool,
) -> None:
    model = _core(
        is_classifier=is_classifier,
        chunk_columns=chunk_columns,
    )
    context = torch.randn(2, 4, 5)
    context_target = _target(
        is_classifier=is_classifier,
        batch_size=2,
        num_rows=4,
    )
    cache = Cache()
    model(
        context,
        context_target,
        torch.tensor([4, 3]),
        cache=cache,
    )
    cache.freeze()

    def replay(
        input: torch.Tensor,
        target: torch.Tensor,
        train_size: torch.Tensor,
    ) -> torch.Tensor:
        return model(input, target, train_size, cache=cache)

    compiled = torch.compile(
        replay,
        backend="eager",
        fullgraph=True,
        dynamic=True,
    )
    train_size = torch.zeros(2, dtype=torch.long)

    for num_rows in (3, 2):
        input = torch.randn(2, num_rows, 5)
        target = torch.full(
            (2, num_rows),
            -100,
            dtype=context_target.dtype,
        )
        expected = replay(input, target, train_size)
        output = compiled(input, target, train_size)

        torch.testing.assert_close(output, expected)


def main() -> None:
    """Run one isolated compile case selected by command-line arguments."""
    if len(sys.argv) != 4:
        raise ValueError(
            "usage: _compile_worker.py MODE IS_CLASSIFIER CHUNK_COLUMNS"
        )
    mode, is_classifier_arg, chunk_columns_arg = sys.argv[1:]
    is_classifier = is_classifier_arg == "true"
    chunk_columns = chunk_columns_arg == "true"
    if mode == "all":
        _check_core_compilation(
            is_classifier=False,
            chunk_columns=False,
        )
        _check_core_compilation(
            is_classifier=True,
            chunk_columns=True,
        )
        _check_cached_replay_compilation(
            is_classifier=True,
            chunk_columns=False,
        )
        return
    if mode == "core":
        _check_core_compilation(
            is_classifier=is_classifier,
            chunk_columns=chunk_columns,
        )
        return
    if mode == "cache":
        _check_cached_replay_compilation(
            is_classifier=is_classifier,
            chunk_columns=chunk_columns,
        )
        return
    raise ValueError(f"unsupported compile mode: {mode!r}")


if __name__ == "__main__":
    main()
