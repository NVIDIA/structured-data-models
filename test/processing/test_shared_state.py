import copy

from sdm.processing import SharedState


def test_shared_state_survives_deepcopy_by_reference() -> None:
    state = SharedState([1, 2, 3])

    assert copy.deepcopy(state) is state
    assert copy.copy(state) is state


def test_shared_state_value_is_shared_across_copies() -> None:
    state: SharedState[dict[str, tuple[str, ...]]] = SharedState({"vocab": ()})
    holder = {"state": state}

    clone = copy.deepcopy(holder)
    state.value = {"vocab": ("a",)}

    assert clone["state"].value == {"vocab": ("a",)}
