"""Unit tests for lapzero/training/policy.py: network shapes, action bounds and checkpoints."""

from pathlib import Path

import numpy as np
import pytest
import torch
from hypothesis import given
from hypothesis import strategies as st
from hypothesis.extra.numpy import arrays
from torch import nn

from lapzero.contract import ContractViolationError
from lapzero.training.policy import BCPolicy, load_policy, major_version, save_policy

ANY_FINITE_OBS = arrays(np.float32, (5, 113), elements=st.floats(-1e6, 1e6, width=32))


def _policy(hidden_size: int = 32) -> BCPolicy:
    torch.manual_seed(0)
    return BCPolicy(hidden_size)


def test_a_batch_of_observations_maps_to_a_batch_of_actions() -> None:
    assert _policy()(torch.zeros(7, 113)).shape == (7, 2)


def test_a_single_observation_maps_to_a_single_action() -> None:
    assert _policy()(torch.zeros(113)).shape == (2,)


def test_network_has_three_linear_layers_sized_by_the_contract() -> None:
    layers = [module for module in _policy().modules() if isinstance(module, nn.Linear)]
    assert [(layer.in_features, layer.out_features) for layer in layers] == [
        (113, 32),
        (32, 32),
        (32, 2),
    ]


@given(ANY_FINITE_OBS)
def test_actions_stay_finite_and_inside_the_action_space(obs: np.ndarray) -> None:
    with torch.no_grad():
        actions = _policy()(torch.from_numpy(obs))
    assert torch.isfinite(actions).all()
    assert ((actions >= -1.0) & (actions <= 1.0)).all()


def test_saved_policy_reloads_in_eval_mode_with_identical_outputs(tmp_path: Path) -> None:
    policy = _policy(hidden_size=16)
    path = tmp_path / "models" / "bc.pt"
    save_policy(policy, path)
    loaded = load_policy(path)
    obs = torch.rand(4, 113)
    assert loaded.hidden_size == 16
    assert not loaded.training
    with torch.no_grad():
        torch.testing.assert_close(loaded(obs), policy(obs))


@pytest.mark.parametrize(("field", "value"), [("contract_version", "2.0.0"), ("obs_dim", 110)])
def test_loading_a_checkpoint_for_another_contract_raises(
    tmp_path: Path, field: str, value: object
) -> None:
    path = tmp_path / "bc.pt"
    save_policy(_policy(), path)
    checkpoint = torch.load(path, weights_only=True)
    checkpoint[field] = value
    torch.save(checkpoint, path)
    with pytest.raises(ContractViolationError, match="saved for"):
        load_policy(path)


def test_major_version_is_the_first_number() -> None:
    assert major_version("1.4.2") == 1
    assert major_version("12.0.0") == 12
