"""The policy network trained by behaviour cloning, and its .pt checkpoint format."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch import nn

from lapzero.contract import ACT_DIM, CONTRACT_VERSION, OBS_DIM, ContractViolationError


def major_version(version: str) -> int:
    """Return the major number of a semantic version string such as "1.0.0"."""
    return int(version.split(".", 1)[0])


class BCPolicy(nn.Module):
    """Three-layer MLP from a contract observation to a contract action in [-1, 1]."""

    def __init__(self, hidden_size: int = 256) -> None:
        """Build the network.

        Args:
            hidden_size: Units in each of the two hidden layers.
        """
        super().__init__()
        self.hidden_size = hidden_size
        self.net = nn.Sequential(
            nn.Linear(OBS_DIM, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, ACT_DIM),
            nn.Tanh(),
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """Map (..., OBS_DIM) observations to (..., ACT_DIM) normalized actions."""
        actions: torch.Tensor = self.net(obs)
        return actions


def save_policy(policy: BCPolicy, path: Path) -> None:
    """Save weights plus the shape and contract metadata needed to load them (.pt file)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "state_dict": policy.state_dict(),
        "hidden_size": policy.hidden_size,
        "obs_dim": OBS_DIM,
        "act_dim": ACT_DIM,
        "contract_version": CONTRACT_VERSION,
    }
    torch.save(checkpoint, path)


def load_policy(path: Path) -> BCPolicy:
    """Load a policy written by save_policy, in eval mode.

    Raises:
        ContractViolationError: If it was saved for other observation/action sizes or another
            contract major version.
    """
    # torch.load is typed as returning Any; save_policy always writes this dict.
    checkpoint: dict[str, Any] = torch.load(path, weights_only=True)
    shapes = (checkpoint["obs_dim"], checkpoint["act_dim"])
    version = str(checkpoint["contract_version"])
    if shapes != (OBS_DIM, ACT_DIM) or major_version(version) != major_version(CONTRACT_VERSION):
        raise ContractViolationError(
            f"{path.name} was saved for (obs, act) {shapes} under contract {version}, "
            f"need {(OBS_DIM, ACT_DIM)} under {CONTRACT_VERSION}"
        )
    policy = BCPolicy(int(checkpoint["hidden_size"]))
    policy.load_state_dict(checkpoint["state_dict"])
    policy.eval()
    return policy
