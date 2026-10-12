"""PPO fine-tuning (Stable-Baselines3) warm-started from a behaviour cloning policy.

The PPO actor has the BCPolicy architecture (OBS_DIM -> hidden -> hidden -> ACT_DIM, ReLU, tanh
on the action mean), so BC weights load into it unchanged and the deterministic PPO policy starts
out acting exactly like the BC policy. Only the critic and the action noise start fresh, and a
critic warm-up fits the critic to the BC policy's returns before PPO may change the actor. The
fine-tuned actor is exported back to a BCPolicy checkpoint, so BC and PPO models load the same way.
"""

from __future__ import annotations

import dataclasses
import logging
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

import gymnasium as gym
import numpy as np
import torch
from numpy.typing import NDArray
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.logger import Logger
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.type_aliases import Schedule
from torch import nn

from lapzero.contract import CONTRACT_VERSION
from lapzero.sim.env import EnvConfig
from lapzero.sim.reward import REWARD_VERSION, RewardConfig
from lapzero.training.callbacks import RaceMetricsCallback, TrackerOutputFormat
from lapzero.training.policy import BCPolicy, load_policy, save_policy
from lapzero.training.tracking import (
    REGISTERED_MODEL,
    ParamValue,
    RunTracker,
    contract_params,
    git_commit,
)

logger = logging.getLogger(__name__)

BC_TO_PPO_LAYERS: Final = {
    "net.0": "mlp_extractor.policy_net.0",
    "net.2": "mlp_extractor.policy_net.2",
    "net.4": "action_net.0",
}
"""BCPolicy layer name -> WarmStartPolicy layer name holding the same weights and biases."""


@dataclass(frozen=True, slots=True)
class PPOConfig:
    """PPO hyperparameters, set so fine-tuning does not wreck the BC policy before it improves.

    The critic starts untrained, so its first advantage estimates are noise. The critic warm-up
    trains it alone first; after that, small, clipped, KL-capped updates keep the actor close to
    the BC policy while both improve.

    Attributes:
        total_timesteps: Control steps of experience to collect.
        learning_rate: Adam learning rate; a tenth of SB3's 3e-4 default.
        n_steps: Control steps per rollout before each update.
        batch_size: Minibatch size, at least 2.
        n_epochs: Passes over each rollout; half SB3's default to limit drift per update.
        gamma: Discount per control step; 0.99 at 25 Hz looks about 4 s ahead.
        gae_lambda: GAE bias/variance trade-off.
        clip_range: PPO probability-ratio clip; tighter than SB3's 0.2.
        ent_coef: Entropy bonus. 0 because exploration comes from log_std_init, and a bonus
            would only widen the action noise around a policy that can already drive.
        vf_coef: Value loss weight.
        max_grad_norm: Gradient norm clip.
        target_kl: An update stops its remaining epochs once the approximate KL divergence
            from the rollout policy exceeds 1.5x this.
        log_std_init: Initial log standard deviation of the Gaussian action noise, in
            normalized action units. -1.0 gives std 0.37; SB3's 0.0 (std 1.0) would swamp the
            BC actions.
        critic_warmup_rollouts: Rollouts at the start whose updates train only the critic
            (actor frozen), so PPO's first actor updates use a fitted value baseline. 0
            disables the warm-up; it must leave timesteps for the actor to train.
        seed: Seeds PPO, torch, NumPy and the env resets.
    """

    total_timesteps: int = 1_000_000
    learning_rate: float = 3e-5
    n_steps: int = 2048
    batch_size: int = 256
    n_epochs: int = 5
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_range: float = 0.1
    ent_coef: float = 0.0
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    target_kl: float = 0.02
    log_std_init: float = -1.0
    critic_warmup_rollouts: int = 5
    seed: int = 0

    def __post_init__(self) -> None:
        minimums = {"total_timesteps": 1, "n_steps": 2, "batch_size": 2, "n_epochs": 1}
        for name, minimum in minimums.items():
            value = getattr(self, name)
            if value < minimum:
                raise ValueError(f"PPOConfig.{name} must be >= {minimum}, got {value}")
        positive = ("learning_rate", "clip_range", "vf_coef", "max_grad_norm", "target_kl")
        for name in positive:
            value = getattr(self, name)
            if not (math.isfinite(value) and value > 0.0):
                raise ValueError(f"PPOConfig.{name} must be finite and > 0, got {value}")
        for name in ("gamma", "gae_lambda"):
            if not 0.0 < getattr(self, name) <= 1.0:
                raise ValueError(f"PPOConfig.{name} must be in (0, 1], got {getattr(self, name)}")
        if not (math.isfinite(self.ent_coef) and self.ent_coef >= 0.0):
            raise ValueError(f"PPOConfig.ent_coef must be finite and >= 0, got {self.ent_coef}")
        if not math.isfinite(self.log_std_init):
            raise ValueError(f"PPOConfig.log_std_init must be finite, got {self.log_std_init}")
        if self.critic_warmup_rollouts < 0:
            raise ValueError(
                f"PPOConfig.critic_warmup_rollouts must be >= 0, got {self.critic_warmup_rollouts}"
            )
        if self.critic_warmup_timesteps >= self.total_timesteps:
            raise ValueError(
                f"PPOConfig.critic_warmup_rollouts ({self.critic_warmup_rollouts} x n_steps "
                f"{self.n_steps}) uses all {self.total_timesteps} total_timesteps"
            )

    @property
    def critic_warmup_timesteps(self) -> int:
        """Timesteps collected during the critic warm-up."""
        return self.critic_warmup_rollouts * self.n_steps


class WarmStartPolicy(ActorCriticPolicy):
    """SB3 actor-critic whose action mean goes through tanh, like BCPolicy's output layer.

    Actions are sampled around the tanh mean and clipped to the action space by SB3, so the
    deterministic action equals the output of a BCPolicy holding the same actor weights.
    Build it with net_arch {"pi": [h, h], "vf": [h, h]} and activation_fn nn.ReLU.
    """

    def _build(self, lr_schedule: Schedule) -> None:
        """Build the networks and optimizer as SB3 does, then put tanh after the mean layer."""
        super()._build(lr_schedule)
        # The Sequential reuses the same Linear module, so the optimizer built above still
        # holds its parameters.
        self.action_net = nn.Sequential(self.action_net, nn.Tanh())


def load_bc_weights(policy: ActorCriticPolicy, bc: BCPolicy) -> None:
    """Copy a BC policy's weights into a WarmStartPolicy's actor.

    The critic (value network) and the action noise (log_std) are left unchanged.

    Args:
        policy: PPO policy built with the BC policy's hidden size.
        bc: Trained behaviour cloning policy.

    Raises:
        ValueError: If a PPO actor layer is missing or its shape differs from the BC layer.
    """
    target = policy.state_dict()
    actor: dict[str, torch.Tensor] = {}
    for bc_key, weights in bc.state_dict().items():
        bc_layer, param = bc_key.rsplit(".", 1)
        key = f"{BC_TO_PPO_LAYERS[bc_layer]}.{param}"
        found = target.get(key)
        if found is None or found.shape != weights.shape:
            shape = None if found is None else tuple(found.shape)
            raise ValueError(
                f"PPO policy {key} has shape {shape}, BC {bc_key} has {tuple(weights.shape)}"
            )
        actor[key] = weights
    policy.load_state_dict(actor, strict=False)


def export_actor(policy: ActorCriticPolicy) -> BCPolicy:
    """Copy a WarmStartPolicy's actor into a new BCPolicy (its deterministic policy).

    Args:
        policy: Policy built as WarmStartPolicy.

    Returns:
        A BCPolicy in eval mode with the actor's weights.
    """
    state = policy.state_dict()
    hidden_size = int(state[f"{BC_TO_PPO_LAYERS['net.0']}.weight"].shape[0])
    actor = BCPolicy(hidden_size)
    actor.load_state_dict(
        {
            f"{bc_layer}.{param}": state[f"{ppo_layer}.{param}"]
            for bc_layer, ppo_layer in BC_TO_PPO_LAYERS.items()
            for param in ("weight", "bias")
        }
    )
    actor.eval()
    return actor


def build_ppo(
    env: gym.Env[NDArray[np.float32], NDArray[np.float32]], bc: BCPolicy, config: PPOConfig
) -> PPO:
    """Create a seeded CPU PPO model on `env` whose actor starts as `bc`.

    Args:
        env: Training env with the contract observation and action spaces.
        bc: Policy whose weights initialise the actor; its hidden size sets both networks'.
        config: PPO hyperparameters and seed.

    Returns:
        The untrained PPO model.
    """
    logger.info("building PPO with seed=%d, warm-started from BC", config.seed)
    hidden = [bc.hidden_size, bc.hidden_size]
    policy_kwargs: dict[str, Any] = {
        "net_arch": {"pi": hidden, "vf": hidden},
        "activation_fn": nn.ReLU,
        "log_std_init": config.log_std_init,
    }
    model = PPO(
        WarmStartPolicy,
        env,
        learning_rate=config.learning_rate,
        n_steps=config.n_steps,
        batch_size=config.batch_size,
        n_epochs=config.n_epochs,
        gamma=config.gamma,
        gae_lambda=config.gae_lambda,
        clip_range=config.clip_range,
        ent_coef=config.ent_coef,
        vf_coef=config.vf_coef,
        max_grad_norm=config.max_grad_norm,
        target_kl=config.target_kl,
        policy_kwargs=policy_kwargs,
        seed=config.seed,
        device="cpu",  # small MLPs train faster on CPU, and CPU keeps seeded runs reproducible
    )
    load_bc_weights(model.policy, bc)
    return model


def set_actor_trainable(policy: ActorCriticPolicy, trainable: bool) -> None:
    """Freeze or unfreeze the actor: its hidden layers, mean layer and action noise (log_std).

    Frozen parameters get no gradients, so PPO updates train only the critic, which shares no
    parameters with the actor (separate pi and vf networks, parameter-free features extractor).

    Args:
        policy: Policy built as WarmStartPolicy.
        trainable: False to freeze the actor, True to let PPO update it again.
    """
    actor = (
        *policy.mlp_extractor.policy_net.parameters(),
        *policy.action_net.parameters(),
        policy.log_std,
    )
    for param in actor:
        param.requires_grad_(trainable)


def learn_with_critic_warmup(model: PPO, config: PPOConfig, callback: BaseCallback) -> None:
    """Train for config.total_timesteps; the first critic_warmup_rollouts update only the critic.

    During the warm-up the BC policy (with its action noise) collects rollouts and PPO's value
    loss fits the critic to their returns while the actor stays frozen. Training then continues
    from the same env state and timestep count with actor and critic both updating.

    Args:
        model: PPO model built by build_ppo.
        config: Total and warm-up timesteps.
        callback: Receives every step of both phases.
    """
    warmup = config.critic_warmup_timesteps
    if warmup:
        logger.info("critic warm-up: %d timesteps with the actor frozen", warmup)
        set_actor_trainable(model.policy, False)
        model.learn(warmup, callback=callback)
        set_actor_trainable(model.policy, True)
        logger.info("critic warm-up done at step %d; actor unfrozen", model.num_timesteps)
    remaining = config.total_timesteps - model.num_timesteps
    model.learn(remaining, callback=callback, reset_num_timesteps=False)


def _prefixed(prefix: str, config: PPOConfig | EnvConfig | RewardConfig) -> dict[str, ParamValue]:
    """Return a config's fields as params named "<prefix>.<field>"."""
    return {f"{prefix}.{name}": value for name, value in dataclasses.asdict(config).items()}


def run_ppo_training(
    env: gym.Env[NDArray[np.float32], NDArray[np.float32]],
    bc_path: Path,
    output_path: Path,
    config: PPOConfig,
    tracker: RunTracker,
    *,
    track_id: str,
    env_config: EnvConfig,
    reward_config: RewardConfig,
    registered_model: str = REGISTERED_MODEL,
) -> BCPolicy:
    """Fine-tune a BC policy with PPO, save the actor and record the run (CLAUDE.md section 9).

    The actor is registered as a new version of `registered_model`, tagged algorithm=ppo; it
    becomes the challenger, or the champion if the model has none yet.

    Args:
        env: Training env, normally a RacingReward around a LapZeroEnv.
        bc_path: BC checkpoint written by save_policy, e.g. models/bc_v1.pt.
        output_path: Where to write the fine-tuned actor, e.g. models/ppo_v1.pt.
        config: PPO hyperparameters and seed.
        tracker: Run tracker (an MLflow run in production).
        track_id: Track the env drives, logged as track_ids.
        env_config: The env's settings, logged as env.* params.
        reward_config: The env's reward weights, logged as reward.* params.
        registered_model: Model registry name to add the version to.

    Returns:
        The fine-tuned deterministic actor, as saved to output_path.

    Raises:
        ContractViolationError: If the BC checkpoint was saved under another contract.
    """
    bc = load_policy(bc_path)
    tracker.log_params(
        {
            **contract_params(),
            **_prefixed("ppo", config),
            **_prefixed("env", env_config),
            **_prefixed("reward", reward_config),
            "contract_version": CONTRACT_VERSION,
            "reward_version": REWARD_VERSION,
            "seed": config.seed,
            "track_ids": track_id,
            "warm_start": bc_path.name,
            "hidden_size": bc.hidden_size,
            "git_commit": git_commit(),
        }
    )
    model = build_ppo(env, bc, config)
    model.set_logger(Logger(folder=None, output_formats=[TrackerOutputFormat(tracker)]))
    learn_with_critic_warmup(model, config, RaceMetricsCallback(tracker))
    actor = export_actor(model.policy)
    save_policy(actor, output_path)
    logger.info("saved PPO policy to %s", output_path)
    tags = {
        "algorithm": "ppo",
        "contract_version": CONTRACT_VERSION,
        "reward_version": REWARD_VERSION,
    }
    tracker.register_policy(output_path, registered_model, tags)
    return actor
