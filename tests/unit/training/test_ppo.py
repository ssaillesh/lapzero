"""Unit tests for lapzero/training/ppo.py: config checks, BC warm start, critic warm-up, actor
export, checkpoints and run logging, on one env or a two-track VecEnv.

PPO trains on CPU with tiny networks, on a RacingReward-wrapped LapZeroEnv around the
FakeSimulator; episodes are 5 control steps long and rollouts 16 steps per env.
"""

from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv

from lapzero.contract import OBS_HIGH, OBS_LOW
from lapzero.sim.env import EnvConfig, LapZeroEnv
from lapzero.sim.reward import REWARD_VERSION, RacingReward, RewardConfig
from lapzero.sim.track import Centreline
from lapzero.training.policy import BCPolicy, load_policy, save_policy
from lapzero.training.ppo import (
    PPOConfig,
    build_ppo,
    export_actor,
    learn_with_critic_warmup,
    load_bc_weights,
    run_ppo_training,
    set_actor_trainable,
)
from tests.conftest import START_POSE, FakeSimulator
from tests.unit.training.test_bc import FakeTracker

TINY = PPOConfig(total_timesteps=48, n_steps=16, batch_size=8, n_epochs=2, critic_warmup_rollouts=1)
ENV_CONFIG = EnvConfig(max_episode_s=0.2)  # 5 control steps per episode
SQUARE = Centreline(np.array([[-5.0, -5.0], [5.0, -5.0], [5.0, 5.0], [-5.0, 5.0]]))


@pytest.fixture
def env(fake_sim: FakeSimulator) -> RacingReward:
    return RacingReward(LapZeroEnv(fake_sim, START_POSE, config=ENV_CONFIG, centreline=SQUARE))


def _bc(hidden_size: int = 16) -> BCPolicy:
    torch.manual_seed(1)
    policy = BCPolicy(hidden_size)
    policy.eval()
    return policy


def _observations(count: int = 32) -> np.ndarray:
    rng = np.random.default_rng(0)
    return rng.uniform(OBS_LOW, OBS_HIGH, size=(count, 113)).astype(np.float32)


ACTOR_PREFIXES = ("mlp_extractor.policy_net.", "action_net.", "log_std")
CRITIC_PREFIXES = ("mlp_extractor.value_net.", "value_net.")


class RolloutSnapshots(BaseCallback):
    """Copies the actor's and the critic's weights at the start of every rollout."""

    def __init__(self) -> None:
        super().__init__()
        self.actors: list[dict[str, torch.Tensor]] = []
        self.critics: list[dict[str, torch.Tensor]] = []

    def _on_rollout_start(self) -> None:
        state = self.model.policy.state_dict()
        for prefixes, snapshots in ((ACTOR_PREFIXES, self.actors), (CRITIC_PREFIXES, self.critics)):
            snapshots.append({k: v.clone() for k, v in state.items() if k.startswith(prefixes)})

    def _on_step(self) -> bool:
        return True


def _equal(first: dict[str, torch.Tensor], second: dict[str, torch.Tensor]) -> bool:
    return first.keys() == second.keys() and all(torch.equal(first[k], second[k]) for k in first)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("total_timesteps", 0),
        ("n_steps", 1),
        ("batch_size", 1),
        ("n_epochs", 0),
        ("learning_rate", 0.0),
        ("learning_rate", float("nan")),
        ("clip_range", -0.1),
        ("vf_coef", float("inf")),
        ("max_grad_norm", 0.0),
        ("target_kl", 0.0),
        ("gamma", 0.0),
        ("gamma", 1.5),
        ("gae_lambda", 0.0),
        ("ent_coef", -0.01),
        ("ent_coef", float("nan")),
        ("log_std_init", float("inf")),
        ("critic_warmup_rollouts", -1),
    ],
)
def test_invalid_hyperparameters_are_rejected(field: str, value: Any) -> None:
    with pytest.raises(ValueError, match=field):
        PPOConfig(**{field: value})


def test_default_hyperparameters_are_conservative_for_warm_starting() -> None:
    config = PPOConfig()
    assert config.learning_rate < 3e-4  # below SB3's default
    assert config.clip_range < 0.2  # tighter than SB3's default
    assert config.target_kl > 0.0
    assert config.ent_coef == 0.0
    assert np.exp(config.log_std_init) < 0.5  # action noise well inside the [-1, 1] range
    assert 0 < config.critic_warmup_timesteps < config.total_timesteps


def test_a_critic_warmup_that_leaves_no_timesteps_for_the_actor_is_rejected() -> None:
    with pytest.raises(ValueError, match="critic_warmup_rollouts"):
        PPOConfig(total_timesteps=64, n_steps=16, critic_warmup_rollouts=4)


def test_freezing_the_actor_stops_gradients_to_it_but_not_to_the_critic(
    env: RacingReward,
) -> None:
    policy = build_ppo(env, _bc(), TINY).policy
    set_actor_trainable(policy, False)
    trainable = {name: param.requires_grad for name, param in policy.named_parameters()}
    assert not any(v for k, v in trainable.items() if k.startswith(ACTOR_PREFIXES))
    assert all(v for k, v in trainable.items() if k.startswith(CRITIC_PREFIXES))
    set_actor_trainable(policy, True)
    assert all(param.requires_grad for param in policy.parameters())


def test_the_critic_warmup_trains_only_the_critic_before_ppo_updates_the_actor(
    env: RacingReward,
) -> None:
    model = build_ppo(env, _bc(), TINY)  # warm-up of 1 rollout, then 2 more
    snapshots = RolloutSnapshots()
    learn_with_critic_warmup(model, TINY, snapshots)
    assert model.num_timesteps == 48
    assert len(snapshots.actors) == 3
    assert _equal(snapshots.actors[1], snapshots.actors[0]), "warm-up update moved the actor"
    assert not _equal(snapshots.critics[1], snapshots.critics[0]), "warm-up left the critic"
    assert not _equal(snapshots.actors[2], snapshots.actors[1]), "PPO never updated the actor"
    assert all(param.requires_grad for param in model.policy.parameters())


def test_without_a_critic_warmup_ppo_updates_the_actor_from_the_first_rollout(
    env: RacingReward,
) -> None:
    config = PPOConfig(total_timesteps=32, n_steps=16, batch_size=8, critic_warmup_rollouts=0)
    model = build_ppo(env, _bc(), config)
    snapshots = RolloutSnapshots()
    learn_with_critic_warmup(model, config, snapshots)
    assert model.num_timesteps == 32
    assert not _equal(snapshots.actors[1], snapshots.actors[0])


def test_warm_started_ppo_acts_exactly_like_the_bc_policy(env: RacingReward) -> None:
    bc = _bc()
    model = build_ppo(env, bc, TINY)
    obs = _observations()
    actions, _ = model.predict(obs, deterministic=True)
    with torch.no_grad():
        expected = bc(torch.from_numpy(obs)).numpy()
    np.testing.assert_allclose(actions, expected, atol=1e-6)


def test_warm_start_sets_the_action_noise_from_log_std_init(env: RacingReward) -> None:
    model = build_ppo(env, _bc(), PPOConfig(log_std_init=-1.5))
    torch.testing.assert_close(model.policy.log_std.detach(), torch.full((2,), -1.5))


def test_loading_bc_weights_leaves_the_critic_and_action_noise_untouched(
    env: RacingReward,
) -> None:
    model = build_ppo(env, _bc(), TINY)
    before = {
        key: value.clone()
        for key, value in model.policy.state_dict().items()
        if "value" in key or key == "log_std"
    }
    torch.manual_seed(2)
    load_bc_weights(model.policy, BCPolicy(16))
    after = model.policy.state_dict()
    assert before
    for key, value in before.items():
        torch.testing.assert_close(after[key], value)


def test_exported_actor_holds_exactly_the_bc_weights_before_training(env: RacingReward) -> None:
    bc = _bc()
    actor = export_actor(build_ppo(env, bc, TINY).policy)
    assert actor.hidden_size == 16
    assert not actor.training
    for key, value in bc.state_dict().items():
        torch.testing.assert_close(actor.state_dict()[key], value, rtol=0.0, atol=0.0)


def test_bc_weights_of_another_hidden_size_are_rejected(env: RacingReward) -> None:
    model = build_ppo(env, _bc(16), TINY)
    with pytest.raises(ValueError, match="shape"):
        load_bc_weights(model.policy, BCPolicy(8))


def test_bc_weights_are_rejected_by_a_policy_without_the_tanh_mean_layer(
    env: RacingReward,
) -> None:
    plain = PPO("MlpPolicy", env, policy_kwargs={"net_arch": [16, 16]}, device="cpu")
    with pytest.raises(ValueError, match=r"action_net\.0\.weight has shape None"):
        load_bc_weights(plain.policy, _bc())


def test_run_ppo_training_saves_a_loadable_fine_tuned_actor_and_logs_the_run(
    env: RacingReward, tmp_path: Path
) -> None:
    bc = _bc()
    bc_path, output = tmp_path / "bc_v1.pt", tmp_path / "models" / "ppo_v1.pt"
    save_policy(bc, bc_path)
    tracker = FakeTracker()

    actor = run_ppo_training(
        env,
        bc_path,
        output,
        TINY,
        tracker,
        track_ids=["track_a"],
        env_config=ENV_CONFIG,
        reward_config=RewardConfig(),
    )

    obs = torch.from_numpy(_observations())
    with torch.no_grad():
        torch.testing.assert_close(load_policy(output)(obs), actor(obs))
        assert not torch.equal(actor(obs), bc(obs)), "PPO updates should move the actor"
    ppo_tags = {"algorithm": "ppo", "contract_version": "1.1.0", "reward_version": REWARD_VERSION}
    assert tracker.registered == [(output, "lapzero-policy", ppo_tags)]
    assert tracker.artifacts == []  # weights go to the model registry, not raw run artifacts
    params = tracker.params
    assert params["contract_version"] == "1.1.0"
    assert params["contract.OBS_DIM"] == 113
    assert params["reward_version"] == REWARD_VERSION
    assert params["seed"] == 0
    assert params["track_ids"] == "track_a"
    assert params["warm_start"] == "bc_v1.pt"
    assert params["hidden_size"] == 16
    assert params["ppo.learning_rate"] == TINY.learning_rate
    assert params["ppo.clip_range"] == TINY.clip_range
    assert params["ppo.critic_warmup_rollouts"] == 1
    assert params["env.max_episode_s"] == 0.2
    assert params["reward.collision_penalty"] == RewardConfig().collision_penalty
    assert "git_commit" in params
    keys = {name for name, _, _ in tracker.metrics}
    assert {"train/approx_kl", "train/value_loss", "train/clip_fraction"} <= keys
    episode_lengths = [value for name, value, _ in tracker.metrics if name == "race/episode_length"]
    assert episode_lengths == [5.0] * 9  # 48 control steps hold nine whole 5-step episodes
    # Timesteps keep counting from the warm-up into the PPO phase.
    assert tracker.steps("race/episode_length") == list(range(5, 46, 5))
    assert all(np.isfinite(value) for _, value, _ in tracker.metrics)


def _two_track_vec_env() -> DummyVecEnv:
    """Two monitored FakeSimulator envs in one process, shaped like make_track_vec_env's."""

    def track_env() -> Monitor[np.ndarray, np.ndarray]:
        sim_env = LapZeroEnv(FakeSimulator(), START_POSE, config=ENV_CONFIG, centreline=SQUARE)
        return Monitor(RacingReward(sim_env))

    return DummyVecEnv([track_env, track_env])


def test_with_several_envs_the_critic_warmup_still_lasts_its_number_of_rollouts() -> None:
    config = PPOConfig(
        total_timesteps=96, n_steps=16, batch_size=8, n_epochs=2, critic_warmup_rollouts=2
    )
    model = build_ppo(_two_track_vec_env(), _bc(), config)  # rollouts of 2 envs x 16 steps
    snapshots = RolloutSnapshots()
    learn_with_critic_warmup(model, config, snapshots)
    assert model.num_timesteps == 96
    assert len(snapshots.actors) == 3
    assert _equal(snapshots.actors[2], snapshots.actors[0]), "warm-up ended before 2 rollouts"
    assert not _equal(snapshots.critics[2], snapshots.critics[0]), "warm-up left the critic"
    state = model.policy.state_dict()
    final_actor = {k: v for k, v in state.items() if k.startswith(ACTOR_PREFIXES)}
    assert not _equal(final_actor, snapshots.actors[0]), "PPO never updated the actor"


def test_a_critic_warmup_that_several_envs_stretch_over_all_timesteps_is_rejected() -> None:
    config = PPOConfig(total_timesteps=48, n_steps=16, batch_size=8, critic_warmup_rollouts=2)
    model = build_ppo(_two_track_vec_env(), _bc(), config)  # fine with one env: 32 < 48
    with pytest.raises(ValueError, match="2 rollouts x 2 envs"):
        learn_with_critic_warmup(model, config, RolloutSnapshots())


def test_run_ppo_training_on_two_tracks_logs_every_race_metric_per_track(tmp_path: Path) -> None:
    bc_path = tmp_path / "bc_v1.pt"
    save_policy(_bc(), bc_path)
    tracker = FakeTracker()

    run_ppo_training(
        _two_track_vec_env(),
        bc_path,
        tmp_path / "ppo.pt",
        TINY,
        tracker,
        track_ids=["track_a", "track_b"],
        env_config=ENV_CONFIG,
        reward_config=RewardConfig(),
    )

    assert tracker.params["track_ids"] == "track_a,track_b"
    assert tracker.params["n_envs"] == 2
    assert "checkpoint_interval" not in tracker.params

    def values(key: str) -> list[float]:
        return [value for name, value, _ in tracker.metrics if name == key]

    # 2 envs x 32 steps: each track drives six whole 5-step episodes.
    assert values("race/track_a/episode_length") == [5.0] * 6
    assert values("race/track_b/episode_length") == [5.0] * 6
    for name in ("episode_length", "episode_reward", "episode_collision"):
        per_track = values(f"race/track_a/{name}") + values(f"race/track_b/{name}")
        assert sorted(values(f"race/{name}")) == sorted(per_track)


def test_run_ppo_training_needs_one_track_id_per_env(env: RacingReward, tmp_path: Path) -> None:
    bc_path = tmp_path / "bc_v1.pt"
    save_policy(_bc(), bc_path)
    tracker = FakeTracker()
    with pytest.raises(ValueError, match="2 track_ids for 1 envs"):
        run_ppo_training(
            env,
            bc_path,
            tmp_path / "ppo.pt",
            TINY,
            tracker,
            track_ids=["track_a", "track_b"],
            env_config=ENV_CONFIG,
            reward_config=RewardConfig(),
        )
    assert tracker.params == {}


def test_run_ppo_training_saves_the_actor_at_every_checkpoint_interval(
    env: RacingReward, tmp_path: Path
) -> None:
    bc = _bc()
    bc_path, output = tmp_path / "bc_v1.pt", tmp_path / "models" / "ppo_v1.pt"
    save_policy(bc, bc_path)
    tracker = FakeTracker()

    actor = run_ppo_training(
        env,
        bc_path,
        output,
        TINY,
        tracker,
        track_ids=["track_a"],
        env_config=ENV_CONFIG,
        reward_config=RewardConfig(),
        checkpoint_interval=16,
    )

    assert tracker.params["checkpoint_interval"] == 16
    directory = tmp_path / "models" / "checkpoints"
    names = ["ppo_v1_step_16.pt", "ppo_v1_step_32.pt", "ppo_v1_step_48.pt"]
    assert sorted(path.name for path in directory.iterdir()) == names
    obs = torch.from_numpy(_observations())
    with torch.no_grad():
        # Step 16 ends the warm-up rollout, before any update: still the BC actor.
        torch.testing.assert_close(load_policy(directory / names[0])(obs), bc(obs))
        # Step 48 ends the last rollout: PPO has updated the actor once, the final update not yet.
        last = load_policy(directory / names[2])(obs)
        assert not torch.equal(last, bc(obs))
        assert not torch.equal(last, actor(obs))
