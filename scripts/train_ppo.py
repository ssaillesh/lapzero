"""Fine-tune the behaviour cloning policy with PPO on one track and track the run in MLflow.

Every PPOConfig and RewardConfig field is a flag (e.g. --learning-rate, --critic-warmup-rollouts,
--collision-penalty). The centreline CSV lists x, y in metres in driving order (f1tenth_racetracks
format works); the env measures info["progress"] along it. The fine-tuned actor is registered in
the MLflow model registry as challenger (or champion if the registered model has none yet).

Usage (from the repo root):
    uv run python scripts/train_ppo.py --map tracks/track_02/map.yaml --start 0 0 0 \
        --centreline tracks/track_02/centreline.csv --track-id track_02
"""

from __future__ import annotations

import argparse
import dataclasses
import logging
from pathlib import Path
from typing import TypeVar

from lapzero.contract import LAPS_PER_EPISODE_TRAIN, MAX_EPISODE_S
from lapzero.sim.env import EnvConfig
from lapzero.sim.f1tenth import make_env
from lapzero.sim.reward import RacingReward, RewardConfig
from lapzero.sim.simulator import Pose
from lapzero.sim.track import load_centreline
from lapzero.training.ppo import PPOConfig, run_ppo_training
from lapzero.training.tracking import REGISTERED_MODEL, mlflow_run

_Config = TypeVar("_Config", PPOConfig, RewardConfig)


def _add_fields(parser: argparse.ArgumentParser, defaults: PPOConfig | RewardConfig) -> None:
    """Add a --flag for each field of a config dataclass, typed and defaulted from `defaults`."""
    for field in dataclasses.fields(defaults):
        default = getattr(defaults, field.name)
        flag = f"--{field.name.replace('_', '-')}"
        parser.add_argument(flag, type=type(default), default=default, help=f"default {default}")


def _from_args(cls: type[_Config], args: argparse.Namespace) -> _Config:
    """Build config dataclass `cls` from the parsed flags of its fields."""
    return cls(**{field.name: getattr(args, field.name) for field in dataclasses.fields(cls)})


def main(argv: list[str] | None = None) -> None:
    """Parse arguments, build the F1TENTH training env, open an MLflow run and fine-tune."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--map", type=Path, required=True, help="ROS map yaml of the track")
    parser.add_argument(
        "--start",
        type=float,
        nargs=3,
        required=True,
        metavar=("X", "Y", "THETA"),
        help="start pose in the map frame: metres, metres, rad",
    )
    parser.add_argument("--centreline", type=Path, required=True, help="centreline CSV")
    parser.add_argument("--track-id", required=True)
    parser.add_argument("--bc-weights", type=Path, default=Path("models/bc_v1.pt"))
    parser.add_argument("--output", type=Path, default=Path("models/ppo_v1.pt"))
    parser.add_argument("--laps", type=int, default=LAPS_PER_EPISODE_TRAIN, help="laps/episode")
    parser.add_argument("--max-episode-s", type=float, default=MAX_EPISODE_S)
    parser.add_argument(
        "--randomize-start",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="jitter the start pose each episode (training only)",
    )
    parser.add_argument("--tracking-uri", default=Path("mlruns").resolve().as_uri())
    parser.add_argument("--experiment", default="lapzero-ppo")
    parser.add_argument("--registered-model", default=REGISTERED_MODEL)
    _add_fields(parser, PPOConfig())
    _add_fields(parser, RewardConfig())
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    ppo_config = _from_args(PPOConfig, args)
    reward_config = _from_args(RewardConfig, args)
    env_config = EnvConfig(
        laps_per_episode=args.laps,
        max_episode_s=args.max_episode_s,
        randomize_start=args.randomize_start,
    )
    sim_env = make_env(
        args.map,
        Pose(*args.start),
        seed=ppo_config.seed,
        config=env_config,
        centreline=load_centreline(args.centreline),
    )
    env = RacingReward(sim_env, reward_config)
    try:
        with mlflow_run(
            tracking_uri=args.tracking_uri, experiment=args.experiment, run_name=args.output.stem
        ) as tracker:
            run_ppo_training(
                env,
                args.bc_weights,
                args.output,
                ppo_config,
                tracker,
                track_id=args.track_id,
                env_config=env_config,
                reward_config=reward_config,
                registered_model=args.registered_model,
            )
    finally:
        sim_env.close()


if __name__ == "__main__":
    main()
