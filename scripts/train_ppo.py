"""Fine-tune the behaviour cloning policy with PPO on one or more tracks, tracked in MLflow.

--tracks ID [ID ...] trains on several tracks at once, one env per track, each in its own worker
process (F1TENTH keeps its map in per-process state). Track ID is read from
<tracks-dir>/<ID>/<ID>.yaml and <tracks-dir>/<ID>/centreline.csv, and starts at the centreline's
first point facing the second. Alternatively --map, --centreline and --track-id (and optionally
--start) give one track explicitly.

Every PPOConfig and RewardConfig field is a flag (e.g. --learning-rate, --critic-warmup-rollouts,
--collision-penalty); --n-steps counts per env. The centreline CSV lists x, y in metres in driving
order (f1tenth_racetracks format works); the env measures info["progress"] along it. The
fine-tuned actor is registered in the MLflow model registry as challenger (or champion if the
registered model has none yet). --friction-range LOW HIGH draws the tyre friction mu uniformly
from that range every episode. --checkpoint-interval N also saves the actor every N timesteps to
<output dir>/checkpoints/<output stem>_step_<N>.pt.

Usage (from the repo root):
    uv run python scripts/train_ppo.py --tracks spielberg moscowraceway \
        --bc-weights models/ppo_v1.pt --output models/ppo_multitrack_v1.pt
"""

from __future__ import annotations

import argparse
import dataclasses
import logging
from pathlib import Path
from typing import TypeVar

from lapzero.contract import LAPS_PER_EPISODE_TRAIN, MAX_EPISODE_S
from lapzero.sim.env import EnvConfig
from lapzero.sim.reward import RewardConfig
from lapzero.sim.simulator import Pose
from lapzero.training.ppo import PPOConfig, run_ppo_training
from lapzero.training.tracking import REGISTERED_MODEL, mlflow_run
from lapzero.training.vec_env import TrackEnvFactory, make_track_vec_env, track_files

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
    parser.add_argument(
        "--tracks", nargs="+", metavar="ID", help="train on these tracks of --tracks-dir at once"
    )
    parser.add_argument("--tracks-dir", type=Path, default=Path("data/tracks"))
    parser.add_argument("--map", type=Path, help="one track: ROS map yaml (instead of --tracks)")
    parser.add_argument(
        "--start",
        type=float,
        nargs=3,
        metavar=("X", "Y", "THETA"),
        help="with --map: start pose in the map frame, metres, metres, rad (default: from the "
        "centreline)",
    )
    parser.add_argument("--centreline", type=Path, help="with --map: centreline CSV")
    parser.add_argument("--track-id", help="with --map: track name for logs and metrics")
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
    parser.add_argument(
        "--friction-range",
        type=float,
        nargs=2,
        metavar=("LOW", "HIGH"),
        help="draw the tyre-road friction mu uniformly from [LOW, HIGH] each episode (default off)",
    )
    parser.add_argument(
        "--checkpoint-interval",
        type=int,
        metavar="N",
        help="also save the actor every N timesteps (default: only the final actor)",
    )
    parser.add_argument("--tracking-uri", default=Path("mlruns").resolve().as_uri())
    parser.add_argument("--experiment", default="lapzero-ppo")
    parser.add_argument("--registered-model", default=REGISTERED_MODEL)
    _add_fields(parser, PPOConfig())
    _add_fields(parser, RewardConfig())
    args = parser.parse_args(argv)
    single_track = (args.map, args.start, args.centreline, args.track_id)
    if args.tracks is not None and any(value is not None for value in single_track):
        parser.error("--tracks cannot be combined with --map, --start, --centreline, --track-id")
    if args.tracks is None and None in (args.map, args.centreline, args.track_id):
        parser.error("give --tracks, or --map with --centreline and --track-id")

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    ppo_config = _from_args(PPOConfig, args)
    reward_config = _from_args(RewardConfig, args)
    env_config = EnvConfig(
        laps_per_episode=args.laps,
        max_episode_s=args.max_episode_s,
        randomize_start=args.randomize_start,
        friction_range=None if args.friction_range is None else tuple(args.friction_range),
    )
    if args.tracks is not None:
        tracks = [(track_id, *track_files(args.tracks_dir, track_id)) for track_id in args.tracks]
        start = None
    else:
        tracks = [(args.track_id, args.map, args.centreline)]
        start = None if args.start is None else Pose(*args.start)
    factories = [
        TrackEnvFactory(
            track_id,
            map_yaml,
            centreline,
            seed=ppo_config.seed + index,
            env_config=env_config,
            reward_config=reward_config,
            start_pose=start,
        )
        for index, (track_id, map_yaml, centreline) in enumerate(tracks)
    ]
    env = make_track_vec_env(factories)
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
                track_ids=[factory.track_id for factory in factories],
                env_config=env_config,
                reward_config=reward_config,
                registered_model=args.registered_model,
                checkpoint_interval=args.checkpoint_interval,
            )
    finally:
        env.close()


if __name__ == "__main__":
    main()
