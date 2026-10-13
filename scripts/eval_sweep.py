"""Sweep a policy over tyre-road friction on one track: completion, lap time and crash points.

The policy comes from the MLflow model registry (--alias, default champion) or a local
save_policy checkpoint (--checkpoint) and always drives its deterministic action. Each friction
level in --frictions LOW HIGH STEP (default 0.50 to 1.05 in 0.05 steps) runs --episodes episodes
seeded --seed + i; start jitter (--randomize-start, on by default) gives each seed a different
start pose, the same at every level. Episodes start from --start, or by default the first
centreline point facing the second.

Writes sweep.json and one directory of traces per level to <out>/<track_id>/<sweep_id>/
(default out: data/eval/friction_sweep). Sweeps compared by the release gate must use the same
settings.

Usage (from the repo root):
    .venv/bin/python scripts/eval_sweep.py --map data/tracks/oschersleben/oschersleben.yaml \
        --centreline data/tracks/oschersleben/centreline.csv
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from lapzero.eval.sweep import (
    FRICTION_SWEEP_DIR,
    SweepRun,
    friction_grid,
    read_sweep,
    sweep_friction,
    write_sweep,
)
from lapzero.sim.f1tenth import F1TenthSimulator
from lapzero.sim.simulator import Pose
from lapzero.sim.track import load_centreline, start_pose_from_centreline
from lapzero.training.policy import load_policy
from lapzero.training.tracking import (
    REGISTERED_MODEL,
    ModelAlias,
    git_commit,
    load_registered_policy,
    mlflow_client,
)


def main(argv: list[str] | None = None) -> None:
    """Parse arguments, load the policy, sweep the friction levels, export and print results."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--map", type=Path, required=True, help="ROS map yaml of the track")
    parser.add_argument("--centreline", type=Path, required=True, help="centreline CSV")
    parser.add_argument(
        "--start",
        type=float,
        nargs=3,
        metavar=("X", "Y", "THETA"),
        help="start pose in the map frame: metres, metres, rad (default: from the centreline)",
    )
    parser.add_argument("--track-id", help="names the export directory (default: map file name)")
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--alias", choices=list(ModelAlias), default=ModelAlias.CHAMPION)
    source.add_argument("--checkpoint", type=Path, help="local .pt instead of the registry")
    parser.add_argument("--registered-model", default=REGISTERED_MODEL)
    parser.add_argument("--tracking-uri", default=Path("mlruns").resolve().as_uri())
    parser.add_argument(
        "--frictions",
        type=float,
        nargs=3,
        default=[0.50, 1.05, 0.05],
        metavar=("LOW", "HIGH", "STEP"),
        help="friction mu levels LOW to HIGH inclusive (default 0.50 1.05 0.05)",
    )
    parser.add_argument("--episodes", type=int, default=10, help="episodes per friction level")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--randomize-start",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="jitter the start pose per episode seed (default on)",
    )
    parser.add_argument("--out", type=Path, default=Path(FRICTION_SWEEP_DIR))
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    frictions = friction_grid(*args.frictions)
    centreline = load_centreline(args.centreline)
    start = Pose(*args.start) if args.start else start_pose_from_centreline(centreline)
    if args.checkpoint is not None:
        policy, label, version = load_policy(args.checkpoint), str(args.checkpoint), None
    else:
        policy = load_registered_policy(
            tracking_uri=args.tracking_uri, model_name=args.registered_model, alias=args.alias
        )
        label = f"{args.registered_model}@{args.alias}"
        client = mlflow_client(args.tracking_uri)
        version = str(client.get_model_version_by_alias(args.registered_model, args.alias).version)
    run = SweepRun(
        git_commit=git_commit(),
        policy=label,
        policy_version=version,
        track_id=args.track_id or args.map.stem,
        map_yaml=str(args.map),
        start_pose=(start.x, start.y, start.theta),
        seed=args.seed,
        episodes_per_level=args.episodes,
        randomize_start=args.randomize_start,
    )
    simulator = F1TenthSimulator(args.map, seed=args.seed)
    try:
        levels = sweep_friction(
            simulator,
            start,
            policy,
            frictions=frictions,
            episodes=args.episodes,
            seed=args.seed,
            randomize_start=args.randomize_start,
            centreline=centreline,
        )
    finally:
        simulator.close()
    out_dir = write_sweep(args.out, run, levels)
    report = read_sweep(out_dir)

    jitter = "jittered" if args.randomize_start else "fixed"
    print(f"\n{label} on {run.track_id}, {args.episodes} episodes per level, {jitter} start")
    print(f"  policy version {version or '-'}, start {start}")
    print("      mu  completed  crashed  mean lap")
    for level in report.levels:
        lap = f"{level.mean_lap_time_s:6.2f} s" if level.mean_lap_time_s is not None else "   -"
        print(
            f"  {level.friction_mu:6.3f}  {level.completed:4d}/{level.episodes:<4d}"
            f"  {level.crashed:7d}  {lap}"
        )
    for crash in report.crashes:
        print(
            f"  crash at mu {crash.friction_mu:.3f} seed {crash.seed:3d}: "
            f"x {crash.x:7.2f} y {crash.y:7.2f}, {crash.lap_fraction:6.1%} of the lap"
        )
    print(f"sweep.json and traces: {out_dir}")


if __name__ == "__main__":
    main()
