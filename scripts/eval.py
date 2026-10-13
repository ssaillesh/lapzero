<<<<<<< HEAD
import argparse
import pathlib
import numpy as np
import mlflow
import torch
import gymnasium as gym

def load_champion_policy():
    model_uri = "models:/lapzero-policy@champion"
    print(f"Loading champion policy from {model_uri}...")
    local_path = mlflow.artifacts.download_artifacts(model_uri=model_uri)
    policy_path = pathlib.Path(local_path) / "model.pt"
    if not policy_path.exists():
        policy_path = pathlib.Path("models/ppo_v1.pt")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    policy = torch.load(policy_path, map_location=device)
    policy.eval()
    return policy, device

def main():
    parser = argparse.ArgumentParser(description="Evaluate champion policy noise-free.")
    parser.add_argument("--map", type=str, required=True, help="Path to track map yaml")
    parser.add_argument("--episodes", type=int, default=5, help="Number of evaluation laps")
    args = parser.parse_args()

    # Initialize environment directly via gym/lapzero wrapper if registered, 
    # or fallback to standard gym make if map is passed
    try:
        env = gym.make("lapzero/f1tenth-v0", map_path=args.map)
    except Exception:
        from lapzero.sim.env import LapZeroEnv
        env = LapZeroEnv(map_path=args.map)

    policy, device = load_champion_policy()

    successes = 0
    lap_times = []

    for ep in range(args.episodes):
        obs, info = env.reset(seed=42 + ep)
        done = False
        truncated = False
        steps = 0
        total_reward = 0.0

        while not (done or truncated):
            obs_tensor = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
            with torch.no_grad():
                # Handle policy output structure (actor action)
                output = policy(obs_tensor)
                action = output[0] if isinstance(output, tuple) else output
            
            action = action.cpu().numpy().squeeze(0)
            obs, reward, done, truncated, info = env.step(action)
            steps += 1
            total_reward += reward

        crashed = info.get("collision", False) if isinstance(info, dict) else False
        if not crashed:
            successes += 1
            lap_time = steps * 0.04
            lap_times.append(lap_time)
            print(f"Episode {ep + 1}: SUCCESS | Time: {lap_time:.2f}s | Reward: {total_reward:.2f}")
        else:
            print(f"Episode {ep + 1}: CRASHED at step {steps} | Reward: {total_reward:.2f}")

    print("\n--- Evaluation Summary ---")
    print(f"Success Rate: {successes}/{args.episodes} ({successes / args.episodes * 100}%)")
    if lap_times:
        print(f"Mean Lap Time: {np.mean(lap_times):.2f}s (Min: {np.min(lap_times):.2f}s)")
=======
"""Evaluate a policy on one track: lap completion, lap time, crash rate and cross-track error.

The policy comes from the MLflow model registry (--alias, default champion) or a local
save_policy checkpoint (--checkpoint) and always drives its deterministic action. Episodes start
from the fixed start pose: --start, or by default the first centreline point facing the second
(the f1tenth_racetracks start line). The contract reserves start jitter for training;
--randomize-start is a robustness diagnostic only, as are --friction (tyre-road friction mu) and
--lidar-noise, --speed-noise and --yaw-rate-noise (Gaussian noise on what the policy observes).

--out writes summary.json and one CSV trace per episode to <out>/<track_id>/<eval_id>/ (bare
--out: data/eval).

Usage (from the repo root):
    uv run python scripts/eval.py --map data/tracks/spielberg/spielberg.yaml \
        --centreline data/tracks/spielberg/centreline.csv --out
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from lapzero.eval.export import EVAL_DIR, EvalRun, write_eval
from lapzero.eval.perturb import ObservationNoise
from lapzero.eval.runner import evaluate_policy
from lapzero.sim.env import EnvConfig, LapZeroEnv
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
    """Parse arguments, load the policy, run the episodes, export and print the results."""
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
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--randomize-start", action="store_true", help="diagnostic only")
    parser.add_argument("--friction", type=float, help="tyre-road friction mu (default: F1TENTH)")
    parser.add_argument("--lidar-noise", type=float, default=0.0, help="beam noise std, metres")
    parser.add_argument("--speed-noise", type=float, default=0.0, help="speed noise std, m/s")
    parser.add_argument("--yaw-rate-noise", type=float, default=0.0, help="noise std, rad/s")
    parser.add_argument(
        "--out",
        type=Path,
        nargs="?",
        const=Path(EVAL_DIR),
        help=f"export traces and summary.json under this root (bare --out: {EVAL_DIR})",
    )
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    centreline = load_centreline(args.centreline)
    start = Pose(*args.start) if args.start else start_pose_from_centreline(centreline)
    noise = ObservationNoise(args.lidar_noise, args.speed_noise, args.yaw_rate_noise)
    if args.checkpoint is not None:
        policy, label, version = load_policy(args.checkpoint), str(args.checkpoint), None
    else:
        policy = load_registered_policy(
            tracking_uri=args.tracking_uri, model_name=args.registered_model, alias=args.alias
        )
        label = f"{args.registered_model}@{args.alias}"
        client = mlflow_client(args.tracking_uri)
        version = str(client.get_model_version_by_alias(args.registered_model, args.alias).version)
    simulator = F1TenthSimulator(args.map, seed=args.seed, friction=args.friction)
    env = LapZeroEnv(
        simulator,
        start,
        config=EnvConfig(randomize_start=args.randomize_start),
        centreline=centreline,
    )
    run = EvalRun(
        git_commit=git_commit(),
        policy=label,
        policy_version=version,
        track_id=args.track_id or args.map.stem,
        map_yaml=str(args.map),
        start_pose=(start.x, start.y, start.theta),
        seed=args.seed,
        randomize_start=args.randomize_start,
        friction_mu=simulator.friction,
        obs_noise=noise,
    )
    try:
        summary = evaluate_policy(
            env,
            policy,
            episodes=args.episodes,
            seed=args.seed,
            obs_noise=None if noise.is_zero else noise,
        )
    finally:
        env.close()
    out_dir = write_eval(args.out, run, summary) if args.out is not None else None

    noise_text = "noise-free" if noise.is_zero else f"noisy ({noise})"
    print(f"\n{label} on {run.track_id}, {args.episodes} {noise_text} episodes")
    print(f"  policy version {version or '-'}, friction mu {run.friction_mu}, start {start}")
    for e in summary.episodes:
        lap = f"{e.lap_time_s:6.2f} s" if e.lap_time_s is not None else "     -  "
        trace = e.trace
        end = f"  ended at x {trace.x[-1]:7.2f} y {trace.y[-1]:7.2f}" if trace else ""
        cte = f"  max |cte| {trace.max_abs_cte_m:4.2f} m" if trace else ""
        print(
            f"  seed {e.seed:3d}  {e.status:9s} lap {lap}  driven {e.lap_fraction:6.1%}"
            f"  mean speed {e.mean_speed_mps:4.2f} m/s{cte}{end if e.lap_time_s is None else ''}"
        )
    if out_dir is not None:
        print(f"traces and summary.json: {out_dir}")
    mean = summary.mean_lap_time_s
    print(
        f"completed {summary.completion_rate:.0%}, crashed {summary.crash_rate:.0%}, "
        f"mean lap time {f'{mean:.2f} s' if mean is not None else 'n/a'}"
    )

>>>>>>> b16f4dc (feat: multi-track subproc training, contract v1.1.0 patch, and gate promotion)

if __name__ == "__main__":
    main()
