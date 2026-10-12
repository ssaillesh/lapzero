"""Drive a track with the keyboard or a gamepad and record every lap for behaviour cloning.

Focus the F1TENTH window and press throttle (Up, or the right trigger) to start each episode.
Arrows steer, Up is throttle and Down is brake; R restarts the episode and Esc quits. F1TENTH
draws the world mirrored, so steering follows the screen (recorded actions stay physical). Laps
are written to <laps-dir>/<track-id>/<lap_id>.json + .npz.

Usage (from the repo root):
    uv run python scripts/drive.py --map tracks/track_02/map.yaml --start 0 0 0 \
        --track-id track_02 --direction ccw --length-m 82.4
"""

from __future__ import annotations

import argparse
import importlib.metadata
import logging
import os
from pathlib import Path

from lapzero.contract import LAPS_DIR, LAPS_PER_EPISODE_TRAIN
from lapzero.io.telemetry import DriverInfo, DriverType, InputDevice, TrackDirection, TrackInfo
from lapzero.record.driver import DriverConfig, HumanDriver
from lapzero.record.pygame_input import GamepadLayout, open_input
from lapzero.record.recorder import LapRecorder
from lapzero.record.session import drive_session
from lapzero.sim.env import EnvConfig, LapZeroEnv
from lapzero.sim.f1tenth import RENDER_MIRRORED, F1TenthSimulator
from lapzero.sim.simulator import Pose

_LAYOUT = GamepadLayout()
_RATES = DriverConfig()


def main(argv: list[str] | None = None) -> None:
    """Parse arguments, open the simulator window and record a driving session."""
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
    parser.add_argument("--track-id", required=True)
    parser.add_argument("--direction", choices=list(TrackDirection), required=True)
    parser.add_argument("--length-m", type=float, required=True, help="centreline length, metres")
    parser.add_argument("--device", choices=list(InputDevice), default=InputDevice.KEYBOARD)
    parser.add_argument("--driver-name", default="human")
    parser.add_argument("--gamepad-index", type=int, default=0)
    parser.add_argument("--steer-axis", type=int, default=_LAYOUT.steer_axis)
    parser.add_argument("--throttle-axis", type=int, default=_LAYOUT.throttle_axis)
    parser.add_argument("--brake-axis", type=int, default=_LAYOUT.brake_axis)
    parser.add_argument(
        "--steer-rate", type=float, default=_RATES.steer_rate, help="steer per step, turning in"
    )
    parser.add_argument(
        "--steer-return-rate",
        type=float,
        default=_RATES.steer_return_rate,
        help="steer per step, back toward centre",
    )
    parser.add_argument("--throttle-rate", type=float, default=_RATES.throttle_rate)
    parser.add_argument("--laps", type=int, default=LAPS_PER_EPISODE_TRAIN, help="laps per episode")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--laps-dir", type=Path, default=Path(LAPS_DIR))
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    device = InputDevice(args.device)
    config = EnvConfig(laps_per_episode=args.laps)
    track = TrackInfo(
        track_id=args.track_id, direction=TrackDirection(args.direction), length_m=args.length_m
    )
    driver = DriverInfo(type=DriverType.HUMAN, name=args.driver_name, human_input_device=device)
    layout = GamepadLayout(args.steer_axis, args.throttle_axis, args.brake_axis)
    rates = DriverConfig(args.steer_rate, args.steer_return_rate, args.throttle_rate)
    # SDL turns Ctrl+C into a window-close event that F1TENTH's renderer discards; keep Python's
    # signal handling so Ctrl+C still stops the script. Read when the window opens.
    os.environ.setdefault("SDL_NO_SIGNAL_HANDLERS", "1")
    simulator = F1TenthSimulator(args.map, seed=args.seed, render=True)
    env = LapZeroEnv(simulator, Pose(*args.start), config=config)
    try:
        source = open_input(
            device,
            gamepad_index=args.gamepad_index,
            layout=layout,
            mirror_steering=RENDER_MIRRORED,
        )
        recorder = LapRecorder(
            track,
            driver,
            sim_version=importlib.metadata.version("f1tenth-gym"),
            env_config=config,
            root=args.laps_dir,
        )
        laps = drive_session(
            env,
            recorder,
            source,
            seed=args.seed,
            driver=HumanDriver(rates),
            render=simulator.render,
        )
    finally:
        env.close()
    for lap in laps:
        print(f"{lap.status:>9}  {lap.json_path}")
    print(f"recorded {len(laps)} lap(s)")


if __name__ == "__main__":
    main()
