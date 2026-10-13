"""Unit tests for lapzero/record/recorder.py: lap segmentation, frames, files and validation.

Sessions run LapZeroEnv on the scripted FakeSimulator and write into a temporary directory.
"""

import itertools
import json
import math
import re
from collections.abc import Callable, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from lapzero.contract import ContractViolationError
from lapzero.io.telemetry import (
    ULID_PATTERN,
    DriverInfo,
    DriverType,
    InputDevice,
    LapStatus,
    LapTelemetry,
    TrackDirection,
    TrackInfo,
    read_lap_telemetry,
)
from lapzero.record.recorder import LapFiles, LapRecorder
from lapzero.record.ulid import encode_ulid
from lapzero.sim.env import EnvConfig, LapZeroEnv
from lapzero.sim.simulator import Pose
from tests.conftest import FakeSimulator

TRACK = TrackInfo(track_id="track_02", direction=TrackDirection.CCW, length_m=82.4)
HUMAN = DriverInfo(type=DriverType.HUMAN, name="tester", human_input_device=InputDevice.KEYBOARD)
NOW = datetime(2026, 10, 20, 14, 32, 5, 987654, tzinfo=UTC)
SEED = 7
FULL_THROTTLE = (0.0, 1.0)
CONTRACT_KEYS = {
    "schema",
    "schema_version",
    "contract_version",
    "lap_id",
    "episode_id",
    "lap_index",
    "created_at",
    "track",
    "driver",
    "sim",
    "result",
    "frames",
}
FRAME_KEYS = {
    "t",
    "x",
    "y",
    "theta",
    "v_long",
    "steer",
    "throttle",
    "steer_rad",
    "v_target",
    "progress",
}
DECIMALS = {
    "t": 2,
    "x": 3,
    "y": 3,
    "theta": 4,
    "v_long": 3,
    "steer": 3,
    "throttle": 3,
    "steer_rad": 4,
    "v_target": 3,
    "progress": 4,
}


class Session:
    """Drives a LapZeroEnv and hands every reset and step result to a LapRecorder."""

    def __init__(self, env: LapZeroEnv, recorder: LapRecorder) -> None:
        self.env = env
        self.recorder = recorder
        obs, info = env.reset(seed=SEED)
        recorder.start_episode(obs, info, seed=SEED)
        self.observations = [obs]
        self.actions: list[np.ndarray] = []
        self.files: list[LapFiles] = []

    def step(self, action: Sequence[float] = FULL_THROTTLE) -> list[LapFiles]:
        act = np.array(action, dtype=np.float32)
        obs, reward, terminated, truncated, info = self.env.step(act)
        written = self.recorder.record_step(act, obs, reward, terminated, truncated, info)
        self.observations.append(obs)
        self.actions.append(act)
        self.files += written
        return written

    def drive(self, steps: int, action: Sequence[float] = FULL_THROTTLE) -> None:
        for _ in range(steps):
            self.step(action)


SessionFactory = Callable[..., Session]


@pytest.fixture
def start_session(make_env: Callable[..., LapZeroEnv], tmp_path: Path) -> Callable[..., Session]:
    """Start a recorded session; keyword arguments go to both the env and the recorder config."""
    ids = itertools.count()

    def _start(**config: Any) -> Session:
        recorder = LapRecorder(
            TRACK,
            HUMAN,
            sim_version="1.0.0.dev0",
            env_config=EnvConfig(**config),
            root=tmp_path,
            clock=lambda: NOW,
            new_id=lambda: encode_ulid(1_790_000_000_000 + next(ids), bytes(10)),
        )
        return Session(make_env(**config), recorder)

    return _start


def _complete_lap_after(session: Session, fake_sim: FakeSimulator, steps: int) -> LapFiles:
    """Drive `steps` control steps, the last of which crosses the start line."""
    session.drive(steps - 1)
    fake_sim.lap_count += 1
    (files,) = session.step()
    return files


# Files and schema


def test_completed_lap_writes_json_and_npz_named_by_lap_id_under_the_track_directory(
    start_session: SessionFactory, fake_sim: FakeSimulator, tmp_path: Path
) -> None:
    files = _complete_lap_after(start_session(laps_per_episode=2), fake_sim, steps=5)
    assert files.status is LapStatus.COMPLETED
    assert files.json_path == tmp_path / "track_02" / f"{files.lap_id}.json"
    assert files.npz_path == tmp_path / "track_02" / f"{files.lap_id}.npz"
    assert files.json_path.is_file()
    assert files.npz_path.is_file()


def test_exported_json_validates_and_carries_the_contract_header(
    start_session: SessionFactory, fake_sim: FakeSimulator
) -> None:
    files = _complete_lap_after(start_session(laps_per_episode=2), fake_sim, steps=5)
    raw = json.loads(files.json_path.read_text())
    telemetry = read_lap_telemetry(files.json_path)
    assert set(raw) == CONTRACT_KEYS
    assert set(raw["frames"]) == FRAME_KEYS
    assert raw["schema"] == "lapzero.lap_telemetry"
    assert (raw["schema_version"], raw["contract_version"]) == ("1.0.0", "1.1.0")
    assert raw["created_at"] == "2026-10-20T14:32:05Z"
    assert re.fullmatch(ULID_PATTERN, raw["lap_id"])
    assert re.fullmatch(ULID_PATTERN, raw["episode_id"])
    assert telemetry.lap_id == files.lap_id
    assert raw["track"] == {"track_id": "track_02", "direction": "ccw", "length_m": 82.4}
    assert raw["driver"] == {
        "type": "human",
        "name": "tester",
        "model_version": None,
        "mlflow_run_id": None,
        "human_input_device": "keyboard",
    }
    assert raw["sim"] == {
        "simulator": "f1tenth_gym",
        "sim_version": "1.0.0.dev0",
        "sim_dt": 0.01,
        "action_repeat": 4,
        "ctrl_hz": 25,
        "v_max_train": 8.0,
        "seed": SEED,
    }


# Frames and arrays


def test_frames_are_sampled_at_the_25_hz_control_rate(
    start_session: SessionFactory, fake_sim: FakeSimulator
) -> None:
    files = _complete_lap_after(start_session(laps_per_episode=2), fake_sim, steps=12)
    telemetry = read_lap_telemetry(files.json_path)
    assert telemetry.result.sample_count == 12
    assert telemetry.frames.t == pytest.approx([0.04 * i for i in range(12)])
    assert np.diff(telemetry.frames.t) == pytest.approx([0.04] * 11)


def test_npz_holds_the_observation_seen_before_each_recorded_action(
    start_session: SessionFactory, fake_sim: FakeSimulator
) -> None:
    session = start_session(laps_per_episode=2)
    session.drive(3, action=(0.5, 1.0))
    fake_sim.lap_count = 1
    (files,) = session.step((-0.25, 0.5))
    with np.load(files.npz_path) as arrays:
        assert arrays["obs"].dtype == np.float32
        assert arrays["obs"].shape == (4, 113)
        np.testing.assert_array_equal(arrays["obs"], np.stack(session.observations[:4]))
        assert arrays["actions"].dtype == np.float32
        np.testing.assert_array_equal(arrays["actions"], np.stack(session.actions))
        assert arrays["rewards"].dtype == np.float32
        assert arrays["rewards"].shape == (4,)
        assert arrays["dones"].dtype == np.bool_
        assert not arrays["dones"].any()
        assert str(arrays["contract_version"]) == "1.1.0"


def test_each_frame_pairs_the_state_before_a_step_with_its_action_and_command(
    start_session: SessionFactory, fake_sim: FakeSimulator
) -> None:
    files = _complete_lap_after(start_session(laps_per_episode=2), fake_sim, steps=3)
    frames = read_lap_telemetry(files.json_path).frames
    assert frames.v_long == pytest.approx([0.0, 0.6, 1.2])  # speed before each step
    assert frames.throttle == [1.0, 1.0, 1.0]
    assert frames.v_target == pytest.approx([0.6, 1.2, 1.8])  # command from that speed
    assert frames.steer == [0.0, 0.0, 0.0]
    assert frames.steer_rad == [0.0, 0.0, 0.0]


def test_every_value_is_rounded_to_its_contract_precision(
    start_session: SessionFactory, fake_sim: FakeSimulator
) -> None:
    session = start_session(laps_per_episode=2)
    fake_sim.pose = Pose(1.23456789, -2.3456789, 0.123456789)
    session.step((0.123456, 0.987654))
    fake_sim.lap_count = 1
    (files,) = session.step((0.123456, 0.987654))
    raw = json.loads(files.json_path.read_text())
    assert raw["frames"]["x"][1] == 1.235
    assert raw["frames"]["theta"][1] == 0.1235
    assert raw["frames"]["steer"][1] == 0.123
    for name, values in raw["frames"].items():
        assert all(round(value, DECIMALS[name]) == value for value in values), name


# Lap segmentation and results


def test_completed_lap_time_is_measured_from_the_lap_start(
    start_session: SessionFactory, fake_sim: FakeSimulator
) -> None:
    files = _complete_lap_after(start_session(laps_per_episode=2), fake_sim, steps=10)
    result = read_lap_telemetry(files.json_path).result
    assert result.lap_time_s == pytest.approx(0.4)
    assert result.collision is False
    assert result.off_track_events == 0


def test_each_lap_of_an_episode_gets_its_own_files_index_and_start_time(
    start_session: SessionFactory, fake_sim: FakeSimulator
) -> None:
    session = start_session(laps_per_episode=3)
    first = read_lap_telemetry(_complete_lap_after(session, fake_sim, steps=5).json_path)
    second = read_lap_telemetry(_complete_lap_after(session, fake_sim, steps=3).json_path)
    assert (first.lap_index, second.lap_index) == (1, 2)
    assert first.episode_id == second.episode_id
    assert first.lap_id < second.lap_id
    assert second.frames.t[0] == pytest.approx(0.2)
    assert (first.result.lap_time_s, second.result.lap_time_s) == pytest.approx((0.2, 0.12))


def test_speed_summary_comes_from_the_recorded_longitudinal_speeds(
    start_session: SessionFactory, fake_sim: FakeSimulator
) -> None:
    files = _complete_lap_after(start_session(laps_per_episode=2), fake_sim, steps=4)
    result = read_lap_telemetry(files.json_path).result
    assert result.max_speed_mps == pytest.approx(1.8)  # speeds 0.0, 0.6, 1.2, 1.8
    assert result.mean_speed_mps == pytest.approx(0.9)


def test_crash_writes_a_crashed_lap_without_lap_time(
    start_session: SessionFactory, fake_sim: FakeSimulator
) -> None:
    fake_sim.collision_at_step = 10  # physics step 10 falls in control step 3
    session = start_session()
    session.drive(3)
    (files,) = session.files
    telemetry = read_lap_telemetry(files.json_path)
    assert files.status is LapStatus.CRASHED
    assert telemetry.result.lap_time_s is None
    assert telemetry.result.collision is True
    with np.load(files.npz_path) as arrays:
        assert arrays["dones"].tolist() == [False, False, True]


def test_hitting_the_time_limit_writes_a_timeout_lap(start_session: SessionFactory) -> None:
    session = start_session(max_episode_s=0.2)
    session.drive(5)
    assert [files.status for files in session.files] == [LapStatus.TIMEOUT]


def test_a_car_stuck_for_three_seconds_writes_a_stuck_lap(
    start_session: SessionFactory, fake_sim: FakeSimulator
) -> None:
    fake_sim.follow_target = False
    session = start_session()
    session.drive(75)
    assert [files.status for files in session.files] == [LapStatus.STUCK]


def test_no_failed_lap_follows_a_lap_that_ends_the_episode(
    start_session: SessionFactory, fake_sim: FakeSimulator
) -> None:
    session = start_session(laps_per_episode=1)
    _complete_lap_after(session, fake_sim, steps=5)
    assert [files.status for files in session.files] == [LapStatus.COMPLETED]


def test_starting_a_new_episode_discards_the_unfinished_lap(
    start_session: SessionFactory, fake_sim: FakeSimulator, tmp_path: Path
) -> None:
    session = start_session(laps_per_episode=2)
    session.drive(6)
    obs, info = session.env.reset(seed=SEED + 1)
    session.recorder.start_episode(obs, info, seed=SEED + 1)
    files = _complete_lap_after(session, fake_sim, steps=2)
    telemetry = read_lap_telemetry(files.json_path)
    assert telemetry.result.sample_count == 2
    assert telemetry.sim.seed == SEED + 1
    assert list((tmp_path / "track_02").glob("*.json")) == [files.json_path]


# Validation: recorded laps are training data and must satisfy the contract spaces


@pytest.mark.parametrize("action", [(1.5, 0.0), (0.0, -1.01), (math.nan, 0.0), (0.0, 0.0, 0.0)])
def test_actions_outside_the_action_space_are_rejected_not_clipped(
    start_session: SessionFactory, action: tuple[float, ...]
) -> None:
    session = start_session()
    obs, reward, terminated, truncated, info = session.env.step(np.zeros(2, dtype=np.float32))
    with pytest.raises(ContractViolationError, match="action"):
        session.recorder.record_step(action, obs, reward, terminated, truncated, info)


@pytest.mark.parametrize(
    "obs", [np.zeros(112, dtype=np.float32), np.full(113, 2.0), np.full(113, np.nan)]
)
def test_observations_outside_the_observation_space_are_rejected(
    start_session: SessionFactory, obs: np.ndarray
) -> None:
    session = start_session()
    _, info = session.env.reset(seed=SEED)
    with pytest.raises(ContractViolationError, match="observation"):
        session.recorder.start_episode(obs, info, seed=SEED)


def test_non_finite_reward_is_rejected(start_session: SessionFactory) -> None:
    session = start_session()
    obs, _, terminated, truncated, info = session.env.step(np.zeros(2, dtype=np.float32))
    with pytest.raises(ContractViolationError, match="reward"):
        session.recorder.record_step((0.0, 0.0), obs, math.inf, terminated, truncated, info)


def test_termination_without_collision_or_finished_lap_breaks_the_contract(
    start_session: SessionFactory,
) -> None:
    session = start_session()
    obs, reward, _, _, info = session.env.step(np.zeros(2, dtype=np.float32))
    with pytest.raises(ContractViolationError, match="terminated"):
        session.recorder.record_step((0.0, 0.0), obs, reward, True, False, info)


def test_recording_outside_an_episode_raises_runtime_error(
    start_session: SessionFactory, fake_sim: FakeSimulator
) -> None:
    fake_sim.collision_at_step = 1
    session = start_session()
    session.step()
    obs, info = session.observations[-1], {"t": 0.04}
    with pytest.raises(RuntimeError, match="start_episode"):
        session.recorder.record_step((0.0, 0.0), obs, 0.0, False, False, info)


def test_written_telemetry_round_trips_through_the_model(
    start_session: SessionFactory, fake_sim: FakeSimulator
) -> None:
    files = _complete_lap_after(start_session(laps_per_episode=2), fake_sim, steps=3)
    text = files.json_path.read_text()
    assert LapTelemetry.model_validate_json(text).model_dump_json(by_alias=True) == text


def test_default_clock_and_ids_stamp_laps_with_current_utc_time_and_fresh_ulids(
    make_env: Callable[..., LapZeroEnv], fake_sim: FakeSimulator, tmp_path: Path
) -> None:
    recorder = LapRecorder(TRACK, HUMAN, sim_version="1.0.0.dev0", root=tmp_path)
    session = Session(make_env(laps_per_episode=2), recorder)
    before = datetime.now(UTC).replace(microsecond=0)
    files = _complete_lap_after(session, fake_sim, steps=2)
    telemetry = read_lap_telemetry(files.json_path)
    assert before <= telemetry.created_at <= datetime.now(UTC)
    assert re.fullmatch(ULID_PATTERN, telemetry.lap_id)
    assert telemetry.lap_id != telemetry.episode_id
