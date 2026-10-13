"""Unit tests for lapzero/inference/schemas.py: contract bounds, finiteness and the consistency
rules between an episode's frames and its summary.
"""

import math
from typing import Any

import pytest
from pydantic import ValidationError

from lapzero.contract import CTRL_DT, SIM_V_MAX, SIM_V_MIN, STEER_MAX
from lapzero.inference.schemas import EpisodeResult, StepTelemetry
from lapzero.io.telemetry import LapStatus

FLOAT_FIELDS = [
    "t",
    "x",
    "y",
    "theta",
    "v_long",
    "v_lat",
    "speed",
    "steer",
    "throttle",
    "steer_rad",
    "v_target",
    "progress",
    "lap_fraction",
    "lap_time",
    "reward",
    "friction_mu",
]


def frame(**overrides: Any) -> StepTelemetry:
    """A valid mid-lap frame at step 10, with any field overridden."""
    values: dict[str, Any] = {
        "step": 10,
        "t": 10 * CTRL_DT,
        "x": 1.5,
        "y": -0.25,
        "theta": 0.1,
        "v_long": 3.0,
        "v_lat": 0.4,
        "speed": math.hypot(3.0, 0.4),
        "steer": 0.2,
        "throttle": 0.8,
        "steer_rad": 0.2 * STEER_MAX,
        "v_target": 3.5,
        "progress": 0.05,
        "lap_fraction": 0.05,
        "lap_count": 0,
        "lap_time": 10 * CTRL_DT,
        "collision": False,
        "reward": 0.12,
        "friction_mu": 1.05,
    }
    values.update(overrides)
    return StepTelemetry(**values)


def episode_frames(
    steps: int, *, lap_count: int = 0, collision: bool = False
) -> list[StepTelemetry]:
    """Frames 0..steps, one CTRL_DT apart; the last frame gets lap_count and collision."""
    frames = [frame(step=i, t=i * CTRL_DT, lap_time=i * CTRL_DT, lap_count=0) for i in range(steps)]
    end = steps * CTRL_DT
    frames.append(frame(step=steps, t=end, lap_time=0.0, lap_count=lap_count, collision=collision))
    return frames


def episode(frames: list[StepTelemetry], status: LapStatus, **overrides: Any) -> EpisodeResult:
    """An episode whose summary matches its last frame, with any field overridden."""
    last = frames[-1]
    values: dict[str, Any] = {
        "status": status,
        "collision": last.collision,
        "steps": last.step,
        "lap_count": last.lap_count,
        "episode_time_s": last.t,
        "lap_time_s": last.t if status is LapStatus.COMPLETED else None,
        "frames": frames,
    }
    values.update(overrides)
    return EpisodeResult(**values)


# StepTelemetry


def test_step_telemetry_round_trips_through_json() -> None:
    original = frame()
    assert StepTelemetry.model_validate_json(original.model_dump_json()) == original


@pytest.mark.parametrize("bad", [math.nan, math.inf, -math.inf])
@pytest.mark.parametrize("name", FLOAT_FIELDS)
def test_step_telemetry_rejects_nan_and_inf_in_every_float_field(name: str, bad: float) -> None:
    with pytest.raises(ValidationError, match=rf"(?m)^{name}$"):
        frame(**{name: bad})


def test_step_telemetry_rejects_unknown_fields() -> None:
    with pytest.raises(ValidationError, match="extra"):
        frame(yaw=0.0)


def test_step_telemetry_is_frozen() -> None:
    with pytest.raises(ValidationError, match="frozen"):
        frame().x = 2.0  # type: ignore[misc]


@pytest.mark.parametrize(
    ("name", "value"),
    [
        ("step", -1),
        ("t", -0.01),
        ("theta", -math.pi),
        ("theta", math.pi + 1e-6),
        ("steer", 1.001),
        ("steer", -1.001),
        ("throttle", 1.001),
        ("throttle", -1.001),
        ("steer_rad", STEER_MAX + 1e-6),
        ("steer_rad", -STEER_MAX - 1e-6),
        ("v_target", SIM_V_MAX + 0.1),
        ("v_target", SIM_V_MIN - 0.1),
        ("progress", -0.001),
        ("progress", 1.001),
        ("lap_count", -1),
        ("lap_time", -0.01),
        ("friction_mu", 0.0),
    ],
)
def test_step_telemetry_rejects_values_outside_contract_bounds(name: str, value: float) -> None:
    with pytest.raises(ValidationError, match=rf"(?m)^{name}$"):
        frame(**{name: value})


@pytest.mark.parametrize(
    ("name", "value"),
    [
        ("theta", math.pi),
        ("steer", 1.0),
        ("steer", -1.0),
        ("throttle", -1.0),
        ("steer_rad", STEER_MAX),
        ("steer_rad", -STEER_MAX),
        ("v_target", SIM_V_MAX),
        ("v_target", SIM_V_MIN),
        ("progress", 0.0),
        ("progress", 1.0),
    ],
)
def test_step_telemetry_accepts_values_on_contract_bounds(name: str, value: float) -> None:
    assert getattr(frame(**{name: value}), name) == value


@pytest.mark.parametrize("lap_fraction", [-0.2, 1.0, 2.35])
def test_lap_fraction_can_pass_one_and_go_negative(lap_fraction: float) -> None:
    assert frame(lap_fraction=lap_fraction).lap_fraction == lap_fraction


def test_speed_must_be_the_magnitude_of_long_and_lat_velocity() -> None:
    with pytest.raises(ValidationError, match="speed must be hypot"):
        frame(v_long=3.0, v_lat=4.0, speed=7.0)


def test_speed_of_a_car_sliding_backwards_is_positive() -> None:
    assert frame(v_long=-3.0, v_lat=-4.0, speed=5.0).speed == 5.0


def test_lap_time_cannot_exceed_episode_time() -> None:
    with pytest.raises(ValidationError, match="exceeds episode time"):
        frame(t=1.0, lap_time=1.04)


# EpisodeResult


def test_completed_episode_round_trips_through_json() -> None:
    original = episode(episode_frames(5, lap_count=1), LapStatus.COMPLETED)
    restored = EpisodeResult.model_validate_json(original.model_dump_json())
    assert restored == original
    assert restored.frames[-1] == original.frames[-1]


def test_reset_frame_alone_is_an_episode_of_zero_steps() -> None:
    result = episode(episode_frames(0), LapStatus.STUCK)
    assert result.steps == 0
    assert result.episode_time_s == 0.0


@pytest.mark.parametrize("status", list(LapStatus))
def test_success_is_true_only_for_completed_episodes(status: LapStatus) -> None:
    frames = episode_frames(
        5,
        lap_count=1 if status is LapStatus.COMPLETED else 0,
        collision=status is LapStatus.CRASHED,
    )
    assert episode(frames, status).success is (status is LapStatus.COMPLETED)


def test_episode_needs_at_least_the_reset_frame() -> None:
    with pytest.raises(ValidationError, match="frames"):
        EpisodeResult(
            status=LapStatus.STUCK,
            collision=False,
            steps=0,
            lap_count=0,
            episode_time_s=0.0,
            frames=(),
        )


def test_frames_must_be_numbered_from_zero_without_gaps() -> None:
    frames = episode_frames(3)
    del frames[2]
    with pytest.raises(ValidationError, match=r"frames\[2\]\.step is 3, expected 2"):
        episode(frames, LapStatus.STUCK)


def test_frames_must_start_at_the_reset_state() -> None:
    frames = episode_frames(3)[1:]
    with pytest.raises(ValidationError, match=r"frames\[0\]\.step is 1, expected 0"):
        episode(frames, LapStatus.STUCK)


def test_frame_times_must_increase() -> None:
    frames = episode_frames(3)
    frames[2] = frame(step=2, t=frames[1].t, lap_time=frames[1].t)
    with pytest.raises(ValidationError, match=r"frames\[2\]\.t 0\.04 s does not follow 0\.04 s"):
        episode(frames, LapStatus.STUCK)


def test_lap_count_cannot_go_down_between_frames() -> None:
    frames = episode_frames(3)
    frames[1] = frame(step=1, t=CTRL_DT, lap_time=CTRL_DT, lap_count=1)
    with pytest.raises(ValidationError, match=r"frames\[2\]\.lap_count 0 is below 1"):
        episode(frames, LapStatus.STUCK)


def test_episode_cannot_continue_after_a_collision() -> None:
    frames = episode_frames(3)
    frames[1] = frame(step=1, t=CTRL_DT, lap_time=CTRL_DT, collision=True)
    with pytest.raises(ValidationError, match=r"frames\[1\] has a collision"):
        episode(frames, LapStatus.STUCK)


@pytest.mark.parametrize(
    ("name", "value"),
    [("steps", 4), ("lap_count", 0), ("episode_time_s", 1.0)],
)
def test_summary_must_match_the_last_frame(name: str, value: float) -> None:
    frames = episode_frames(5, lap_count=1)
    with pytest.raises(ValidationError, match=f"{name} is {value}, the last frame has"):
        episode(frames, LapStatus.TIMEOUT, **{name: value})


def test_collision_flag_must_match_the_last_frame() -> None:
    frames = episode_frames(5, collision=True)
    with pytest.raises(ValidationError, match="collision is False, the last frame has True"):
        episode(frames, LapStatus.CRASHED, collision=False)


def test_crash_on_the_last_frame_must_be_reported_as_crashed() -> None:
    frames = episode_frames(5, collision=True)
    with pytest.raises(ValidationError, match="collision is True but status is timeout"):
        episode(frames, LapStatus.TIMEOUT)


def test_crashed_status_needs_a_collision() -> None:
    with pytest.raises(ValidationError, match="collision is False but status is crashed"):
        episode(episode_frames(5), LapStatus.CRASHED)


def test_completed_episode_needs_a_lap_time() -> None:
    frames = episode_frames(5, lap_count=1)
    with pytest.raises(ValidationError, match="lap_time_s must be set for completed"):
        episode(frames, LapStatus.COMPLETED, lap_time_s=None)


@pytest.mark.parametrize("status", [LapStatus.TIMEOUT, LapStatus.STUCK])
def test_unfinished_episode_has_no_lap_time(status: LapStatus) -> None:
    with pytest.raises(ValidationError, match="null otherwise"):
        episode(episode_frames(5), status, lap_time_s=5 * CTRL_DT)


def test_start_line_pass_the_env_did_not_count_is_not_a_completed_episode() -> None:
    frames = episode_frames(5, lap_count=0)
    with pytest.raises(ValidationError, match="at least one counted lap"):
        episode(frames, LapStatus.COMPLETED)


def test_lap_time_must_equal_the_episode_time() -> None:
    frames = episode_frames(5, lap_count=1)
    with pytest.raises(ValidationError, match="must equal episode_time_s"):
        episode(frames, LapStatus.COMPLETED, lap_time_s=0.1)
