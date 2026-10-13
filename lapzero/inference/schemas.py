"""Typed telemetry of the inference engine: one record per control step, and the episode result.

These models cross I/O (the backend API and the live replay UI will serve them as JSON), so they
are Pydantic models that reject NaN, inf and unknown fields (CLAUDE.md rule 7). The backend may
import this module, so it must never import lapzero.sim or lapzero.training.

Lap counts are LapZeroEnv's info["lap_count"], which counts a start-line pass only once
LAP_MIN_PROGRESS of the lap has been driven, net of distance driven backwards (contract
Section 3). These models carry that count; they never re-derive it.
"""

from __future__ import annotations

import math
from itertools import pairwise
from typing import Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from lapzero.contract import SIM_V_MAX, SIM_V_MIN, STEER_MAX
from lapzero.io.telemetry import LapStatus


class _Strict(BaseModel):
    """Base for inference models: unknown fields, NaN and inf are rejected; instances are frozen."""

    model_config = ConfigDict(extra="forbid", allow_inf_nan=False, frozen=True)


class StepTelemetry(_Strict):
    """State of the car after one control step; step 0 is the state right after reset.

    Attributes:
        step: Control steps since reset; 0 for the reset state.
        t: Episode time, s since reset.
        x: World-frame x, metres.
        y: World-frame y, metres.
        theta: Heading, rad, in (-pi, pi].
        v_long: Car-frame longitudinal speed (x forward), m/s.
        v_lat: Car-frame lateral speed (y left), m/s.
        speed: Magnitude of the car's velocity, hypot(v_long, v_lat), m/s.
        steer: Normalized steering action applied in this step, in [-1, 1]; 0.0 at reset.
        throttle: Normalized throttle action applied in this step, in [-1, 1]; 0.0 at reset.
        steer_rad: Steering angle commanded in this step, rad.
        v_target: Target velocity commanded in this step, m/s.
        progress: Fraction of the lap from the start line along the centreline, in [0, 1];
            wraps back to 0 each lap (contract Section 4 info["progress"]).
        lap_fraction: Laps driven since reset along the centreline, net of distance driven
            backwards: passes 1.0 after a full lap, negative if the car reversed from the start.
        lap_count: Laps completed since reset (contract Section 3 lap completion rule).
        lap_time: Time on the current lap so far, s.
        collision: True if the car hit something in this step.
        reward: Reward summed over this step's physics steps; 0.0 at reset.
        friction_mu: Tyre-road friction coefficient of the episode (dimensionless).
    """

    step: int = Field(ge=0)
    t: float = Field(ge=0.0)
    x: float
    y: float
    theta: float = Field(gt=-math.pi, le=math.pi)
    v_long: float
    v_lat: float
    speed: float = Field(ge=0.0)
    steer: float = Field(ge=-1.0, le=1.0)
    throttle: float = Field(ge=-1.0, le=1.0)
    steer_rad: float = Field(ge=-STEER_MAX, le=STEER_MAX)
    v_target: float = Field(ge=SIM_V_MIN, le=SIM_V_MAX)
    progress: float = Field(ge=0.0, le=1.0)
    lap_fraction: float
    lap_count: int = Field(ge=0)
    lap_time: float = Field(ge=0.0)
    collision: bool
    reward: float
    friction_mu: float = Field(gt=0.0)

    @model_validator(mode="after")
    def _speed_matches_velocity(self) -> Self:
        expected = math.hypot(self.v_long, self.v_lat)
        if not math.isclose(self.speed, expected, rel_tol=1e-9, abs_tol=1e-9):
            raise ValueError(
                f"speed must be hypot(v_long, v_lat) = {expected} m/s, got {self.speed}"
            )
        return self

    @model_validator(mode="after")
    def _lap_time_within_episode(self) -> Self:
        if self.lap_time > self.t:
            raise ValueError(f"lap_time {self.lap_time} s exceeds episode time t {self.t} s")
        return self


class EpisodeResult(_Strict):
    """Outcome of one inference episode, with the telemetry of every control step.

    The summary fields repeat the last frame so a client can read the outcome without scanning
    the frames; validation rejects any mismatch.

    Attributes:
        status: completed, crashed, timeout or stuck (contract Section 3 episode rules).
        collision: True if the episode ended in a crash.
        steps: Control steps driven; frames holds steps + 1 records.
        lap_count: Laps completed, the last frame's lap_count.
        episode_time_s: Time from reset to the end of the episode, s.
        lap_time_s: Time to complete the episode's laps, s; None unless status is completed.
        frames: Telemetry from the reset state (step 0) to the last step, one per control step.
    """

    status: LapStatus
    collision: bool
    steps: int = Field(ge=0)
    lap_count: int = Field(ge=0)
    episode_time_s: float = Field(ge=0.0)
    lap_time_s: float | None = Field(default=None, gt=0.0)
    frames: tuple[StepTelemetry, ...] = Field(min_length=1)

    @property
    def success(self) -> bool:
        """True if the episode completed its laps."""
        return self.status is LapStatus.COMPLETED

    @model_validator(mode="after")
    def _frames_form_one_episode(self) -> Self:
        for i, frame in enumerate(self.frames):
            if frame.step != i:
                raise ValueError(f"frames[{i}].step is {frame.step}, expected {i}")
        for i, (prev, curr) in enumerate(pairwise(self.frames), 1):
            if curr.t <= prev.t:
                raise ValueError(f"frames[{i}].t {curr.t} s does not follow {prev.t} s")
            if curr.lap_count < prev.lap_count:
                raise ValueError(
                    f"frames[{i}].lap_count {curr.lap_count} is below {prev.lap_count}"
                )
            if prev.collision:
                raise ValueError(f"frames[{i - 1}] has a collision but the episode went on")
        return self

    @model_validator(mode="after")
    def _summary_matches_last_frame(self) -> Self:
        last = self.frames[-1]
        expected = {
            "steps": last.step,
            "lap_count": last.lap_count,
            "episode_time_s": last.t,
            "collision": last.collision,
        }
        for name, value in expected.items():
            if getattr(self, name) != value:
                raise ValueError(f"{name} is {getattr(self, name)}, the last frame has {value}")
        return self

    @model_validator(mode="after")
    def _status_matches_outcome(self) -> Self:
        if self.collision != (self.status is LapStatus.CRASHED):
            raise ValueError(f"collision is {self.collision} but status is {self.status}")
        if self.success != (self.lap_time_s is not None):
            raise ValueError("lap_time_s must be set for completed episodes and null otherwise")
        if self.success and self.lap_count < 1:
            raise ValueError("a completed episode needs at least one counted lap, got 0")
        if self.lap_time_s is not None and self.lap_time_s != self.episode_time_s:
            raise ValueError(
                f"lap_time_s {self.lap_time_s} s must equal episode_time_s {self.episode_time_s} s"
            )
        return self
