"""Real-time driving session: a human drives LapZeroEnv while LapRecorder writes every lap.

One control step per CTRL_DT seconds of wall time (25 Hz): draw, wait for the tick, poll the
input device, move the driver's action toward the request, step the env and record the step.
After each reset the car waits on the grid until the throttle is pressed, so no time passes in
the env (and the stuck timer of contract Section 3 cannot run) before the human sets off.
"""

from __future__ import annotations

import itertools
import logging
import time
from collections.abc import Callable, Iterable
from enum import Enum, auto

from lapzero.contract import CTRL_DT
from lapzero.record.driver import HumanDriver, InputSource
from lapzero.record.recorder import LapFiles, LapRecorder
from lapzero.sim.env import LapZeroEnv

logger = logging.getLogger(__name__)


class _End(Enum):
    """Why an episode loop returned."""

    FINISHED = auto()  # terminated or truncated
    RESTART = auto()
    QUIT = auto()


class _Pacer:
    """Holds a loop to one pass per CTRL_DT of wall time; a late pass resets the schedule."""

    def __init__(self, clock: Callable[[], float], sleep: Callable[[float], None]) -> None:
        self._clock = clock
        self._sleep = sleep
        self._next: float | None = None

    def wait(self) -> None:
        """Sleep until CTRL_DT after the previous wait() returned (no wait on the first call)."""
        now = self._clock()
        if self._next is not None and now < self._next:
            self._sleep(self._next - now)
            now = self._next
        self._next = now + CTRL_DT


def _no_render() -> None:
    """Draw nothing (headless sessions)."""


def drive_session(
    env: LapZeroEnv,
    recorder: LapRecorder,
    source: InputSource,
    *,
    seed: int,
    driver: HumanDriver | None = None,
    render: Callable[[], None] = _no_render,
    max_episodes: int | None = None,
    clock: Callable[[], float] = time.monotonic,
    sleep: Callable[[float], None] = time.sleep,
) -> list[LapFiles]:
    """Let a human drive episodes until they quit, recording every lap.

    Episode k (from 0) is reset with seed + k. A restart or quit mid-lap discards the unfinished
    lap, as it has no contract status.

    Args:
        env: The env to drive; `recorder` must be built with the same EnvConfig.
        recorder: Writes each completed or failed lap.
        source: Input device, polled once per control step.
        seed: Seed of the first episode; logged by the env and written to every lap.
        driver: Turns input requests into actions; defaults to HumanDriver().
        render: Draws the current state; called once per control step.
        max_episodes: Stop after this many episodes (restarts count); None runs until quit.
        clock: Monotonic wall-clock time in seconds; injectable for tests.
        sleep: Sleeps for the given seconds; injectable for tests.

    Returns:
        The files of every lap written, in order.
    """
    driver = driver or HumanDriver()
    pacer = _Pacer(clock, sleep)
    episodes: Iterable[int] = itertools.count() if max_episodes is None else range(max_episodes)
    written: list[LapFiles] = []
    for episode in episodes:
        episode_seed = seed + episode
        obs, info = env.reset(seed=episode_seed)
        recorder.start_episode(obs, info, seed=episode_seed)
        driver.reset()
        logger.info("episode %d ready (seed=%d): press throttle to start", episode, episode_seed)
        end = _drive_episode(env, recorder, source, driver, render, pacer, written)
        if end is _End.QUIT:
            break
    logger.info("session over: %d lap(s) recorded", len(written))
    return written


def _drive_episode(
    env: LapZeroEnv,
    recorder: LapRecorder,
    source: InputSource,
    driver: HumanDriver,
    render: Callable[[], None],
    pacer: _Pacer,
    written: list[LapFiles],
) -> _End:
    """Run one episode from the grid to its end, appending written laps to `written`."""
    started = False
    while True:
        render()
        pacer.wait()
        frame = source.poll()
        if frame.quit or frame.restart:
            if started:
                logger.info("unfinished lap discarded")
            return _End.QUIT if frame.quit else _End.RESTART
        started = started or frame.throttle > 0.0
        if not started:
            continue
        action = driver.act(frame.steer, frame.throttle)
        obs, reward, terminated, truncated, info = env.step(action)
        laps = recorder.record_step(action, obs, reward, terminated, truncated, info)
        for lap in laps:
            logger.info("lap %s: %s", lap.lap_id, lap.status)
        written += laps
        if terminated or truncated:
            return _End.FINISHED
