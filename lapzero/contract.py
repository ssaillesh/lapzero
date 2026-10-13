"""State space contract constants, version 1.1.0 (docs/contracts/state_space.md).

Every contract value lives here and nowhere else; all other modules import from this file.
Changing any value is a contract change: follow Section 7 of the contract before editing.
"""

from typing import Final

import numpy as np
from numpy.typing import NDArray

CONTRACT_VERSION: Final = "1.1.0"


class ContractViolationError(ValueError):
    """Raised when data crossing a module boundary breaks the state space contract."""


def _read_only(values: list[float]) -> NDArray[np.float32]:
    """Build a float32 array that cannot be modified in place."""
    array = np.array(values, dtype=np.float32)
    array.flags.writeable = False
    return array


# Simulator baseline (Section 0), checked against the installed F1TENTH Gym.
SIM_DT: Final = 0.01  # s, physics timestep (100 Hz)
RAW_BEAMS: Final = 1080
LIDAR_FOV: Final = 4.7  # rad, centred on the car heading
LIDAR_MAX: Final = 30.0  # m
STEER_MAX: Final = 0.4189  # rad, symmetric steering limit
SIM_V_MIN: Final = -5.0  # m/s
SIM_V_MAX: Final = 20.0  # m/s

# Project constants (Section 0).
N_BEAMS: Final = 108
BEAM_BIN: Final = 10  # raw beams per downsampled beam
ACTION_REPEAT: Final = 4  # physics steps per policy step
CTRL_DT: Final = SIM_DT * ACTION_REPEAT  # s, policy runs at 25 Hz
CTRL_HZ: Final = 25  # control (and telemetry recording) rate, Hz
V_MAX_TRAIN: Final = 8.0  # m/s, default target velocity cap
V_NORM: Final = 20.0  # m/s
VY_NORM: Final = 5.0  # m/s
YAW_NORM: Final = 10.0  # rad/s
REVERSE_ENABLED: Final = False

# Observation layout (Section 1.2).
OBS_DIM: Final = 113
LIDAR_SLICE: Final = slice(0, N_BEAMS)
V_LONG: Final = 108
V_LAT: Final = 109
YAW_RATE: Final = 110
PREV_STEER: Final = 111
PREV_THROTTLE: Final = 112
OBS_LOW: Final = _read_only([0.0] * N_BEAMS + [-1.0] * 5)
OBS_HIGH: Final = _read_only([1.0] * OBS_DIM)

# Action space and mapping (Section 2).
ACT_DIM: Final = 2
STEER: Final = 0  # -1 full right, +1 full left
THROTTLE: Final = 1  # +1 full throttle, 0 coast, -1 full brake
ACT_LOW: Final = _read_only([-1.0, -1.0])
ACT_HIGH: Final = _read_only([1.0, 1.0])
STEER_RATE_MAX: Final = 0.15  # rad per control step
ACCEL_STEP: Final = 0.6  # m/s of target velocity per control step at full throttle
BRAKE_STEP: Final = 1.2  # m/s of target velocity per control step at full brake

# Episode rules (Section 3).
START_OFFSET_XY: Final = 0.3  # m, training-only start jitter
START_OFFSET_THETA: Final = 0.1  # rad, training-only start jitter
LAPS_PER_EPISODE_EVAL: Final = 1
LAPS_PER_EPISODE_TRAIN: Final = 3
MAX_EPISODE_S: Final = 120.0  # s
STUCK_SPEED: Final = 0.2  # m/s, below this the car counts as stuck
STUCK_S: Final = 3.0  # s stuck before the episode is truncated
LAP_MIN_PROGRESS: Final = 0.9  # net lap fraction driven forward before a start-line pass counts

# Lap telemetry JSON + .npz (Section 5).
TELEMETRY_SCHEMA: Final = "lapzero.lap_telemetry"
TELEMETRY_SCHEMA_VERSION: Final = "1.0.0"
SIMULATOR_NAME: Final = "f1tenth_gym"
LAPS_DIR: Final = "data/laps"  # files go to LAPS_DIR/<track_id>/<lap_id>.json and .npz
DECIMALS_TIME: Final = 2  # t and lap_time_s
DECIMALS_POSITION: Final = 3  # x, y, track length
DECIMALS_ANGLE: Final = 4  # theta, steer_rad
DECIMALS_SPEED: Final = 3  # v_long, v_target, speed summaries
DECIMALS_ACTION: Final = 3  # steer, throttle
DECIMALS_PROGRESS: Final = 4  # not listed in Section 5.2; matches the Section 5.1 example

# Replay trajectory JSON (Section 6).
REPLAY_SCHEMA: Final = "lapzero.replay"
REPLAY_SCHEMA_VERSION: Final = "1.0.0"
REPLAY_HZ: Final = 10  # replay sample rate, Hz
