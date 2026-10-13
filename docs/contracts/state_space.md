# State Space Contract

**Contract version:** `1.1.0`
**Status:** Draft, locked once `tests/contracts/` passes
**Applies to:** `sim/` (env wrapper), `training/` (BC + RL), `eval/` (release gate), `backend/` (API + DB), `frontend/` (replay viewer)

This document is the single source of truth for what crosses the boundary between the simulator, the learning code, and the backend. Any change to a shape, bound, unit, or field name is a contract change and must follow [Versioning](#7-versioning-rules).

---

## 0. Conventions

| Rule | Value |
|---|---|
| Units | SI only: metres, seconds, radians, m/s, rad/s |
| Float type | `float32` for everything entering or leaving a policy network; `float64` allowed inside the simulator only |
| Angles | Radians, counter-clockwise positive, wrapped to `(-π, π]` |
| Frame | Car frame: `x` forward, `y` left. World frame: the track map frame used by F1TENTH Gym |
| Time | Seconds from the start of the episode (`t = 0.0` at reset) |
| Missing or invalid numbers | Never `NaN` or `inf` in any observation, action, or JSON file. See replacement rules per field. |

### Simulator baseline (F1TENTH Gym)

These are the raw simulator values the contract is built on. They come from the default F1TENTH Gym configuration. **Verify them against the installed version on day 1** with `scripts/check_sim_params.py` (Section 8), and if any differ, update this table and bump the contract version.

| Constant | Symbol | Default | Notes |
|---|---|---|---|
| Physics timestep | `SIM_DT` | 0.01 s | 100 Hz |
| LiDAR beams | `RAW_BEAMS` | 1080 | |
| LiDAR field of view | `LIDAR_FOV` | 4.7 rad (about 270°) | Centred on car heading |
| LiDAR max range | `LIDAR_MAX` | 30.0 m | |
| Steering limits | `STEER_MAX` | ±0.4189 rad (about ±24°) | `s_min`, `s_max` in vehicle params |
| Velocity limits | `SIM_V_MIN`, `SIM_V_MAX` | −5.0, 20.0 m/s | Sim accepts a **target velocity**, not torque |

> Important: F1TENTH Gym's native action is `[steering_angle, target_velocity]`. There is no direct throttle or brake input. This contract exposes a normalized throttle/brake command to the policy and converts it to a target velocity inside the env wrapper (Section 2.3). The policy never sees the raw simulator action.

### Project constants (owned by this contract)

| Constant | Value | Why |
|---|---|---|
| `N_BEAMS` | 108 | 1080 / 10. Small enough for an MLP, keeps 2.5° resolution |
| `BEAM_BIN` | 10 | Raw beams per downsampled beam |
| `ACTION_REPEAT` | 4 | Policy runs at 25 Hz (`CTRL_DT = 0.04 s`) |
| `V_MAX_TRAIN` | 8.0 m/s | Training speed cap. Raise later (config), never above `SIM_V_MAX` |
| `V_NORM` | 20.0 m/s | Normalization constant for speed (fixed, independent of `V_MAX_TRAIN` so models stay comparable) |
| `VY_NORM` | 5.0 m/s | Lateral speed normalization |
| `YAW_NORM` | 10.0 rad/s | Yaw rate normalization |
| `REVERSE_ENABLED` | `false` | Policy cannot command reverse in v1 |

---

## 1. Observation space

### 1.1 Definition

```python
observation_space = gymnasium.spaces.Box(
    low=OBS_LOW,     # shape (113,), float32
    high=OBS_HIGH,   # shape (113,), float32
    shape=(113,),
    dtype=np.float32,
)
```

A flat vector, so any MLP policy and Stable-Baselines3 `MlpPolicy` consume it directly.

### 1.2 Layout

| Index | Name | Raw source | Raw unit | Normalization | Bounds after normalization |
|---|---|---|---|---|---|
| `0 … 107` | `lidar[0..107]` | `obs["scans"][0]`, downsampled | m | `min(r, LIDAR_MAX) / LIDAR_MAX` | `[0.0, 1.0]` |
| `108` | `v_long` | `obs["linear_vels_x"][0]` | m/s | `v / V_NORM`, clipped | `[-1.0, 1.0]` |
| `109` | `v_lat` | `obs["linear_vels_y"][0]` | m/s | `v / VY_NORM`, clipped | `[-1.0, 1.0]` |
| `110` | `yaw_rate` | `obs["ang_vels_z"][0]` | rad/s | `w / YAW_NORM`, clipped | `[-1.0, 1.0]` |
| `111` | `prev_steer` | last applied policy action `a[0]` | normalized | already normalized | `[-1.0, 1.0]` |
| `112` | `prev_throttle` | last applied policy action `a[1]` | normalized | already normalized | `[-1.0, 1.0]` |

`OBS_LOW = [0.0]*108 + [-1.0]*5`
`OBS_HIGH = [1.0]*113`

Index ranges are exported as constants so no code hard-codes them:

```python
LIDAR_SLICE = slice(0, 108)
V_LONG, V_LAT, YAW_RATE, PREV_STEER, PREV_THROTTLE = 108, 109, 110, 111, 112
OBS_DIM = 113
```

### 1.3 LiDAR downsampling rule

1. Take the 1080 raw ranges for the ego car.
2. Replace `NaN`, `inf`, and any value `> LIDAR_MAX` with `LIDAR_MAX`. Replace negatives with `0.0`.
3. Reshape to `(108, 10)` and take the **minimum** of each bin.
   *Minimum, not mean:* the policy must see the closest obstacle in each sector. Averaging hides thin walls and other cars.
4. Divide by `LIDAR_MAX`.
5. Index order: beam `0` is the right-most ray (−FOV/2), beam `107` is the left-most (+FOV/2), sweeping counter-clockwise. This matches F1TENTH Gym's raw ordering; the wrapper must not reverse it.

Beam `i` points at angle `θ_i = -LIDAR_FOV/2 + (i + 0.5) * LIDAR_FOV / 108` relative to the car heading. The frontend uses this to draw scans.

### 1.4 Previous action

`prev_steer` and `prev_throttle` are the **normalized policy action** from the previous control step (Section 2), not the simulator command. On `reset()` both are `0.0`.

They give a feed-forward policy enough memory to produce smooth control without frame stacking.

### 1.5 Frame stacking (optional, off by default)

If enabled with `FRAME_STACK = k`, the observation becomes shape `(113 * k,)`, oldest first. This is a contract change (minor version bump) and must be recorded in the model's MLflow params as `obs_frame_stack`.

### 1.6 Not in the observation (on purpose)

| Field | Reason |
|---|---|
| World pose `x, y, θ` | Policy would memorize the track and fail on unseen tracks. Used only for reward, logging, and replay. |
| Lap time, lap count | Reward signal only |
| Collision flag | Ends the episode; not a feature |

---

## 2. Action space

### 2.1 Definition

```python
action_space = gymnasium.spaces.Box(
    low=np.array([-1.0, -1.0], dtype=np.float32),
    high=np.array([1.0, 1.0], dtype=np.float32),
    shape=(2,),
    dtype=np.float32,
)
```

| Index | Name | Range | Meaning |
|---|---|---|---|
| `0` | `steer` | `[-1, 1]` | `-1` full right, `+1` full left |
| `1` | `throttle` | `[-1, 1]` | `+1` full throttle, `0` coast, `-1` full brake |

Both the human keyboard driver and every model emit actions in this exact space, so recorded human laps and model actions are interchangeable for behaviour cloning.

### 2.2 Steering mapping

```
steer_rad = clip(a[0], -1, 1) * STEER_MAX          # STEER_MAX = 0.4189 rad
```

Optional rate limit (on by default, `STEER_RATE_MAX = 0.15 rad` per control step) to stop unrealistic bang-bang steering:

```
steer_rad = clip(steer_rad, prev_steer_rad - STEER_RATE_MAX, prev_steer_rad + STEER_RATE_MAX)
```

### 2.3 Throttle and brake mapping to target velocity

The simulator wants a target velocity. The wrapper converts the throttle command each control step:

```
a = clip(action[1], -1, 1)
v = current longitudinal speed (m/s)

if a >= 0:
    v_target = v + a * ACCEL_STEP            # ACCEL_STEP = 0.6 m/s per control step
else:
    v_target = v + a * BRAKE_STEP            # BRAKE_STEP = 1.2 m/s per control step (brakes harder)

v_target = clip(v_target, 0.0 if not REVERSE_ENABLED else SIM_V_MIN, V_MAX_TRAIN)
```

So throttle behaves like a pedal: holding `+1` accelerates at up to 15 m/s² (0.6 / 0.04 s), holding `-1` brakes at up to 30 m/s², and `0` holds the current speed. The simulator's own acceleration limits may cap these further; the contract only fixes the mapping, not the achieved acceleration.

`ACCEL_STEP` and `BRAKE_STEP` are tunable but must be logged as MLflow params; changing their defaults is a minor version bump.

### 2.4 Control loop timing

```
policy step (25 Hz) ─► map action ─► repeat sim.step() ACTION_REPEAT=4 times at 100 Hz
                                      ─► stop early if collision
                                      ─► build observation from the last sim step
```

Rewards are summed over the repeated sim steps.

---

## 3. Episode rules

| Rule | Value |
|---|---|
| Start | Fixed start pose per track from `tracks/<track_id>/start.yaml`; optional random offset `±0.3 m`, `±0.1 rad` in training only |
| Terminated (`terminated=True`) | Collision, or `LAPS_PER_EPISODE` laps completed (default 1 for eval, 3 for training) |
| Truncated (`truncated=True`) | `MAX_EPISODE_S = 120 s` reached, or car stuck (`v_long < 0.2 m/s` for 3 s) |
| Lap completion | The simulator reports a pass through the start line (F1TENTH Gym `lap_counts` increment) **and** the car has driven at least `LAP_MIN_PROGRESS = 0.90` of the centreline since reset or the previous lap, net of any distance driven backwards. A U-turn or spin back through the start line, or a lap driven the wrong way, is not a lap and does not end the episode. An env built without a centreline (human recording, unit tests) cannot measure progress and counts every simulator lap. |

The reward function is **not** part of this contract (it will change often). It lives in `sim/reward.py` and its version is logged per run.

---

## 4. `info` dict returned by `step()`

Not seen by the policy. Used by logging, the eval runner, and the telemetry writer.

```python
info = {
    "t": float,              # s since reset
    "x": float, "y": float,  # m, world frame
    "theta": float,          # rad
    "v_long": float,         # m/s (raw, not normalized)
    "steer_rad": float,      # applied steering, rad
    "v_target": float,       # commanded target velocity, m/s
    "lap_count": int,        # laps completed since reset (Section 3 lap completion rule)
    "lap_time": float,       # s, current lap so far
    "collision": bool,
    "progress": float,       # 0..1 along the track centreline
}
```

---

## 5. Lap telemetry JSON (full recording)

One file per completed or failed lap. Used for behaviour cloning metadata, the eval runner, and debugging. Recorded at the **control rate (25 Hz)**.

Raw observations are **not** stored in JSON (113 floats × 25 Hz bloats files). They go in a sibling `.npz` file with the same `lap_id` for training. The JSON stores everything a human or the backend needs.

**File name:** `data/laps/<track_id>/<lap_id>.json` (+ `<lap_id>.npz` for obs and actions)

### 5.1 Example

```json
{
  "schema": "lapzero.lap_telemetry",
  "schema_version": "1.0.0",
  "contract_version": "1.0.0",
  "lap_id": "01J9ZK8V3Q7M2T6C1R4X5N8B0A",
  "episode_id": "01J9ZK8T9E0Y3H2P6W1D7F4G5S",
  "lap_index": 1,
  "created_at": "2026-10-20T14:32:05Z",

  "track": {
    "track_id": "track_02",
    "direction": "ccw",
    "length_m": 82.4
  },

  "driver": {
    "type": "model",
    "name": "lapzero-ppo",
    "model_version": "7",
    "mlflow_run_id": "3f2a9c1d8e7b4a6f9c0d1e2f3a4b5c6d",
    "human_input_device": null
  },

  "sim": {
    "simulator": "f1tenth_gym",
    "sim_version": "0.2.1",
    "sim_dt": 0.01,
    "action_repeat": 4,
    "ctrl_hz": 25,
    "v_max_train": 8.0,
    "seed": 42
  },

  "result": {
    "status": "completed",
    "lap_time_s": 35.97,
    "collision": false,
    "off_track_events": 0,
    "max_speed_mps": 7.84,
    "mean_speed_mps": 6.12,
    "sample_count": 900
  },

  "frames": {
    "t":          [0.00, 0.04, 0.08],
    "x":          [0.000, 0.051, 0.118],
    "y":          [0.000, 0.002, 0.006],
    "theta":      [0.0000, 0.0031, 0.0074],
    "v_long":     [0.00, 0.58, 1.17],
    "steer":      [0.000, 0.040, 0.062],
    "throttle":   [1.000, 1.000, 0.950],
    "steer_rad":  [0.0000, 0.0168, 0.0260],
    "v_target":   [0.60, 1.18, 1.74],
    "progress":   [0.0000, 0.0006, 0.0014]
  }
}
```

### 5.2 Field rules

| Field | Type | Rule |
|---|---|---|
| `lap_id`, `episode_id` | string | ULID (sortable by time) |
| `created_at` | string | ISO 8601 UTC with `Z` |
| `driver.type` | enum | `"human"` or `"model"` |
| `driver.model_version`, `driver.mlflow_run_id` | string or null | Required when `type = "model"`, `null` for humans |
| `driver.human_input_device` | enum or null | `"keyboard"` or `"gamepad"` for humans, `null` for models |
| `result.status` | enum | `"completed"`, `"crashed"`, `"timeout"`, `"stuck"` |
| `result.lap_time_s` | number or null | `null` unless `status = "completed"` |
| `frames` | object of arrays | **Columnar**: every array has length `result.sample_count`. Columnar is about 40% smaller than an array of objects and loads straight into NumPy or pandas. |
| `frames.steer`, `frames.throttle` | number | The **normalized policy action** in `[-1, 1]` (Section 2.1) |
| `frames.steer_rad`, `frames.v_target` | number | The **applied simulator command** after mapping |
| Precision | | `t` 2 dp, positions 3 dp, angles 4 dp, speeds and actions 3 dp |

### 5.3 Companion `.npz` (training only)

| Key | Shape | dtype |
|---|---|---|
| `obs` | `(N, 113)` | float32 |
| `actions` | `(N, 2)` | float32 |
| `rewards` | `(N,)` | float32 |
| `dones` | `(N,)` | bool |
| `contract_version` | `()` | str |

`N` equals `result.sample_count` in the JSON. Behaviour cloning loads `obs` and `actions` from here.

---

## 6. Replay trajectory JSON (API and frontend)

A lighter version served by `GET /api/replays/{lap_id}` for the browser replay. Downsampled to **10 Hz** and stripped to what the canvas needs.

```json
{
  "schema": "lapzero.replay",
  "schema_version": "1.0.0",
  "lap_id": "01J9ZK8V3Q7M2T6C1R4X5N8B0A",
  "track_id": "track_02",
  "driver_label": "AI v7",
  "lap_time_s": 35.97,
  "hz": 10,
  "points": {
    "t":     [0.0, 0.1, 0.2],
    "x":     [0.000, 0.162, 0.412],
    "y":     [0.000, 0.009, 0.031],
    "theta": [0.0000, 0.0110, 0.0240],
    "v":     [0.00, 1.46, 2.81],
    "steer": [0.000, 0.071, 0.090]
  }
}
```

Rules:
- Built from the lap telemetry by taking every `ctrl_hz / hz` frame (25 → 10 Hz uses linear interpolation on `t`).
- The ghost-lap comparison aligns two replays by `t`; the frontend interpolates between points for smooth 60 fps animation.
- Track geometry is **not** embedded. It comes from `GET /api/tracks/{track_id}` (centreline and boundaries) so it is downloaded once.

### 6.1 Database mapping (backend)

| Table | Holds |
|---|---|
| `laps` | `lap_id`, `track_id`, driver fields, `result.*`, `created_at`, path to telemetry file |
| `replays` | `lap_id` and the replay JSON (JSONB), generated on ingest |
| `models` | `model_version`, `mlflow_run_id`, `contract_version`, gate result |

The backend rejects any upload whose `contract_version` major number does not match its own.

---

## 7. Versioning rules

Semantic versioning on `contract_version`:

| Change | Bump | Example |
|---|---|---|
| Change a shape, index, unit, or normalization constant | **Major** | `N_BEAMS` 108 → 54; reorder observation indices |
| Add an optional field or feature behind a flag | **Minor** | Enable frame stacking; add `frames.yaw_rate` to telemetry |
| Clarify wording, fix a typo | **Patch** | |

Every trained model logs `contract_version` as an MLflow param. The release gate refuses to compare a challenger and champion with different major versions.

| Version | Change |
|---|---|
| `1.1.0` | Lap completion (Section 3) needs `LAP_MIN_PROGRESS = 0.90` of the lap driven, net, before a start-line pass counts; `info["lap_count"]` (Section 4) counts only those laps. Before, any pass through F1TENTH's start gate counted, in either direction. |

---

## 8. Enforcement (tests that lock this contract)

All live in `tests/contracts/` and run in CI on every push.

| Test | Checks |
|---|---|
| `test_obs_space.py` | `env.observation_space.shape == (113,)`, dtype float32, bounds match Section 1.2 |
| `test_obs_values.py` | 500 random steps: every observation is finite and inside its bounds |
| `test_lidar_downsample.py` | Synthetic scan with one close point in bin 37 → `obs[37]` equals that distance / 30; `NaN` and `inf` map to `1.0` |
| `test_action_space.py` | Shape `(2,)`, bounds `[-1, 1]`; `a = [1, 0]` gives `steer_rad = 0.4189`; out-of-range actions are clipped, never raise |
| `test_throttle_mapping.py` | `+1` raises `v_target` by `ACCEL_STEP`, `-1` lowers it by `BRAKE_STEP`, never below 0 or above `V_MAX_TRAIN` |
| `test_telemetry_schema.py` | Recorded lap validates against `schemas/lap_telemetry.schema.json`; all `frames` arrays equal length; no `NaN` |
| `test_replay_schema.py` | Replay validates against `schemas/replay.schema.json`; `hz = 10` |
| `test_human_model_parity.py` | Keyboard driver output passes the same action-space check as a model |
| `test_lap_completion.py` | A full lap counts and ends a one-lap episode; a start-line pass before 90% of the lap, a U-turn or spin back through it, and a lap driven backwards do not |

`scripts/check_sim_params.py` prints the installed F1TENTH Gym values for every row in the simulator baseline table and exits non-zero if any differ from this document.

---

## 9. Open decisions (resolve before v1.0.0 is locked)

1. Confirm the simulator baseline values with `check_sim_params.py` on the installed version.
2. Confirm `V_MAX_TRAIN = 8.0` is stable for behaviour cloning on all three tracks; raise only after BC laps complete.
3. Decide whether reverse is ever needed (recovery from walls). Default: no.
4. Decide whether to store raw 1080-beam scans for a future CNN policy (would add `scans_raw` to the `.npz`, minor bump).