"""Champion rollout: drive the registry champion for one lap on a track and upload it to the API.

The champion alias is resolved once to a model version and that version's artifact is loaded,
so the uploaded telemetry names the weights that actually drove even if the gate promotes a new
champion meanwhile. Every registered version is a lapzero_policy checkpoint whatever algorithm
trained it (PPO actors are exported to the BC format), so it loads with load_policy_source.

The lap starts from the track's fixed start pose at the simulator's default friction, as human
laps do, and the policy drives its deterministic action. LapRecorder writes the lap to
<laps_dir>/<track_id>/<lap_id>.json + .npz (behaviour cloning reads only human laps), then the
validated telemetry is POSTed to <api_url>/laps.
"""

from __future__ import annotations

import importlib.metadata
import logging
import re
import urllib.parse
from dataclasses import dataclass
from http import HTTPStatus
from http.client import HTTPConnection, HTTPException, HTTPSConnection
from pathlib import Path
from typing import Final

import numpy as np
import torch
from torch import nn

from lapzero.contract import DECIMALS_POSITION, LAPS_DIR
from lapzero.inference.engine import load_policy_source
from lapzero.io.telemetry import (
    TRACK_ID_PATTERN,
    DriverInfo,
    DriverType,
    LapTelemetry,
    TrackDirection,
    TrackInfo,
    read_lap_telemetry,
)
from lapzero.record.recorder import LapFiles, LapRecorder
from lapzero.registry import get_tracking_uri
from lapzero.sim.env import EnvConfig, LapZeroEnv, sanitize_action
from lapzero.sim.f1tenth import make_env
from lapzero.sim.track import Centreline, load_centreline, start_pose_from_centreline
from lapzero.training.tracking import REGISTERED_MODEL, ModelAlias, mlflow_client

logger = logging.getLogger(__name__)

DEFAULT_API_URL: Final = "http://localhost:8000/api"
"""Base URL of the LapZero API; laps are POSTed to <base>/laps."""
TRACKS_DIR: Final = "data/tracks"
"""Default root of track files: <track_id>/<track_id>.yaml (ROS map) and centreline.csv."""
CENTRELINE_FILE: Final = "centreline.csv"
UPLOAD_TIMEOUT_S: Final = 30.0
"""Seconds to wait for the API to connect and to answer a lap upload."""
_SIM_PACKAGE: Final = "f1tenth-gym"
_CONNECTIONS: Final[dict[str, type[HTTPConnection]]] = {
    "http": HTTPConnection,
    "https": HTTPSConnection,
}


class LapUploadError(RuntimeError):
    """Raised when the API does not answer a lap upload with 201 Created.

    Attributes:
        status: HTTP status the API answered, or None if it could not be reached.
    """

    def __init__(self, message: str, status: int | None = None) -> None:
        super().__init__(message)
        self.status = status


@dataclass(frozen=True, slots=True)
class ChampionVersion:
    """The model version the champion alias pointed at when it was resolved.

    Attributes:
        model_name: Registered model name.
        version: Model version number, e.g. "8".
        run_id: MLflow run that registered the version.
        source: Artifact URI of the version's lapzero_policy model, e.g. "runs:/<run_id>/policy".
    """

    model_name: str
    version: str
    run_id: str
    source: str

    @property
    def driver(self) -> DriverInfo:
        """Telemetry driver block of this version, named "champion_v<version>"."""
        return DriverInfo(
            type=DriverType.MODEL,
            name=f"champion_v{self.version}",
            model_version=self.version,
            mlflow_run_id=self.run_id,
        )


@dataclass(frozen=True, slots=True, eq=False)  # Centreline compares by identity
class TrackFiles:
    """A track's simulator map, its centreline and the telemetry track block they define.

    Attributes:
        map_yaml: ROS map yaml the simulator loads.
        centreline: Centreline in driving order, metres.
        info: Track block of the lap telemetry (length_m in metres).
    """

    map_yaml: Path
    centreline: Centreline
    info: TrackInfo


def resolve_champion(*, tracking_uri: str, model_name: str = REGISTERED_MODEL) -> ChampionVersion:
    """Return the model version the champion alias points at.

    Args:
        tracking_uri: MLflow tracking URI holding the registry.
        model_name: Registered model name.

    Returns:
        The champion's version, run ID and artifact URI.

    Raises:
        ValueError: If the version has no MLflow run (which model telemetry must name) or no
            artifact source.
        mlflow.exceptions.MlflowException: If the model or its champion alias does not exist.
    """
    version = mlflow_client(tracking_uri).get_model_version_by_alias(
        model_name, ModelAlias.CHAMPION
    )
    if not (version.run_id and version.source):
        raise ValueError(
            f"{model_name} version {version.version} ({ModelAlias.CHAMPION}) has no MLflow run "
            f"ID or artifact source (run {version.run_id!r}, source {version.source!r}); a "
            "rollout needs both to load it and to name it in the lap telemetry"
        )
    return ChampionVersion(model_name, str(version.version), version.run_id, version.source)


def track_direction(centreline: Centreline) -> TrackDirection:
    """Return the racing direction of a centreline from the sign of its shoelace area.

    Args:
        centreline: Closed centreline in driving order, in the world (map) frame.

    Returns:
        CCW if the loop encloses a positive area (y is to the left of x), otherwise CW.
    """
    x, y = centreline.points[:, 0], centreline.points[:, 1]
    twice_area = float(np.dot(x, np.roll(y, -1)) - np.dot(np.roll(x, -1), y))
    return TrackDirection.CCW if twice_area > 0.0 else TrackDirection.CW


def load_track_files(track_id: str, tracks_dir: Path = Path(TRACKS_DIR)) -> TrackFiles:
    """Find a track's map and read its centreline from tracks_dir/<track_id>/.

    Args:
        track_id: Track ID, e.g. "spielberg"; also its directory and map file name.
        tracks_dir: Root holding <track_id>/<track_id>.yaml and <track_id>/centreline.csv.

    Returns:
        The map path, the centreline, and the track block with the centreline's length (metres)
        and direction.

    Raises:
        ValueError: If `track_id` is not a valid track ID or the centreline is invalid.
        FileNotFoundError: If the map yaml or the centreline CSV is missing.
    """
    if re.fullmatch(TRACK_ID_PATTERN, track_id) is None:
        raise ValueError(f"track_id {track_id!r} does not match {TRACK_ID_PATTERN}")
    directory = tracks_dir / track_id
    map_yaml = directory / f"{track_id}.yaml"
    if not map_yaml.is_file():
        raise FileNotFoundError(f"track {track_id!r} has no map file {map_yaml}")
    centreline_csv = directory / CENTRELINE_FILE
    if not centreline_csv.is_file():
        raise FileNotFoundError(f"track {track_id!r} has no centreline file {centreline_csv}")
    centreline = load_centreline(centreline_csv)
    info = TrackInfo(
        track_id=track_id,
        direction=track_direction(centreline),
        length_m=round(centreline.length, DECIMALS_POSITION),
    )
    return TrackFiles(map_yaml, centreline, info)


def drive_lap(env: LapZeroEnv, policy: nn.Module, recorder: LapRecorder, *, seed: int) -> LapFiles:
    """Drive one episode with the policy's deterministic action and record its lap.

    The episode runs until the lap is completed or the env ends it (crash, timeout, stuck car).
    Actions are sanitized before they reach the env, so the recorder stores what was applied.

    Args:
        env: Env with laps_per_episode == 1.
        policy: Maps a (OBS_DIM,) float32 observation tensor to a (ACT_DIM,) action tensor.
        recorder: Built with the env's EnvConfig; writes the lap's .json and .npz files.
        seed: Seed for env.reset; written to the lap telemetry.

    Returns:
        The files of the lap, completed or not.

    Raises:
        ValueError: If the env runs more than one lap per episode.
        ContractViolationError: If an observation or reward breaks the contract.
    """
    if env.config.laps_per_episode != 1:
        raise ValueError(
            f"a rollout needs laps_per_episode == 1, got {env.config.laps_per_episode}"
        )
    policy.eval()
    obs, info = env.reset(seed=seed)
    recorder.start_episode(obs, info, seed=seed)
    written: list[LapFiles] = []
    terminated = truncated = False
    while not (terminated or truncated):
        with torch.no_grad():
            action = sanitize_action(policy(torch.from_numpy(obs)).numpy())
        obs, reward, terminated, truncated, info = env.step(action)
        written += recorder.record_step(action, obs, reward, terminated, truncated, info)
    (lap,) = written  # with one lap per episode, the episode ends on the step that writes it
    return lap


def upload_lap(
    telemetry: LapTelemetry,
    api_url: str = DEFAULT_API_URL,
    *,
    timeout_s: float = UPLOAD_TIMEOUT_S,
) -> str:
    """POST a lap's telemetry JSON to <api_url>/laps.

    Args:
        telemetry: The lap, as written by LapRecorder.
        api_url: Base URL of the API, e.g. "http://localhost:8000/api".
        timeout_s: Seconds to wait for the connection and for the answer.

    Returns:
        The lap_id of the stored lap.

    Raises:
        ValueError: If `api_url` is not an http or https URL.
        LapUploadError: If the API cannot be reached or answers anything but 201 Created
            (409 for a lap already uploaded, 422 for telemetry it rejects).
    """
    url = f"{api_url.rstrip('/')}/laps"
    parts = urllib.parse.urlsplit(url)
    connection_type = _CONNECTIONS.get(parts.scheme)
    if connection_type is None or not parts.hostname:
        raise ValueError(f"api_url must be an http or https URL, got {api_url!r}")
    connection = connection_type(parts.hostname, parts.port, timeout=timeout_s)
    body = telemetry.model_dump_json(by_alias=True).encode()
    try:
        connection.request(
            "POST", parts.path, body=body, headers={"Content-Type": "application/json"}
        )
        response = connection.getresponse()
        status, detail = response.status, response.read().decode(errors="replace")
    except (OSError, HTTPException) as error:
        raise LapUploadError(
            f"could not upload lap {telemetry.lap_id} to {url}: {error}"
        ) from error
    finally:
        connection.close()
    if status != HTTPStatus.CREATED:
        raise LapUploadError(
            f"POST {url} answered {status} for lap {telemetry.lap_id}: {detail}", status
        )
    logger.info("uploaded lap %s to %s", telemetry.lap_id, url)
    return telemetry.lap_id


def evaluate_champion_on_track(
    track_id: str,
    api_url: str = DEFAULT_API_URL,
    *,
    tracks_dir: Path = Path(TRACKS_DIR),
    laps_dir: Path = Path(LAPS_DIR),
    tracking_uri: str | None = None,
    model_name: str = REGISTERED_MODEL,
    seed: int = 0,
) -> str:
    """Drive the registry champion for one lap on a track and upload the lap to the API.

    Completed and failed laps (crashed, timeout, stuck) are both uploaded, with their status.

    Args:
        track_id: Track to drive, found under tracks_dir (see load_track_files).
        api_url: Base URL of the API; the lap is POSTed to <api_url>/laps.
        tracks_dir: Root of the track files.
        laps_dir: Where LapRecorder writes <track_id>/<lap_id>.json and .npz.
        tracking_uri: MLflow tracking URI; defaults to the repo-root store (get_tracking_uri).
        model_name: Registered model whose champion drives.
        seed: Simulator seed (LiDAR noise) and episode seed; logged and written to the lap.

    Returns:
        The lap_id of the uploaded lap.

    Raises:
        ValueError: If track_id is invalid, the champion has no MLflow run or is not a
            lapzero_policy model, or api_url is not an http(s) URL.
        FileNotFoundError: If the track's map or centreline file is missing.
        ContractViolationError: If the policy was saved under another contract major version.
        mlflow.exceptions.MlflowException: If the model or its champion alias does not exist.
        LapUploadError: If the API does not store the lap.
    """
    track = load_track_files(track_id, tracks_dir)
    uri = tracking_uri or get_tracking_uri()
    champion = resolve_champion(tracking_uri=uri, model_name=model_name)
    policy = load_policy_source(champion.source, tracking_uri=uri)
    config = EnvConfig()
    recorder = LapRecorder(
        track.info,
        champion.driver,
        sim_version=importlib.metadata.version(_SIM_PACKAGE),
        env_config=config,
        root=laps_dir,
    )
    logger.info(
        "driving %s version %s (%s) on %s with seed=%d",
        model_name,
        champion.version,
        ModelAlias.CHAMPION,
        track_id,
        seed,
    )
    env = make_env(
        track.map_yaml,
        start_pose_from_centreline(track.centreline),
        seed=seed,
        config=config,
        centreline=track.centreline,
    )
    try:
        lap = drive_lap(env, policy, recorder, seed=seed)
    finally:
        env.close()
    telemetry = read_lap_telemetry(lap.json_path)
    logger.info("champion lap %s on %s: %s", lap.lap_id, track_id, lap.status)
    return upload_lap(telemetry, api_url)
