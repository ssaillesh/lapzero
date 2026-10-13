"""Unit tests for lapzero/eval/rollout.py: champion lookup, track files, the recorded rollout and
the lap upload.

Laps are driven on a fake simulator (CircuitSimulator moves the car round a 40 m square, so a
lap takes 80 control steps), the registry is a file store in a temporary directory, and uploads go
to a local HTTP server standing in for the API.
"""

from __future__ import annotations

import json
import socket
import threading
from collections.abc import Iterator
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch
from mlflow.exceptions import MlflowException
from torch import nn

from lapzero.eval import rollout
from lapzero.eval.rollout import (
    ChampionVersion,
    LapUploadError,
    drive_lap,
    evaluate_champion_on_track,
    load_track_files,
    resolve_champion,
    track_direction,
    upload_lap,
)
from lapzero.io.lap_arrays import read_lap_arrays
from lapzero.io.telemetry import (
    DriverType,
    LapStatus,
    LapTelemetry,
    TrackDirection,
    TrackInfo,
    read_lap_telemetry,
)
from lapzero.record.recorder import LapFiles, LapRecorder
from lapzero.sim.env import EnvConfig, LapZeroEnv
from lapzero.sim.simulator import Pose, SimState
from lapzero.sim.track import Centreline
from lapzero.training.policy import BCPolicy, save_policy
from lapzero.training.tracking import REGISTERED_MODEL, ModelAlias, mlflow_run
from tests.conftest import FakeSimulator, lap_telemetry_document, register_model_versions

SQUARE_POINTS = [(0.0, 0.0), (10.0, 0.0), (10.0, 10.0), (0.0, 10.0)]  # 40 m, counter-clockwise
SQUARE = Centreline(np.array(SQUARE_POINTS))
SQUARE_INFO = TrackInfo(track_id="square", direction=TrackDirection.CCW, length_m=40.0)
ORIGIN = Pose(0.0, 0.0, 0.0)
RUN_ID = "3f2a9c1d8e7b4a6f9c0d1e2f3a4b5c6d"
CHAMPION = ChampionVersion(REGISTERED_MODEL, "8", RUN_ID, f"runs:/{RUN_ID}/policy")


class CircuitSimulator(FakeSimulator):
    """Drives the car round SQUARE, 0.125 m per physics step: one lap is 80 control steps."""

    METRES_PER_STEP = 0.125

    def step(self, steer_rad: float, v_target: float) -> SimState:
        distance = (self.steps + 1) * self.METRES_PER_STEP
        side, along = divmod(distance % 40.0, 10.0)
        corners = [(0.0, 0.0, 1.0, 0.0), (10.0, 0.0, 0.0, 1.0), (10.0, 10.0, -1.0, 0.0)]
        x0, y0, dx, dy = (*corners, (0.0, 10.0, 0.0, -1.0))[int(side)]
        self.pose = Pose(x0 + dx * along, y0 + dy * along, 0.0)
        self.lap_count = int(distance // 40.0)
        return super().step(steer_rad, v_target)


class FixedOutput(nn.Module):
    """Always outputs the same raw action tensor, which may be out of range or NaN."""

    def __init__(self, steer: float, throttle: float) -> None:
        super().__init__()
        self.register_buffer("action", torch.tensor([steer, throttle]))

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.get_buffer("action").clone()


def constant_policy(directory: Path, steer: float, throttle: float) -> Path:
    """Save a BCPolicy whose action is always [steer, throttle] (each strictly inside +/-1)."""
    policy = BCPolicy(8)
    output = policy.net[-2]
    with torch.no_grad():
        output.weight.zero_()
        output.bias.copy_(torch.atanh(torch.tensor([steer, throttle])))
    path = directory / f"constant_{steer}_{throttle}.pt"
    save_policy(policy, path)
    return path


def write_track(tracks_dir: Path, track_id: str = "square") -> Path:
    """Write tracks_dir/<track_id>/ with a placeholder map yaml and the SQUARE centreline CSV."""
    directory = tracks_dir / track_id
    directory.mkdir(parents=True)
    (directory / f"{track_id}.yaml").write_text("image: map.png\nresolution: 0.05\n")
    rows = "\n".join(f"{x},{y},1.0,1.0" for x, y in SQUARE_POINTS)
    (directory / "centreline.csv").write_text(f"# x_m, y_m, w_tr_right_m, w_tr_left_m\n{rows}\n")
    return directory


def _drive(tmp_path: Path, sim: FakeSimulator, policy: nn.Module, **config: Any) -> LapFiles:
    """Drive one lap round SQUARE as CHAMPION with seed 11, recording under tmp_path/laps."""
    env_config = EnvConfig(**config)
    env = LapZeroEnv(sim, ORIGIN, config=env_config, centreline=SQUARE)
    recorder = LapRecorder(
        SQUARE_INFO, CHAMPION.driver, sim_version="test", env_config=env_config, root=tmp_path
    )
    return drive_lap(env, policy, recorder, seed=11)


@dataclass
class FakeApi:
    """Records POSTed JSON documents and answers each with `status`."""

    url: str = ""
    status: int = 201
    uploads: list[tuple[str, str, dict[str, Any]]] = field(default_factory=list)


@pytest.fixture
def api() -> Iterator[FakeApi]:
    fake = FakeApi()

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self) -> None:
            document = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            fake.uploads.append((self.path, self.headers["Content-Type"], document))
            answer = {"lap_id": document["lap_id"]} if fake.status == 201 else {"detail": "nope"}
            reply = json.dumps(answer).encode()
            self.send_response(fake.status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(reply)))
            self.end_headers()
            self.wfile.write(reply)

        def log_message(self, format: str, *args: Any) -> None:
            """Keep the server's access log out of the test output."""

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    fake.url = f"http://127.0.0.1:{server.server_port}/api"
    yield fake
    server.shutdown()
    server.server_close()
    thread.join()


@pytest.fixture
def tracking_uri(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> str:
    monkeypatch.setenv("MLFLOW_ALLOW_FILE_STORE", "true")  # restored after the test
    return (tmp_path / "mlruns").as_uri()


@pytest.fixture
def registered(tracking_uri: str, tmp_path: Path) -> list[str]:
    """Version 1 (champion, steer -0.2) and 2 (challenger, steer 0.2); their run IDs in order."""
    run_ids = []
    for steer in (-0.2, 0.2):
        checkpoint = constant_policy(tmp_path, steer, 0.5)
        with mlflow_run(tracking_uri=tracking_uri, experiment="test") as tracker:
            tracker.register_policy(checkpoint, REGISTERED_MODEL, {"algorithm": "ppo"})
        run_ids.append(tracker.run_id)
    return run_ids


def test_a_counter_clockwise_centreline_is_ccw_and_its_reverse_is_cw() -> None:
    assert track_direction(SQUARE) is TrackDirection.CCW
    assert track_direction(Centreline(np.array(SQUARE_POINTS[::-1]))) is TrackDirection.CW


def test_track_files_give_the_map_and_the_centreline_length_and_direction(tmp_path: Path) -> None:
    directory = write_track(tmp_path)
    track = load_track_files("square", tmp_path)
    assert track.map_yaml == directory / "square.yaml"
    assert track.centreline.length == pytest.approx(40.0)
    assert track.info == SQUARE_INFO


@pytest.mark.parametrize("track_id", ["../square", "square/..", ""])
def test_a_track_id_that_is_not_a_plain_name_is_refused(tmp_path: Path, track_id: str) -> None:
    write_track(tmp_path)
    with pytest.raises(ValueError, match="track_id"):
        load_track_files(track_id, tmp_path)


@pytest.mark.parametrize("missing", ["square.yaml", "centreline.csv"])
def test_a_track_without_its_map_or_centreline_is_reported_missing(
    tmp_path: Path, missing: str
) -> None:
    (write_track(tmp_path) / missing).unlink()
    with pytest.raises(FileNotFoundError, match=missing):
        load_track_files("square", tmp_path)


def test_the_champion_resolves_to_its_version_run_and_artifact(
    tracking_uri: str, registered: list[str]
) -> None:
    champion = resolve_champion(tracking_uri=tracking_uri)
    assert (champion.model_name, champion.version, champion.run_id) == (
        REGISTERED_MODEL,
        "1",
        registered[0],
    )
    assert champion.source.startswith(f"runs:/{registered[0]}/")


def test_the_champion_drives_as_a_model_named_after_its_version() -> None:
    driver = CHAMPION.driver
    assert driver.type is DriverType.MODEL
    assert (driver.name, driver.model_version, driver.mlflow_run_id) == ("champion_v8", "8", RUN_ID)
    assert driver.human_input_device is None


def test_a_champion_registered_without_a_run_is_refused(tracking_uri: str) -> None:
    register_model_versions(tracking_uri, REGISTERED_MODEL, 1, {ModelAlias.CHAMPION: "1"})
    with pytest.raises(ValueError, match="no MLflow run"):
        resolve_champion(tracking_uri=tracking_uri)


def test_a_model_without_a_champion_raises(tracking_uri: str) -> None:
    register_model_versions(tracking_uri, REGISTERED_MODEL, 1, {ModelAlias.CHALLENGER: "1"})
    with pytest.raises(MlflowException):
        resolve_champion(tracking_uri=tracking_uri)


def test_a_completed_lap_is_recorded_with_its_lap_time_and_seed(tmp_path: Path) -> None:
    lap = _drive(tmp_path, CircuitSimulator(), FixedOutput(0.1, 0.5))
    telemetry = read_lap_telemetry(lap.json_path)
    assert lap.status is LapStatus.COMPLETED
    assert telemetry.result.lap_time_s == pytest.approx(3.2)  # 80 control steps of 0.04 s
    assert telemetry.result.sample_count == 80
    assert telemetry.sim.seed == 11
    assert lap.npz_path.is_file()


def test_a_crash_is_recorded_as_a_crashed_lap(tmp_path: Path, fake_sim: FakeSimulator) -> None:
    fake_sim.collision_at_step = 10
    lap = _drive(tmp_path, fake_sim, FixedOutput(0.0, 1.0))
    assert lap.status is LapStatus.CRASHED
    assert read_lap_telemetry(lap.json_path).result.lap_time_s is None


def test_the_recorded_action_is_the_sanitized_action_the_env_applied(
    tmp_path: Path, fake_sim: FakeSimulator
) -> None:
    lap = _drive(tmp_path, fake_sim, FixedOutput(float("nan"), 5.0), max_episode_s=0.2)
    frames = read_lap_telemetry(lap.json_path).frames
    assert set(frames.steer) == {0.0}  # NaN means straight
    assert set(frames.throttle) == {1.0}  # clipped to full throttle
    np.testing.assert_array_equal(read_lap_arrays(lap.npz_path).actions, [[0.0, 1.0]] * 5)


def test_the_policy_drives_in_eval_mode(tmp_path: Path) -> None:
    policy = FixedOutput(0.0, 0.5)
    _drive(tmp_path, CircuitSimulator(), policy)
    assert not policy.training


def test_a_rollout_needs_one_lap_per_episode(tmp_path: Path, fake_sim: FakeSimulator) -> None:
    with pytest.raises(ValueError, match="laps_per_episode"):
        _drive(tmp_path, fake_sim, FixedOutput(0.0, 0.5), laps_per_episode=3)


def test_an_upload_posts_the_telemetry_json_and_returns_the_lap_id(api: FakeApi) -> None:
    telemetry = LapTelemetry.model_validate(lap_telemetry_document())
    assert upload_lap(telemetry, api.url) == telemetry.lap_id
    ((path, content_type, document),) = api.uploads
    assert (path, content_type) == ("/api/laps", "application/json")
    assert document == telemetry.model_dump(mode="json", by_alias=True)


def test_a_trailing_slash_on_the_api_url_is_ignored(api: FakeApi) -> None:
    upload_lap(LapTelemetry.model_validate(lap_telemetry_document()), api.url + "/")
    assert api.uploads[0][0] == "/api/laps"


@pytest.mark.parametrize("status", [200, 409, 422, 500])
def test_any_answer_but_201_raises_with_the_status_and_detail(api: FakeApi, status: int) -> None:
    api.status = status
    telemetry = LapTelemetry.model_validate(lap_telemetry_document())
    match = f"answered {status} for lap {telemetry.lap_id}.*nope"
    with pytest.raises(LapUploadError, match=match) as raised:
        upload_lap(telemetry, api.url)
    assert raised.value.status == status


def test_an_unreachable_api_raises_without_a_status() -> None:
    with socket.socket() as probe:  # a port nothing listens on once the probe closes
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
    telemetry = LapTelemetry.model_validate(lap_telemetry_document())
    with pytest.raises(LapUploadError, match="could not upload") as raised:
        upload_lap(telemetry, f"http://127.0.0.1:{port}/api", timeout_s=5.0)
    assert raised.value.status is None


@pytest.mark.parametrize("api_url", ["file:///tmp/api", "localhost:8000/api", "http:///api"])
def test_an_api_url_that_is_not_http_is_refused(api_url: str) -> None:
    with pytest.raises(ValueError, match="http or https"):
        upload_lap(LapTelemetry.model_validate(lap_telemetry_document()), api_url)


@dataclass
class EnvCalls:
    """Arguments make_env was called with, and the simulator behind the env it returned."""

    calls: list[tuple[Path, Pose, int]] = field(default_factory=list)
    sim: CircuitSimulator = field(default_factory=CircuitSimulator)


@pytest.fixture
def fake_make_env(monkeypatch: pytest.MonkeyPatch) -> EnvCalls:
    """Replace F1TENTH with CircuitSimulator in the rollout's make_env."""
    recorded = EnvCalls()

    def make_env(
        map_yaml: Path,
        start_pose: Pose,
        *,
        seed: int,
        config: EnvConfig | None = None,
        centreline: Centreline | None = None,
    ) -> LapZeroEnv:
        recorded.calls.append((map_yaml, start_pose, seed))
        return LapZeroEnv(recorded.sim, start_pose, config=config, centreline=centreline)

    monkeypatch.setattr(rollout, "make_env", make_env)
    return recorded


def test_the_champion_drives_one_lap_that_is_written_and_uploaded(
    tmp_path: Path,
    tracking_uri: str,
    registered: list[str],
    fake_make_env: EnvCalls,
    api: FakeApi,
) -> None:
    track_dir = write_track(tmp_path / "tracks")
    lap_id = evaluate_champion_on_track(
        "square",
        api.url,
        tracks_dir=tmp_path / "tracks",
        laps_dir=tmp_path / "laps",
        tracking_uri=tracking_uri,
        seed=7,
    )
    ((_, _, document),) = api.uploads
    telemetry = LapTelemetry.model_validate(document)
    assert telemetry.lap_id == lap_id
    assert read_lap_telemetry(tmp_path / "laps" / "square" / f"{lap_id}.json") == telemetry
    assert telemetry.driver == ChampionVersion(REGISTERED_MODEL, "1", registered[0], "").driver
    assert set(telemetry.frames.steer) == {-0.2}  # the champion's constant steer, not v2's +0.2
    assert telemetry.track == SQUARE_INFO
    assert telemetry.result.status is LapStatus.COMPLETED
    assert telemetry.sim.seed == 7
    assert fake_make_env.calls == [(track_dir / "square.yaml", ORIGIN, 7)]
    assert fake_make_env.sim.closed


def test_a_rejected_upload_raises_after_the_lap_is_written(
    tmp_path: Path,
    tracking_uri: str,
    registered: list[str],
    fake_make_env: EnvCalls,
    api: FakeApi,
) -> None:
    write_track(tmp_path / "tracks")
    api.status = 422
    with pytest.raises(LapUploadError) as raised:
        evaluate_champion_on_track(
            "square",
            api.url,
            tracks_dir=tmp_path / "tracks",
            laps_dir=tmp_path / "laps",
            tracking_uri=tracking_uri,
        )
    assert raised.value.status == 422
    assert len(list((tmp_path / "laps" / "square").glob("*.json"))) == 1
    assert fake_make_env.sim.closed
