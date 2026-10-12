# CLAUDE.md: LapZero

Instructions for Claude when working in this repository. Read this file fully before every task. If anything here conflicts with a prompt, stop and ask.

---

## 1. What this project is

LapZero trains an AI race driver in the F1TENTH Gym simulator. It first copies human laps (behaviour cloning), then improves with reinforcement learning (PPO or SAC). A CI release gate promotes a new model only if it is faster **and** no less safe than the current champion. A FastAPI backend and React dashboard show lap times, a leaderboard, model history, and ghost-lap replays.

Source of truth documents, in priority order:

1. `docs/contracts/state_space.md`: observation space, action space, telemetry and replay JSON. **Binding.**
2. `README.md`: scope, stack, roadmap.
3. This file: how to work.

---

## 2. Hard rules (never break these)

1. **Only touch files relevant to the active prompt.** Do not refactor, reformat, rename, or "tidy" anything outside the task. If you notice an unrelated problem, mention it at the end of your reply; do not fix it.
2. **Never change a test to make it pass.** If a test fails, fix the implementation. You may only change a test's expected behaviour after explaining why the test is wrong and getting explicit approval in the conversation. Fixing a typo or import in a test is allowed if you say so.
3. **Never change the state space contract silently.** Any change to shapes, indices, bounds, units, normalization constants, or JSON field names requires: approval, an update to `docs/contracts/state_space.md`, a `contract_version` bump per its Section 7, and updated contract tests.
4. **No new dependencies without asking.** State the package, version, why, and what it replaces. The allowed stack is in Section 4.
5. **No secrets in code or commits.** Config comes from environment variables loaded via `backend/app/config.py` (pydantic-settings). `.env` is gitignored; `.env.example` lists keys with dummy values.
6. **Don't claim a task is done until the checks in Section 8 pass.** Show the command output summary in your reply.
7. **No `NaN` or `inf`** may cross a module boundary (observations, actions, rewards, JSON). Sanitize at the source.
8. **Ask instead of guessing** when a requirement is ambiguous, a contract value seems wrong, or a change would span more than one top-level folder.

---

## 3. Repository layout

```
lapzero/
├── CLAUDE.md
├── README.md
├── pyproject.toml              # one Python project: deps, ruff, mypy, pytest config
├── docker-compose.yml          # api, db, mlflow, frontend
├── .github/workflows/
│   ├── ci.yml                  # lint, type check, tests (Python + frontend)
│   └── release-gate.yml        # evaluates challenger vs champion
├── docs/
│   └── contracts/state_space.md
├── schemas/                    # JSON Schema files generated from / matching the contract
│   ├── lap_telemetry.schema.json
│   └── replay.schema.json
├── lapzero/                    # importable Python package
│   ├── contract.py             # ALL contract constants live here and nowhere else
│   ├── sim/                    # env wrapper, lidar downsampling, action mapping, reward, tracks
│   ├── record/                 # keyboard/gamepad driver, telemetry writer (.json + .npz)
│   ├── training/               # behaviour cloning, PPO/SAC, callbacks, configs
│   ├── eval/                   # eval runner, release gate logic
│   └── io/                     # telemetry/replay readers, schema validation
├── backend/
│   ├── app/                    # FastAPI: main.py, routers/, models/, schemas/, services/, config.py
│   └── alembic/                # migrations
├── frontend/                   # React + Vite + TypeScript
│   └── src/
├── scripts/                    # CLI entry points only, thin wrappers over lapzero/
│   └── check_sim_params.py
├── tracks/<track_id>/          # map files, start.yaml, centreline.csv
├── data/                       # gitignored: laps/, replays/
├── mlruns/                     # gitignored: local MLflow store
└── tests/
    ├── contracts/              # contract tests (Section 8 of state_space.md)
    ├── unit/                   # mirrors lapzero/ and backend/app/ structure
    └── integration/            # env rollouts, API + DB, gate end to end
```

Rules:
- Contract constants (`OBS_DIM`, `N_BEAMS`, `STEER_MAX`, `V_NORM`, index slices, etc.) are defined **only** in `lapzero/contract.py`. Everything else imports them. No magic numbers.
- `scripts/` contain argument parsing and a call into `lapzero/`. No logic.
- The backend imports `lapzero.io` and `lapzero.contract` for schemas and validation; it never imports `lapzero.sim` or `lapzero.training`.

---

## 4. Tech stack (allowed list)

| Area | Allowed | Not allowed without approval |
|---|---|---|
| Language | Python 3.11+ | Python < 3.11 features or workarounds |
| Simulator | F1TENTH Gym, wrapped as a Gymnasium env | CARLA (stretch goal only, separate approval), other simulators |
| RL / BC | Stable-Baselines3 (PPO, SAC), `imitation` for BC, PyTorch | RLlib, custom RL algorithm implementations, JAX |
| Numerics | NumPy, pandas (offline analysis only) | |
| Tracking | MLflow (tracking + model registry) | Weights & Biases, Comet |
| Backend | FastAPI, Pydantic v2, SQLAlchemy 2.x, Alembic, Uvicorn | Django, Flask, raw SQL strings in routers |
| Database | PostgreSQL | SQLite outside tests, MongoDB |
| Frontend | React 18+, Vite, TypeScript (strict), Tailwind CSS, Recharts, HTML Canvas | Next.js, Redux (use React state + a small fetch hook), chart libraries other than Recharts |
| Testing | pytest, pytest-cov, hypothesis (property tests), Vitest + React Testing Library | unittest-style test classes |
| Quality | ruff (lint + format), mypy (strict), ESLint, Prettier | black/isort/flake8 (ruff covers them) |
| Infra | Docker, Docker Compose, GitHub Actions | Kubernetes, Terraform (out of scope for v1) |

LLMs and RAG are **not** part of this project. Do not add them.

**Compatibility note:** F1TENTH Gym has historically pinned older `gym`/NumPy versions. Before writing env code, confirm the installed version works on Python 3.11 with Gymnasium. If it does not, stop and report the conflict rather than patching around it.

---

## 5. Python standards

### Typing (mypy strict)

- `mypy --strict` must pass on `lapzero/`, `backend/app/`, and `scripts/`.
- Every function and method has full parameter and return annotations, including `-> None`.
- No `Any` unless unavoidable at a third-party boundary; when used, add `# type: ignore[<code>]` or a cast **with a one-line comment explaining why**.
- NumPy arrays use `numpy.typing.NDArray[np.float32]` (or the precise dtype). Observations are always `NDArray[np.float32]` of shape `(OBS_DIM,)`.
- Use `dataclass(frozen=True, slots=True)` for internal value objects and Pydantic models for anything crossing I/O (API, JSON files, config).
- Prefer `Literal` and `enum.StrEnum` for fixed string sets (e.g. `driver.type`, `result.status`).

### Style

- ruff format, line length 100.
- Modules are small and single purpose. A file over ~300 lines is a signal to ask about splitting.
- Pure functions for maths (lidar downsampling, action mapping, normalization). Side effects (sim stepping, file writes, DB) live at the edges.
- Docstrings on every public function: one line summary, then `Args`, `Returns`, `Raises` where relevant. State units in the docstring (`metres`, `rad`, `m/s`).
- Use `logging` (module-level `logger = logging.getLogger(__name__)`), never `print` outside `scripts/`.
- Seed everything that is random (`numpy.random.Generator`, env reset, SB3 `seed=`) and log the seed.

### Errors

- Validate inputs at boundaries and raise specific exceptions (`ValueError`, custom `ContractViolationError`) with a message naming the field and the bad value.
- Never swallow exceptions with bare `except:` or `except Exception: pass`.

---

## 6. Testing standards

- **Every module in `lapzero/` and `backend/app/` has a matching test file** in `tests/unit/` (e.g. `lapzero/sim/lidar.py` → `tests/unit/sim/test_lidar.py`). New module = new test file in the same change.
- Tests are fast and deterministic. Unit tests must not start the full simulator or a real database; use small fakes or fixtures. Anything slower goes in `tests/integration/` and is marked `@pytest.mark.integration`.
- Contract tests in `tests/contracts/` implement Section 8 of `state_space.md` and are **never** edited without the approval process in Rule 2 and Rule 3.
- Use `hypothesis` for the numeric mapping functions (lidar downsampling, steering and throttle mapping, normalization): outputs always finite and inside contract bounds for any input, including `NaN`, `inf`, and out-of-range values.
- Backend tests use FastAPI's `TestClient` with a PostgreSQL test database (Docker service in CI); each test runs in a rolled-back transaction.
- Coverage target: **90% lines** on `lapzero/` and `backend/app/`. Do not write tests that only chase coverage; every test asserts behaviour.
- Name tests by behaviour: `test_throttle_minus_one_never_drops_target_below_zero`, not `test_mapping_2`.

### When a test fails

1. Read the failure. Find the root cause in the implementation.
2. Fix the implementation.
3. If you believe the **test** is wrong, stop. Explain what the test expects, why you think it is wrong, and what you propose. Wait for approval before editing it.
4. Never delete, skip (`@pytest.mark.skip`, `xfail`), or loosen assertions or tolerances to get green.

---

## 7. Frontend standards

- TypeScript `strict: true`, no `any`, no `// @ts-ignore` without a comment explaining why.
- API response types are written once in `frontend/src/api/types.ts` and must match `schemas/*.schema.json` and the contract. If the backend shape changes, update the types in the same change.
- Components are function components with hooks. Data fetching goes through `frontend/src/api/client.ts`; components never call `fetch` directly.
- Replay rendering: canvas, interpolated on `t` at 60 fps, track geometry fetched once from `/api/tracks/{track_id}` (contract Section 6).
- Every component with logic has a Vitest test. `npm run lint`, `npm run typecheck`, and `npm test` must pass.

---

## 8. Commands and definition of done

Run from the repo root.

```bash
# Python
ruff check . && ruff format --check .
mypy --strict lapzero backend/app scripts
pytest -m "not integration" --cov=lapzero --cov=backend/app --cov-fail-under=90
pytest -m integration                         # when the change touches sim, DB, or gate

# Contract
python scripts/check_sim_params.py            # simulator values match the contract
pytest tests/contracts

# Frontend
cd frontend && npm run lint && npm run typecheck && npm test

# Services
docker compose up -d db mlflow
alembic -c backend/alembic.ini upgrade head
```

A task is **done** only when:

1. All commands relevant to the changed area pass (show a short summary of results).
2. New or changed behaviour has tests.
3. No files outside the task scope were modified (list the files you changed).
4. Contract, schemas, and frontend types still agree.
5. Any new config value is in `.env.example` and documented.

If something cannot pass (missing service, sim install issue), say exactly what failed and why. Do not report success.

---

## 9. Domain rules

### Simulation and contract

- The env wrapper (`lapzero/sim/env.py`) is the **only** place that converts between policy actions `[-1, 1]` and simulator commands (steering rad, target velocity). Nothing else calls the raw simulator step.
- The human keyboard/gamepad driver must emit actions in the same `[-1, 1]` space as models. Recorded human laps are training data and must pass the same validation.
- World pose (`x, y, theta`) must never enter the observation vector. It is for reward, logging, and replay only.

### Training

- Every training run logs to MLflow: all contract constants, `contract_version`, reward function version, hyperparameters, seed, track IDs, git commit hash.
- Models are saved through the MLflow registry with aliases `champion` and `challenger`. No model files committed to git.
- Reward function changes go in `lapzero/sim/reward.py` with a `REWARD_VERSION` constant bump; they are not contract changes.

### Release gate

- Gate logic lives in `lapzero/eval/gate.py` as a pure function: inputs are two lists of eval results, output is a typed decision with reasons. The CI workflow only calls it.
- Rules (from README): over 20 laps on 3 tracks, promote only if mean lap time is lower, crash rate is not higher, and at least one unseen track is fully completed. Changing these rules requires approval.
- The gate refuses to compare models with different `contract_version` major numbers.

### Backend

- Routers are thin: validate with Pydantic, call a service, return a response model.
- All DB access via SQLAlchemy 2.x sessions in `services/`; schema changes only through Alembic migrations (never `create_all` outside tests).
- Uploaded telemetry is validated against the JSON schema and contract major version before it is stored.

---

## 10. How to work on a prompt

1. Restate the task in one or two lines and list the files you expect to touch.
2. If the task needs files outside that list, a new dependency, or a contract change, ask first.
3. Write or update the tests for the behaviour, then the implementation.
4. Run the checks in Section 8 for the affected area.
5. Reply with: what changed, files modified, check results, and anything you noticed but did not touch.

Keep replies short. Do not paste whole files back unless asked.

---

## 11. Git

- Conventional commits: `feat(sim): ...`, `fix(backend): ...`, `test(contracts): ...`, `docs: ...`, `chore: ...`.
- One logical change per commit. Contract changes get their own commit referencing the new `contract_version`.
- Never commit `data/`, `mlruns/`, `.env`, model weights, or `node_modules/`.
- Do not push, open PRs, or rewrite history unless asked.