# LapZero dashboard

The browser front end of LapZero. It has three views:

- **Leaderboard** (`/leaderboard`): each driver's fastest completed lap per track.
- **Track Explorer** (`/tracks/<track_id>/replays/<lap_id>`): a canvas replay of a lap on its track, with a synchronized readout and speed and steering charts.
- **Governance** (`/governance`): the MLflow champion and the release-gate evidence recorded on it.

Stack (CLAUDE.md section 4): React 19, Vite, TypeScript (strict), Tailwind CSS v4, Recharts, HTML Canvas, Vitest with React Testing Library, ESLint and Prettier. It uses no router, state, icon or validation library. A small History API router (`src/lib/router.ts`) and hand-written response decoders (`src/api/decode.ts`) replace them.

## Prerequisites

- Node.js 22.12 or newer (`node --version`), npm 10.
- The LapZero API running (see below). The dashboard shows an "API unreachable" state without it; it never falls back to sample data.

## Run it

Start the backend from the repo root, as the root README describes:

```bash
cp .env.example .env                       # once; set POSTGRES_PASSWORD and DATABASE_URL
docker compose up -d db
uv run alembic -c backend/alembic.ini upgrade head
uv run uvicorn backend.app.main:app --port 8000
```

Then start the dashboard:

```bash
cd frontend
npm install
npm run dev                                # http://localhost:5173
```

Port 5173 is in the backend's default `CORS_ORIGINS`. If you serve the dashboard from another origin, add it to `CORS_ORIGINS` in `.env`.

Track geometry comes from `TRACKS_DIR/<track_id>/centreline.csv`. `TRACKS_DIR` defaults to `<repo>/tracks`, and this repo keeps its track files in `data/tracks`. To see track outlines, start the API with `TRACKS_DIR=data/tracks`. Without it, the replay still draws the driven line and says the centreline is missing.

## Configuration

| Variable            | Default                     | Meaning                                                                                                                                                                                 |
| ------------------- | --------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `VITE_API_BASE_URL` | `http://localhost:8000/api` | API root, including the `/api` prefix every router uses. A bare origin such as `http://host:8000` gets `/api` appended. A path such as `/api` targets the same origin (behind a proxy). |

Copy `frontend/.env.example` to `frontend/.env.local` to override it. Vite reads it at build time.

## Checks

```bash
npm run lint          # ESLint (typescript-eslint strict, React hooks), zero warnings allowed
npm run typecheck     # tsc -b, strict with noUncheckedIndexedAccess and exactOptionalPropertyTypes
npm test              # Vitest + React Testing Library
npm run build         # type check, then a production build into dist/
npm run format:check  # Prettier
```

`npm run preview` serves `dist/` on port 4173. A production web server must answer unknown paths with `index.html`, so routes such as `/tracks/spielberg` survive a refresh.

## API endpoints used

All requests go through `src/api/client.ts`. ESLint rejects `fetch` anywhere else. Types live in `src/api/types.ts`, and every response is decoded at runtime before a component sees it.

| Endpoint                                              | Used for                                                                                                    |
| ----------------------------------------------------- | ----------------------------------------------------------------------------------------------------------- |
| `GET /api/health`                                     | Connection indicator and contract version. Polled every 30 s, paused while the tab is hidden.               |
| `GET /api/tracks`                                     | Track lists                                                                                                 |
| `GET /api/tracks/{track_id}`                          | Centreline and edges for the canvas (fetched once per session)                                              |
| `GET /api/leaderboard/{track_id}?driver_type=&limit=` | Leaderboard, lap picker, the champion's standing per track                                                  |
| `GET /api/replays/{lap_id}`                           | 10 Hz replay (contract Section 6)                                                                           |
| `GET /api/models/champion`                            | Governance view and champion highlighting. A 404 means no champion; a 503 means the registry is unreadable. |

### What the API does not provide, and how the UI says so

- **Lateral velocity, lateral acceleration and friction μ.** The replay carries only `t, x, y, theta, v, steer`, so the charts plot speed and steering and state that the rest is not in the contract.
- **A list of every lap.** Replays are picked from the leaderboard (each driver's best completed lap). Any other lap, including one that did not finish, opens by URL: `/tracks/<track_id>/replays/<lap_id>`.
- **Challenger alias, version history, training parameters and metrics, sweep reports.** The governance view shows the champion and its version tags, and names what it cannot show.
- **Champion matching.** A leaderboard entry carries no MLflow run ID, so a lap is the champion's when it is a model lap whose `model_version` equals the champion's registry version.

## Layout

```
src/
  api/          types.ts (response types), decode.ts (runtime checks), client.ts (HTTP), errors.ts
  lib/          pure logic: replay sampling, world-to-canvas transform, playback clock,
                canvas renderer, health monitor, router, formatters, leaderboard sorting
  hooks/        useResource (cached, cancellable loading), useApi, playback and size hooks
  components/   layout/, common/, leaderboard/, replay/, governance/
  views/        LeaderboardView, TrackViewerView, ChampionView
  test/         setup (jsdom shims), fixtures shaped like API responses
```

## Replay engine

- **Clock** (`src/lib/playback.ts`). Replay time comes from the wall clock (anchor plus elapsed time × rate), so a dropped frame never slows playback. One `requestAnimationFrame` loop runs while playing. Scrubbing pauses the clock until the pointer is released.
- **Sampling** (`src/lib/replay.ts`). Samples are sanitized once, dropping non-finite values and times that do not increase. Each frame then uses a binary search, with the previous index as a hint, and interpolates linearly on `t`. Heading interpolates the short way across ±π.
- **Transform** (`src/lib/transform.ts`). World metres (y up, θ counter-clockwise) map to CSS pixels through one proportional scale, with y flipped, so heading θ is drawn at canvas angle −θ.
- **Canvas** (`src/lib/trackRenderer.ts`). The grid, track surface, edges, centreline and full driven line are drawn once per resize into an offscreen layer. Each frame copies that layer and draws the driven line so far, coloured by speed, plus the car. The backing store is scaled by `devicePixelRatio`.
- **No 60 Hz React renders.** The canvas, scrubber, readout and chart playheads subscribe to the clock and write to the DOM directly. React re-renders only when play state or speed changes.
- **Keyboard.** Space or K plays and pauses, ← and → seek 1 s (Shift for 5 s), Home restarts.
