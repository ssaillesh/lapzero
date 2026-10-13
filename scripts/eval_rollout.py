"""Drive the registry champion for one lap on a track and upload the lap to the LapZero API.

The track's files are data/tracks/<track>/<track>.yaml and centreline.csv; the champion is
lapzero-policy@champion in the repo-root MLflow store. Exits 0 once the API stored the lap,
1 if the rollout or the upload failed.

Usage (from the repo root, with the API running):
    uv run python scripts/eval_rollout.py --track spielberg
"""

from __future__ import annotations

import argparse
import logging

from mlflow.exceptions import MlflowException

from lapzero.eval.rollout import DEFAULT_API_URL, LapUploadError, evaluate_champion_on_track

logger = logging.getLogger(__name__)


def main(argv: list[str] | None = None) -> int:
    """Parse arguments, drive and upload the champion's lap, and return the exit code."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--track", required=True, help="track ID, e.g. spielberg")
    parser.add_argument("--api", default=DEFAULT_API_URL, help="API base URL; POSTs to <api>/laps")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    try:
        lap_id = evaluate_champion_on_track(track_id=args.track, api_url=args.api)
    except (LapUploadError, MlflowException, OSError, ValueError) as error:
        logger.error("champion rollout on %s failed: %s", args.track, error)
        return 1
    logger.info("champion lap %s on %s uploaded to %s", lap_id, args.track, args.api)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
