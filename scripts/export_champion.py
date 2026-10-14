"""Export the registry's champion to a JSON snapshot that the API can serve without MLflow.

A hosted API has no access to the local mlruns/ registry. Point its CHAMPION_SNAPSHOT setting at
the file this writes, and GET /api/models/champion (the dashboard's Governance page and Champion
badges) serves it. Re-run after every promotion.

Usage (from the repo root):
    uv run python scripts/export_champion.py
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from lapzero.registry import export_champion, get_tracking_uri
from lapzero.training.tracking import REGISTERED_MODEL


def main(argv: list[str] | None = None) -> int:
    """Parse arguments, write the snapshot and print what it holds; return the exit code."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("backend/champion_snapshot.json"),
        help="snapshot file to write (default: backend/champion_snapshot.json)",
    )
    parser.add_argument("--tracking-uri", default=get_tracking_uri())
    parser.add_argument("--registered-model", default=REGISTERED_MODEL)
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    snapshot = export_champion(
        args.out, tracking_uri=args.tracking_uri, model_name=args.registered_model
    )
    print(
        f"{args.out}: {snapshot.model_name} version {snapshot.version}, "
        f"run {snapshot.run_id or '-'}, gate {snapshot.tags.get('gate_result', '-')}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
