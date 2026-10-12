"""Train the behaviour cloning policy on recorded human laps and track the run in MLflow.

Usage (from the repo root):
    uv run python scripts/train_bc.py --seed 0 --epochs 50
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from lapzero.contract import LAPS_DIR
from lapzero.training.bc import BCConfig, run_bc_training
from lapzero.training.tracking import mlflow_run

_DEFAULTS = BCConfig()


def main(argv: list[str] | None = None) -> None:
    """Parse arguments, open an MLflow run and train."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--laps-dir", type=Path, default=Path(LAPS_DIR))
    parser.add_argument("--output", type=Path, default=Path("models/bc_v1.pt"))
    parser.add_argument("--track", action="append", dest="tracks", help="repeat to pick tracks")
    parser.add_argument("--epochs", type=int, default=_DEFAULTS.epochs)
    parser.add_argument("--batch-size", type=int, default=_DEFAULTS.batch_size)
    parser.add_argument("--learning-rate", type=float, default=_DEFAULTS.learning_rate)
    parser.add_argument("--hidden-size", type=int, default=_DEFAULTS.hidden_size)
    parser.add_argument("--val-fraction", type=float, default=_DEFAULTS.val_fraction)
    parser.add_argument("--seed", type=int, default=_DEFAULTS.seed)
    parser.add_argument("--tracking-uri", default=Path("mlruns").resolve().as_uri())
    parser.add_argument("--experiment", default="lapzero-bc")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    config = BCConfig(
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        hidden_size=args.hidden_size,
        val_fraction=args.val_fraction,
        seed=args.seed,
    )
    with mlflow_run(
        tracking_uri=args.tracking_uri, experiment=args.experiment, run_name=args.output.stem
    ) as tracker:
        run_bc_training(args.laps_dir, args.output, config, tracker, track_ids=args.tracks)


if __name__ == "__main__":
    main()
