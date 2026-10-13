"""Release gate: decide whether the challenger replaces the champion, and optionally promote it.

Reads the champion's and the challenger's friction sweeps (directories written by
scripts/eval_sweep.py), applies the robustness-aware promotion rule (lapzero.eval.gate) and prints
the verdict. With --promote and a PROMOTE verdict, the challenger becomes the champion in the MLflow
model registry. Promotion is refused if either alias no longer points at the version its sweep
evaluated (e.g. training registered a new challenger since), or if a sweep ran a local checkpoint.

Exits 0 on PROMOTE and 1 on REJECT.

Usage (from the repo root):
    .venv/bin/python scripts/eval_gate.py \
        --champion-sweep data/eval/friction_sweep/oschersleben/<champion sweep_id> \
        --challenger-sweep data/eval/friction_sweep/oschersleben/<challenger sweep_id> --promote
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from lapzero.eval.gate import decide
from lapzero.eval.sweep import SweepReport, read_sweep
from lapzero.registry import get_tracking_uri, promote_from_sweeps
from lapzero.training.tracking import REGISTERED_MODEL, ModelAlias


def _describe(role: str, sweep: SweepReport) -> str:
    """One summary line naming the policy, its registry version and the sweep."""
    version = sweep.run.policy_version or "-"
    return f"  {role:<10}  {sweep.run.policy}, version {version}, sweep {sweep.run.sweep_id}"


def main(argv: list[str] | None = None) -> int:
    """Parse arguments, run the gate on two sweeps, print the verdict and promote if asked.

    Returns:
        Exit status: 0 if the gate promotes the challenger, 1 if it rejects it.
    """
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--champion-sweep", type=Path, required=True, help="champion's sweep directory"
    )
    parser.add_argument(
        "--challenger-sweep", type=Path, required=True, help="challenger's sweep directory"
    )
    parser.add_argument(
        "--promote",
        action="store_true",
        help="make the challenger the champion in the registry if the gate passes",
    )
    parser.add_argument("--registered-model", default=REGISTERED_MODEL)
    parser.add_argument("--tracking-uri", default=get_tracking_uri())
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    champion = read_sweep(args.champion_sweep)
    challenger = read_sweep(args.challenger_sweep)
    decision = decide(champion, challenger)

    run = challenger.run
    print(f"\nRelease gate on {run.track_id}, contract {run.contract_version}")
    print(_describe("champion", champion))
    print(_describe("challenger", challenger))
    for reason in decision.reasons:
        print(f"  {reason}")
    print(f"verdict: {'PROMOTE' if decision.promote else 'REJECT'}")
    if decision.promote and args.promote:
        aliases = promote_from_sweeps(
            champion, challenger, tracking_uri=args.tracking_uri, model_name=args.registered_model
        )
        print(
            f"promoted {args.registered_model} version {aliases[ModelAlias.CHAMPION]} to "
            f"champion, replacing version {champion.run.policy_version}"
        )
    elif decision.promote:
        print("dry run: pass --promote to make the challenger the champion")
    return 0 if decision.promote else 1


if __name__ == "__main__":
    raise SystemExit(main())
