"""Check the installed F1TENTH Gym against the contract's simulator baseline table.

Starts the simulator on a blank map (lapzero.sim.f1tenth.installed_baseline), prints the installed
and contract value of every row of the baseline table (docs/contracts/state_space.md Section 0)
and of N_BEAMS, which the LiDAR downsampling derives from them.

Exits 0 if every value matches and 1 if any differs. A difference means the contract table must be
updated and contract_version bumped (Section 7), not that the simulator should be patched.

Usage (from the repo root):
    .venv/bin/python scripts/check_sim_params.py
"""

from __future__ import annotations

import argparse
import logging

from lapzero.contract import CONTRACT_VERSION
from lapzero.sim.f1tenth import installed_baseline


def main(argv: list[str] | None = None) -> int:
    """Start the simulator, print each baseline value next to the contract's, report mismatches.

    Returns:
        Exit status: 0 if every value matches the contract, 1 otherwise.
    """
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    values = installed_baseline()

    print(f"\nF1TENTH Gym simulator baseline vs contract {CONTRACT_VERSION}")
    print(f"  {'value':<32}{'installed':>12}{'contract':>12}")
    for value in values:
        verdict = "ok" if value.matches else "MISMATCH"
        print(f"  {value.name:<32}{value.installed:>12g}{value.contract:>12g}  {verdict}")
    mismatches = [value.name for value in values if not value.matches]
    if mismatches:
        print(
            f"{len(mismatches)} value(s) differ: {', '.join(mismatches)}. Update the baseline "
            "table in docs/contracts/state_space.md and lapzero/contract.py, and bump "
            "contract_version (Section 7)."
        )
        return 1
    print(f"all {len(values)} values match")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
