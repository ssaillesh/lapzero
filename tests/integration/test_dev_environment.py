"""Integration check of the development environment that scripts/ subprocesses run in.

Every script test starts a fresh interpreter from the venv. It only finds lapzero through the
editable install's .pth file, which Python skips if the file is hidden (macOS + iCloud Drive; see
scripts/setup_dev.sh).
"""

import subprocess
import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.integration


def test_a_fresh_interpreter_outside_the_repo_imports_lapzero(tmp_path: Path) -> None:
    completed = subprocess.run(
        [sys.executable, "-c", "import lapzero"],
        cwd=tmp_path,  # not the repo root, so lapzero cannot come from the working directory
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert completed.returncode == 0, (
        f"the venv cannot import lapzero; run scripts/setup_dev.sh\n{completed.stderr}"
    )
