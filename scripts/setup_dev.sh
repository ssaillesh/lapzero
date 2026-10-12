#!/usr/bin/env bash
# Create or repair the development environment: the uv venv with every dependency group, then
# check that a fresh interpreter can import the editable lapzero install.
#
# macOS + iCloud Drive: when ~/Documents is synced, iCloud sets the hidden flag (UF_HIDDEN) on
# every file inside a dot-directory such as .venv within seconds, and Python 3.11.14+ skips
# hidden .pth files. The editable install then vanishes and scripts run as subprocesses fail with
# "No module named 'lapzero'". Clearing the flag does not stick (iCloud re-hides it), so on macOS
# the venv lives in .venv.nosync, which iCloud ignores, and .venv is a symlink to it.
#
# Usage, from anywhere: scripts/setup_dev.sh
set -euo pipefail
cd "$(dirname "$0")/.."

if [[ "$(uname -s)" == "Darwin" ]]; then
    if [[ -d .venv && ! -L .venv ]]; then
        if [[ -e .venv.nosync ]]; then
            echo "both .venv and .venv.nosync exist; delete one of them and rerun" >&2
            exit 1
        fi
        echo "moving .venv to .venv.nosync (iCloud does not touch *.nosync)"
        mv .venv .venv.nosync
    fi
    ln -sfn .venv.nosync .venv
    export UV_PROJECT_ENVIRONMENT=.venv.nosync  # create and sync the real directory
fi

uv sync --all-groups

if [[ "$(uname -s)" == "Darwin" ]]; then
    chflags -R nohidden .venv.nosync  # flags iCloud set before the venv was moved
fi

# Run from / so lapzero can only come from the venv's .pth file, not the working directory.
python="$PWD/.venv/bin/python"
if ! (cd / && "$python" -c "import lapzero") 2>/dev/null; then
    echo "setup failed: $python cannot import lapzero; check for hidden .pth files with" >&2
    echo "  ls -lO .venv/lib/python*/site-packages/*.pth" >&2
    exit 1
fi
echo "dev environment ready: $python imports lapzero"
