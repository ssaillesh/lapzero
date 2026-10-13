#!/usr/bin/env bash
# Create or repair the development environment: the uv venv with every dependency group, then
# check that a fresh interpreter can import the editable lapzero install.
#
# Keep the repo out of iCloud Drive (~/Documents and ~/Desktop when "Desktop & Documents" sync is
# on). iCloud offloads files it has uploaded when the disk fills: they stay listed at full size
# but read back as 0 bytes, so Python silently imports empty modules and git loses its objects.
# It also hides every file under .venv, and Python 3.11.14+ skips hidden .pth files.
#
# Usage, from anywhere: scripts/setup_dev.sh
set -euo pipefail
cd "$(dirname "$0")/.."

if [[ "$(uname -s)" == "Darwin" ]]; then
    case "$PWD" in
        "$HOME/Documents"/* | "$HOME/Desktop"/* | "$HOME/Library/Mobile Documents"/*)
            echo "warning: $PWD may be synced by iCloud Drive, which can offload repo files;" >&2
            echo "  clone the repo somewhere unsynced, e.g. ~/Developer/lapzero" >&2
            ;;
    esac
    # "|| true": head exits early on a long list, and pipefail would abort before the message.
    offloaded=$(find . -path ./.venv -prune -o -type f -flags +dataless -print | head -5 || true)
    if [[ -n "$offloaded" ]]; then
        echo "setup failed: iCloud has offloaded repo files (they read as empty), e.g.:" >&2
        echo "$offloaded" >&2
        exit 1
    fi
fi

uv sync --all-groups

# Run from / so lapzero can only come from the venv's .pth file, not the working directory.
python="$PWD/.venv/bin/python"
if ! (cd / && "$python" -c "import lapzero") 2>/dev/null; then
    echo "setup failed: $python cannot import lapzero; check for hidden .pth files with" >&2
    echo "  ls -lO .venv/lib/python*/site-packages/*.pth" >&2
    exit 1
fi
echo "dev environment ready: $python imports lapzero"
