#!/bin/bash

# Compatibility launcher for the original two-positional-argument interface.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PYTHON_BIN="${AUTO_TRANSCRIPTION_PYTHON:-python3}"

if [ -z "${AUTO_TRANSCRIPTION_PYTHON:-}" ] && [ -x "$SCRIPT_DIR/.venv/bin/python" ]; then
    PYTHON_BIN="$SCRIPT_DIR/.venv/bin/python"
fi

case "${1:-}" in
    "") set -- run ;;
    -*|scan|import|process|run|status|retry|reprocess|cleanup|check|doctor|prepare|maintenance|database|runs|install-launchd) ;;
    *)
        [ "$#" -le 2 ] || { printf 'Usage: %s [RECORDER_MOUNT] [OBSIDIAN_VAULT]\n' "$0" >&2; exit 2; }
        mount_point="$1"
        if [ "$#" -eq 2 ]; then
            set -- --mount "$mount_point" --vault "$2" run
        else
            set -- --mount "$mount_point" run
        fi
        ;;
esac

exec "$PYTHON_BIN" "$SCRIPT_DIR/auto_transcription.py" "$@"
