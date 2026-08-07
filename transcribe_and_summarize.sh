#!/bin/bash

# Compatibility launcher for the original two-positional-argument interface.
set -u

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PYTHON_BIN="${AUTO_TRANSCRIPTION_PYTHON:-python3}"

if [ -z "${AUTO_TRANSCRIPTION_PYTHON:-}" ] && [ -x "$SCRIPT_DIR/.venv/bin/python" ]; then
    PYTHON_BIN="$SCRIPT_DIR/.venv/bin/python"
fi

# Start the main MLX workload at macOS's lowest CPU scheduling priority. Python
# also applies this itself for direct invocations and launchd runs.
run_python() {
    exec /usr/bin/nice -n 20 "$PYTHON_BIN" "$SCRIPT_DIR/auto_transcription.py" "$@"
}

if [ "$#" -ge 1 ] && [[ "$1" == -* ]]; then
    run_python "$@"
fi
if [ "$#" -ge 1 ]; then
    case "$1" in
        scan|import|process|run|status|retry|reprocess|cleanup|check|prepare|install-launchd)
            run_python "$@"
            ;;
    esac
fi

if [ "$#" -gt 2 ]; then
    printf 'Usage: %s [RECORDER_MOUNT] [OBSIDIAN_VAULT]\n' "$0" >&2
    exit 2
fi

case "$#" in
    0)
        run_python run
        ;;
    1)
        run_python --mount "$1" run
        ;;
    2)
        run_python --mount "$1" --vault "$2" run
        ;;
esac
