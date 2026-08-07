#!/bin/bash

set -u
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR" || exit 1
if [ "${1:-}" = "--live" ]; then
    exec "$SCRIPT_DIR/test_live_transcription.sh"
fi
PYTHON_BIN="python3"
if [ -x "$SCRIPT_DIR/.venv/bin/python" ]; then
    PYTHON_BIN="$SCRIPT_DIR/.venv/bin/python"
fi
exec "$PYTHON_BIN" -m unittest discover -s tests -v
