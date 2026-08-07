#!/bin/bash

set -u
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR" || exit 1

python3 -m venv .venv || exit 1
exec .venv/bin/python -m pip install -r requirements-apple-silicon.txt
