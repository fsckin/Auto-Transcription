#!/bin/bash

set -u
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PYTHON_BIN="$SCRIPT_DIR/.venv/bin/python"

if [ ! -x "$PYTHON_BIN" ]; then
    printf 'Missing project environment. Run ./setup.sh first.\n' >&2
    exit 1
fi
if ! command -v say >/dev/null 2>&1 || ! command -v ffmpeg >/dev/null 2>&1; then
    printf 'The live test requires macOS say and FFmpeg.\n' >&2
    exit 1
fi

TEST_ROOT="$(mktemp -d /tmp/auto-transcription-live.XXXXXX)" || exit 1
cleanup() {
    rm -rf "$TEST_ROOT"
}
trap cleanup EXIT INT TERM

mkdir -p "$TEST_ROOT/recorder/REC_FILE" "$TEST_ROOT/vault" || exit 1
say -o "$TEST_ROOT/spoken.aiff" "This is a local transcription pipeline test." || exit 1
ffmpeg -hide_banner -loglevel error -i "$TEST_ROOT/spoken.aiff" "$TEST_ROOT/recorder/REC_FILE/LIVE_TEST.mp3" || exit 1

"$PYTHON_BIN" "$SCRIPT_DIR/auto_transcription.py" \
    --mount "$TEST_ROOT/recorder" \
    --vault "$TEST_ROOT/vault" \
    --state-root "$TEST_ROOT/state" \
    --backend mlx \
    --model tiny \
    --no-summary \
    --no-vad \
    --no-notify \
    run || exit 1

TRANSCRIPT="$(find "$TEST_ROOT/state/Transcripts" -name '*.raw.txt' -type f -print -quit)"
if [ -z "$TRANSCRIPT" ] || ! grep -Eiq 'local.*transcription.*pipeline.*test' "$TRANSCRIPT"; then
    printf 'Live transcript did not contain the expected synthetic phrase.\n' >&2
    exit 1
fi

printf 'Live MLX transcription test passed.\n'
