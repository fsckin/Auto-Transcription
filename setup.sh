#!/bin/bash

# Complete, repeatable macOS installer for Auto-Transcription.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
VENV_DIR="$SCRIPT_DIR/.venv"
VENV_PYTHON="$VENV_DIR/bin/python"
LOCK_FILE="$SCRIPT_DIR/requirements-lock.txt"
APP="$SCRIPT_DIR/auto_transcription.py"
DEFAULT_TEMPLATE="$SCRIPT_DIR/default-note-template.txt"
HOMEBREW_INSTALL_URL="https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh"
DRY_RUN=0
INSTALL_AGENT=1
UPGRADE=0
UNINSTALL=0
MIN_FREE_GB="${AUTO_TRANSCRIPTION_MIN_FREE_GB:-10}"
CONFIG_PATH="${AUTO_TRANSCRIPTION_CONFIG:-}"
BREW_BIN="${AUTO_TRANSCRIPTION_BREW_BIN:-}"
PYTHON_BIN="${AUTO_TRANSCRIPTION_PYTHON_BIN:-}"

usage() {
    printf 'Usage: %s [--config FILE] [--no-launch-agent] [--upgrade] [--uninstall] [--dry-run]\n' "$0"
    printf '\nInstalls Homebrew when needed, Python, FFmpeg, Ollama, Python packages,\n'
    printf 'Whisper and Ollama models, validates the system, and installs the LaunchAgent.\n'
}

log() {
    printf '\n==> %s\n' "$*"
}

die() {
    printf 'setup: %s\n' "$*" >&2
    exit 1
}

print_command() {
    printf '    '
    printf '%q ' "$@"
    printf '\n'
}

run() {
    if [ "$DRY_RUN" -eq 1 ]; then
        print_command "$@"
        return 0
    fi
    "$@"
}

while [ "$#" -gt 0 ]; do
    case "$1" in
        --config)
            [ "$#" -ge 2 ] || die "--config requires a file path"
            CONFIG_PATH="$2"
            shift 2
            ;;
        --no-launch-agent)
            INSTALL_AGENT=0
            shift
            ;;
        --dry-run)
            DRY_RUN=1
            shift
            ;;
        --upgrade)
            UPGRADE=1
            shift
            ;;
        --uninstall)
            UNINSTALL=1
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            die "unknown option: $1"
            ;;
    esac
done

[ "$(uname -s)" = "Darwin" ] || die "this installer supports macOS only"
[ "$(uname -m)" = "arm64" ] || die "mlx-whisper requires an Apple Silicon Mac"
[ -f "$APP" ] || die "application script is missing: $APP"
[ -f "$DEFAULT_TEMPLATE" ] || die "default note template is missing: $DEFAULT_TEMPLATE"
if [ -n "$CONFIG_PATH" ]; then
    [ -f "$CONFIG_PATH" ] || die "configuration file is missing: $CONFIG_PATH"
    CONFIG_PATH="$(cd "$(dirname "$CONFIG_PATH")" && pwd)/$(basename "$CONFIG_PATH")"
fi

if [ "$UNINSTALL" -eq 1 ]; then
    if [ -x "$VENV_PYTHON" ]; then
        PYTHON_BIN="$VENV_PYTHON"
    elif [ -z "$PYTHON_BIN" ] && command -v python3 >/dev/null 2>&1; then
        PYTHON_BIN="$(command -v python3)"
    fi
    [ -n "$PYTHON_BIN" ] && [ -x "$PYTHON_BIN" ] || die "Python is required to uninstall the LaunchAgent"
    log "Uninstalling automatic recorder processing; recordings and state are preserved"
    if [ -n "$CONFIG_PATH" ]; then
        run "$PYTHON_BIN" "$APP" --config "$CONFIG_PATH" --uninstall
    else
        run "$PYTHON_BIN" "$APP" --uninstall
    fi
    exit 0
fi

[ -f "$LOCK_FILE" ] || die "hashed dependency lock is missing: $LOCK_FILE"
case "$MIN_FREE_GB" in
    ''|*[!0-9]*) die "AUTO_TRANSCRIPTION_MIN_FREE_GB must be a whole number" ;;
esac
available_kb="$(df -Pk "$HOME" | awk 'NR==2 {print $4}')"
required_kb=$((MIN_FREE_GB * 1024 * 1024))
[ "$available_kb" -ge "$required_kb" ] || die "at least ${MIN_FREE_GB} GB of free disk space is required"
log "Disk-space preflight passed: $((available_kb / 1024 / 1024)) GB free"

find_brew() {
    if [ -n "$BREW_BIN" ] && [ -x "$BREW_BIN" ]; then
        return 0
    fi
    BREW_BIN=""
    if command -v brew >/dev/null 2>&1; then
        BREW_BIN="$(command -v brew)"
    elif [ -x /opt/homebrew/bin/brew ]; then
        BREW_BIN=/opt/homebrew/bin/brew
    elif [ -x /usr/local/bin/brew ]; then
        BREW_BIN=/usr/local/bin/brew
    fi
    [ -n "$BREW_BIN" ]
}

install_homebrew() {
    if find_brew; then
        log "Homebrew already installed: $BREW_BIN"
        return
    fi
    command -v curl >/dev/null 2>&1 || die "curl is required to install Homebrew"
    log "Installing Homebrew"
    if [ "$DRY_RUN" -eq 1 ]; then
        printf '    NONINTERACTIVE=1 /bin/bash -c "$(curl -fsSL %s)"\n' "$HOMEBREW_INSTALL_URL"
        BREW_BIN=/opt/homebrew/bin/brew
        return
    fi
    NONINTERACTIVE=1 /bin/bash -c "$(curl -fsSL "$HOMEBREW_INSTALL_URL")"
    find_brew || die "Homebrew installation completed but brew could not be found"
}

install_homebrew
if [ "$DRY_RUN" -eq 1 ] && [ ! -x "$BREW_BIN" ]; then
    BREW_PREFIX=/opt/homebrew
else
    BREW_PREFIX="$("$BREW_BIN" --prefix)"
fi
export PATH="$BREW_PREFIX/bin:$BREW_PREFIX/sbin:/Applications/Ollama.app/Contents/Resources:/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin"

ensure_formula() {
    formula="$1"
    shift
    all_available=1
    for executable in "$@"; do
        if ! command -v "$executable" >/dev/null 2>&1; then
            all_available=0
        fi
    done
    if [ "$all_available" -eq 1 ]; then
        log "$formula already available"
        return
    fi
    if [ "$DRY_RUN" -eq 0 ] && "$BREW_BIN" list --formula --versions "$formula" >/dev/null 2>&1; then
        log "$formula is already installed"
        return
    fi
    log "Installing $formula with Homebrew"
    run "$BREW_BIN" install "$formula"
    hash -r
}

python_is_compatible() {
    "$1" -c 'import sys; raise SystemExit(0 if sys.version_info >= (3, 11) else 1)' >/dev/null 2>&1
}

if [ -z "$PYTHON_BIN" ]; then
    if command -v python3 >/dev/null 2>&1 && python_is_compatible "$(command -v python3)"; then
        PYTHON_BIN="$(command -v python3)"
        log "Compatible Python already available: $PYTHON_BIN"
    else
        ensure_formula python python3
        PYTHON_BIN="$BREW_PREFIX/bin/python3"
    fi
fi
if [ "$DRY_RUN" -eq 0 ]; then
    [ -x "$PYTHON_BIN" ] || die "Python was installed but is not executable: $PYTHON_BIN"
    python_is_compatible "$PYTHON_BIN" || die "Python 3.11 or newer is required"
fi

if ! command -v ffmpeg >/dev/null 2>&1 || ! command -v ffprobe >/dev/null 2>&1; then
    ensure_formula ffmpeg ffmpeg
else
    log "FFmpeg and FFprobe already available"
fi
ensure_formula ollama ollama

if [ "$UPGRADE" -eq 1 ]; then
    log "Upgrading installed Homebrew dependencies"
    run "$BREW_BIN" update
    for formula in python ffmpeg ollama; do
        if [ "$DRY_RUN" -eq 1 ] || "$BREW_BIN" list --formula --versions "$formula" >/dev/null 2>&1; then
            run "$BREW_BIN" upgrade "$formula"
        fi
    done
fi

log "Creating or refreshing the Python environment"
VENV_BACKUP=""
rollback_environment() {
    code=$?
    trap - ERR
    if [ -n "$VENV_BACKUP" ] && [ -d "$VENV_BACKUP" ]; then
        failed="$SCRIPT_DIR/.venv.failed-$(date +%Y%m%d-%H%M%S)"
        [ ! -e "$VENV_DIR" ] || mv "$VENV_DIR" "$failed"
        mv "$VENV_BACKUP" "$VENV_DIR"
        printf 'setup: validation failed; restored the previous environment (failed candidate: %s)\n' "$failed" >&2
    fi
    exit "$code"
}
if [ "$DRY_RUN" -eq 0 ] && [ -d "$VENV_DIR" ] && { [ ! -x "$VENV_PYTHON" ] || ! python_is_compatible "$VENV_PYTHON"; }; then
    VENV_BACKUP="$SCRIPT_DIR/.venv.backup-$(date +%Y%m%d-%H%M%S)"
    log "Existing virtual environment is unusable; preserving it at $VENV_BACKUP"
    mv "$VENV_DIR" "$VENV_BACKUP"
elif [ "$DRY_RUN" -eq 0 ] && [ -d "$VENV_DIR" ]; then
    VENV_BACKUP="$SCRIPT_DIR/.venv.backup-$(date +%Y%m%d-%H%M%S)"
    log "Creating an APFS clone of the working environment for rollback: $VENV_BACKUP"
    if ! /bin/cp -cR "$VENV_DIR" "$VENV_BACKUP"; then
        log "APFS cloning is unavailable; creating a regular rollback copy"
        /bin/cp -R "$VENV_DIR" "$VENV_BACKUP"
    fi
fi
if [ "$DRY_RUN" -eq 0 ]; then
    trap rollback_environment ERR
fi
if [ ! -x "$VENV_PYTHON" ]; then
    run "$PYTHON_BIN" -m venv "$VENV_DIR"
else
    log "Reusing existing virtual environment: $VENV_DIR"
fi
run "$VENV_PYTHON" -m pip install --disable-pip-version-check --upgrade pip
run "$VENV_PYTHON" -m pip install --disable-pip-version-check --require-hashes -r "$LOCK_FILE"

run_app() {
    if [ -n "$CONFIG_PATH" ]; then
        run "$VENV_PYTHON" "$APP" --config "$CONFIG_PATH" "$@"
    else
        run "$VENV_PYTHON" "$APP" "$@"
    fi
}

log "Starting Ollama and downloading configured Whisper and summary models"
run_app --no-notify prepare

log "Validating the complete installation"
run_app --no-notify check

if [ "$INSTALL_AGENT" -eq 1 ]; then
    log "Installing and activating automatic recorder processing"
    run_app --install
fi

trap - ERR

log "Auto-Transcription installation complete"
if [ -n "$VENV_BACKUP" ]; then
    printf 'Rollback environment retained at: %s\n' "$VENV_BACKUP"
fi
printf 'Run a manual validation any time with:\n'
if [ -n "$CONFIG_PATH" ]; then
    print_command "$VENV_PYTHON" "$APP" --config "$CONFIG_PATH" --no-notify check
else
    print_command "$VENV_PYTHON" "$APP" --no-notify check
fi
