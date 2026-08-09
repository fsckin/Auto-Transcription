# Auto Transcription

A local, resumable pipeline that imports recordings from a Sony ICD-UX570, transcribes them on Apple Silicon, summarizes them with Ollama, and creates Obsidian notes.

Audio and transcripts stay on the Mac. The source recorder is never modified or cleaned automatically.

## Quick start

The Apple Silicon environment is installed in `.venv`. To recreate it:

```sh
./setup.sh
```

Validate the local tools and configured Ollama model:

```sh
./transcribe_and_summarize.sh --help
.venv/bin/python auto_transcription.py check
```

Run the original-compatible workflow with the default recorder and vault paths:

```sh
./transcribe_and_summarize.sh
```

The compatibility launcher also accepts the original two positional arguments:

```sh
./transcribe_and_summarize.sh "/Volumes/IC RECORDER" "$HOME/Obsidian/Main"
```

When no concrete vault is configured, Auto-Transcription reads Obsidian's local
vault registry and selects the single vault marked open. If only one valid vault
is registered, it uses that vault. Discovery validates both the vault directory
and its `.obsidian` marker and stops with an actionable error instead of guessing
when multiple vaults are ambiguous. An explicit `--vault` or `[paths].vault`
always takes precedence. This supports iCloud vaults at their actual macOS path,
including `~/Library/Mobile Documents/iCloud~md~obsidian/Documents/...`.

For complete CLI control, use `auto_transcription.py` directly. Global options come before the command:

```sh
.venv/bin/python auto_transcription.py --mount "/Volumes/IC RECORDER" --vault "$HOME/Obsidian/Main" run
```

## Commands

- `scan`: show files that are new or need an import retry.
- `import --dry-run`: preview local staging without changing state.
- `import`: copy new recordings into the local inbox.
- `process`: process the local queue without requiring the recorder.
- `run`: import, then process.
- `status`: show counts, recent files, failures, and hash prefixes.
- `retry`: resume failed recordings from their last durable stage.
- `reprocess --hash PREFIX`: deliberately rerun one recording without overwriting its old note.
- `cleanup --older-than-days N --dry-run`: preview archive cleanup.
- `cleanup --older-than-days N`: remove old archived audio while retaining notes and transcripts.
- `check`: validate FFmpeg, the transcription backend, Ollama, and configured paths.
- `doctor`: check dependencies, disk space, database integrity, recorder identity, the LaunchAgent, unresolved recordings, and safe-cleanup candidates.
- `prepare`: download and warm the configured transcription and Ollama models.
- `maintenance --dry-run`: preview removal of reproducible chunks, stale partial copies, confirmed macOS metadata, and old orphaned generated files. Add `--prune-model-cache` to explicitly include unused MLX Whisper model repositories.
- `database check|backup|repair`: inspect or maintain SQLite state; `database export --output FILE` creates a readable SQL export.
- `runs`: show recent run results, work counts, peak memory, and whether that run unmounted the recorder.
- `install-launchd`: write the run-on-mount LaunchAgent without loading it.
- `--install`: install and immediately activate a low-priority LaunchAgent with `StartOnMount`, `RunAtLoad`, and a 60-second fallback check. No manual plist editing or `launchctl` command is required.
- `--uninstall`: unload and remove the LaunchAgent while preserving recordings, notes, configuration, and application state.
- `--reset-state`: delete the configured application state directory and exit. This removes staged audio, the database, chunk checkpoints, transcripts, summaries, and logs; recorder files and Obsidian notes are not touched.

The default profile uses Whisper `small` without a large-model retry, keeping memory use and processing time substantially lower on a 16 GB Mac. The Python pipeline runs at nice level 20, macOS's lowest CPU scheduling priority, and all spawned FFmpeg, ffprobe, Whisper CLI, and helper processes inherit it. When summarization is enabled, the pipeline probes the local Ollama API before loading Whisper; if needed, it starts a detached, low-priority `ollama serve` and waits for readiness. Notifications report how many newly imported recordings are about to be transcribed, a final completed/failed summary when actual work finishes, and immediate errors; empty successful polls stay quiet. After a non-dry-run `run` finds new recorder files and completes with zero failures, that same run unmounts the configured recorder volume with `diskutil unmount`; a later empty poll can never unmount it on behalf of earlier work. Use `--maximum-accuracy` only when you explicitly want `large-v3` and can accept its much higher memory pressure. You can also use `--language auto` for automatic language detection and `--no-summary`, `--no-vad`, or `--no-notify` to disable individual stages.

## Configuration

Copy `config.example.toml` to:

```text
~/.config/auto-transcription/config.toml
```

Alternatively, pass `--config /path/to/config.toml`.

The configuration controls paths, language, model selection, confidence retry, bounded audio chunking, silence-aware voice detection, optional FFmpeg filtering, Ollama prompts, correction terms, audio retention, templates, notifications, and timeouts. Set `paths.vault = "auto"` (the default) to discover the current Obsidian vault from `~/Library/Application Support/obsidian/obsidian.json`, or provide a concrete path to pin one vault.

`default-note-template.txt` is the single default template and lists every supported variable. It records the source path and timestamp, content hash, audio duration and quality measurements, language, transcription backend/model/version, stage timings, summary model, and processing timestamp. Copy it elsewhere and set `output.template` to customize it.

The installer requires the hash-locked Python dependency set, validates at least 10 GB of free space, preserves a rollback copy of an existing environment, downloads configured models, runs a complete dependency check, and installs the LaunchAgent. Use `./setup.sh --upgrade` for an in-place Homebrew and Python-environment upgrade, `./setup.sh --uninstall` to remove automation only, or `./setup.sh --dry-run` to preview the transaction.

## Durable state

The default state root is:

```text
~/Library/Application Support/AutoTranscription
```

It contains:

- `Incoming`: verified local copies waiting for processing.
- `Processing`: the recording currently being handled.
- `Archive`: successfully processed source audio.
- `Failed`: audio whose latest stage failed and can be resumed.
- `Transcripts`: durable raw, corrected, and segment-level transcripts.
- `Summaries`: durable grounded summaries.
- `Chunks`: per-recording transcription checkpoints; decoded temporary audio is removed after every chunk.
- `Backups`: rotating pre-migration, pre-repair, and manual SQLite backups.
- `state.sqlite3`: fingerprints, hashes, stages, errors, timings, and output paths.
- `auto-transcription.log`: rotating diagnostic log.

Copies use a `.partial` suffix until repeated size/time checks, an audio-header check, a short decode, the complete copy size, and SHA-256 have all been verified. The first real recorder mount binds its volume UUID; later UUID mismatches are refused without reading audio. Copy failures use persistent exponential retry timing. SQLite state and atomic file renames make interruption recovery idempotent. Recordings longer than 15 minutes are decoded into one bounded mono WAV chunk at a time with two seconds of overlap. Each completed chunk is checkpointed, so a crash or restart resumes without repeating it. Every MLX inference call runs in a fresh worker process; worker exit forces macOS to reclaim Metal allocations before the next chunk. Whisper is therefore gone before Ollama loads its model, and Ollama is explicitly unloaded after each recording. A summary or note failure does not cause Whisper to run again.

## macOS automation

Generate a LaunchAgent that polls for the recorder and also drains any local processing queue:

```sh
.venv/bin/python auto_transcription.py install-launchd
```

The command prints the `launchctl bootstrap` command needed to activate it. The agent is intentionally small: it invokes this CLI rather than running a permanent desktop application.

## Tests

```sh
./test_transcription.sh
```

The isolated suite uses synthetic/mock transcription and summarization. It covers isolated MLX workers, bounded long-recording chunks, durable chunk resume, overlap de-duplication, model unloading, idempotency, duplicate content, reused filenames, Unicode paths, interrupted stages, persistent copy retry, recorder UUID enforcement, audio readiness and quality reporting, run history, database backup/export/migration, safe maintenance, uninstall preservation, summary failure recovery, note collisions, archive cleanup, templates, attachment hash verification, selective accuracy retry, locking, timeouts, and launchd output.

Run the real Apple Silicon integration check with generated spoken audio and the small test model:

```sh
./test_transcription.sh --live
```
