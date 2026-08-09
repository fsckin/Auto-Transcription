from __future__ import annotations

import contextlib
import datetime as dt
import json
import logging
import os
from pathlib import Path
import subprocess
import sqlite3
import sys
import tempfile
import time
import unittest
from unittest import mock
import urllib.error
import wave

import auto_transcription as at


class FailingSummarizer:
    def summarize(self, transcript: str) -> str:
        raise at.PipelineError("intentional summary failure")


class FakeHTTPResponse:
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def read(self) -> bytes:
        return b'{"response":"Recovered summary"}'


class FakeTagsResponse(FakeHTTPResponse):
    def read(self) -> bytes:
        return b'{"models":[]}'


class InstalledTagsResponse(FakeHTTPResponse):
    def read(self) -> bytes:
        return b'{"models":[{"name":"mistral:latest"}]}'


class FakeTranscriber:
    backend = "mock"

    def hybrid_retry(self, path, result):
        return result

    def version(self):
        return "test"

    def release_models(self):
        self.released = True


class CountingTranscriber(FakeTranscriber):

    def __init__(self, fail_on: int | None = None, prefix: str = "chunk") -> None:
        self.calls = 0
        self.fail_on = fail_on
        self.prefix = prefix
        self.released = False

    def transcribe(self, path, model, clips=None):
        self.calls += 1
        if self.calls == self.fail_on:
            raise at.PipelineError("intentional chunk interruption")
        text = f" {self.prefix}-{self.calls}"
        return {
            "text": text, "language": "en",
            "segments": [{"start": 0.0, "end": 0.5, "text": text,
                          "avg_logprob": -0.1, "no_speech_prob": 0.01}],
        }

class SequenceTranscriber(FakeTranscriber):

    def __init__(self, texts: list[str]) -> None:
        self.texts = list(texts)
        self.calls = 0
        self.models: list[str] = []
        self.released = False

    def transcribe(self, path, model, clips=None):
        self.calls += 1
        self.models.append(model)
        text = self.texts.pop(0)
        segments = [] if not text else [
            {"start": 0.0, "end": 0.5, "text": text,
             "avg_logprob": -0.1, "no_speech_prob": 0.01}
        ]
        return {"text": text, "language": "en", "segments": segments}

class AutoTranscriptionTests(unittest.TestCase):
    APPLEDOUBLE_HEADER = b"\x00\x05\x16\x07\x00\x02\x00\x00" + (b"\0" * 18)

    def test_default_profile_is_low_memory_and_fast(self) -> None:
        cfg = at.Config()
        self.assertEqual(cfg.model, "small")
        self.assertEqual(cfg.retry_model, "small")
        self.assertEqual(cfg.no_text_retry_model, "medium")
        self.assertFalse(cfg.hybrid_retry)
        self.assertEqual(cfg.nice_level, 20)
        self.assertTrue(cfg.unmount_on_success)

    def test_process_priority_is_lowered_to_configured_nice_level(self) -> None:
        with mock.patch.object(at.os, "getpriority", side_effect=[0, 20]), \
                mock.patch.object(at.os, "setpriority") as setpriority:
            self.assertEqual(at.apply_process_priority(20), 20)
        setpriority.assert_called_once_with(at.os.PRIO_PROCESS, 0, 20)

    def test_setup_dry_run_plans_complete_install_on_macos_bash(self) -> None:
        root = Path(self.temp.name) / "installer"
        root.mkdir()
        project = Path(__file__).resolve().parents[1]
        installer = root / "setup.sh"
        installer.write_text((project / "setup.sh").read_text(encoding="utf-8"), encoding="utf-8")
        installer.chmod(0o755)
        (root / "requirements-lock.txt").write_text("mlx-whisper==0.4.3 --hash=sha256:test\n", encoding="utf-8")
        (root / "default-note-template.txt").write_text("{{transcript}}\n", encoding="utf-8")
        (root / "auto_transcription.py").write_text("# placeholder\n", encoding="utf-8")
        mock_bin = root / "mock-bin"
        mock_bin.mkdir()
        uname = mock_bin / "uname"
        uname.write_text(
            '#!/bin/sh\n[ "$1" = "-s" ] && echo Darwin || echo arm64\n', encoding="utf-8",
        )
        uname.chmod(0o755)
        brew = mock_bin / "brew"
        brew.write_text(
            '#!/bin/sh\n[ "$1" = "--prefix" ] && { echo /mock/homebrew; exit 0; }\nexit 1\n',
            encoding="utf-8",
        )
        brew.chmod(0o755)
        environment = dict(
            os.environ,
            PATH=f"{mock_bin}:/usr/bin:/bin",
            AUTO_TRANSCRIPTION_BREW_BIN=str(brew),
            AUTO_TRANSCRIPTION_PYTHON_BIN=sys.executable,
        )

        result = subprocess.run(
            ["/bin/bash", str(installer), "--dry-run"],
            text=True, capture_output=True, env=environment, check=False,
        )

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("downloading configured Whisper and summary models", result.stdout)
        self.assertIn("--no-notify prepare", result.stdout)
        self.assertIn("--no-notify check", result.stdout)
        self.assertIn("--install", result.stdout)

    def test_successful_run_unmounts_only_configured_volume(self) -> None:
        cfg = at.Config(mount_point=Path("/Volumes/IC RECORDER"), unmount_on_success=True)
        completed = subprocess.CompletedProcess([], 0, "Volume unmounted", "")
        with mock.patch.object(at.platform, "system", return_value="Darwin"), \
                mock.patch.object(Path, "exists", return_value=True), \
                mock.patch.object(at.os.path, "ismount", return_value=True), \
                mock.patch.object(at.subprocess, "run", return_value=completed) as run:
            self.assertTrue(at.unmount_recorder(cfg, logging.getLogger("unmount-test")))
        self.assertEqual(run.call_args.args[0], ["/usr/sbin/diskutil", "unmount", "/Volumes/IC RECORDER"])

    def test_unmount_refuses_path_outside_volumes(self) -> None:
        cfg = at.Config(mount_point=Path("/"), unmount_on_success=True)
        with mock.patch.object(at.platform, "system", return_value="Darwin"), \
                mock.patch.object(at.os.path, "ismount", return_value=True):
            with self.assertRaises(at.PipelineError):
                at.unmount_recorder(cfg, logging.getLogger("unmount-safety-test"))

    def test_notification_delivery_failure_is_nonfatal(self) -> None:
        with mock.patch.object(at.platform, "system", return_value="Darwin"), \
                mock.patch.object(at.shutil, "which", return_value="/usr/bin/osascript"), \
                mock.patch.object(at.subprocess, "run", side_effect=subprocess.TimeoutExpired("osascript", 10)):
            at.notify("title", "message", True)

    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory(prefix="auto-transcription-test-")
        root = Path(self.temp.name)
        self.cfg = at.Config(
            mount_point=root / "recorder",
            vault=root / "vault",
            state_root=root / "state",
            backend="mock",
            model="large-v3-turbo",
            retry_model="large-v3",
            hybrid_retry=True,
            stable_wait_seconds=0,
            vad_enabled=False,
            summary_enabled=True,
            summary_model="mock",
            notify=False,
            prevent_sleep=False,
        )
        self.cfg.recorder_folder.mkdir(parents=True)
        at.ensure_private_directories(self.cfg)
        self.logger = logging.getLogger(f"test-{id(self)}")
        self.logger.handlers.clear()
        self.logger.addHandler(logging.NullHandler())
        self.db = at.StateDB(self.cfg.db_path)
        self.pipeline = at.Pipeline(self.cfg, self.db, self.logger)

    def tearDown(self) -> None:
        self.db.close()
        self.temp.cleanup()

    def audio(self, name: str, content: bytes = b"audio-one") -> Path:
        path = self.cfg.recorder_folder / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
        stamp = time.time_ns()
        os.utime(path, ns=(stamp, stamp))
        return path

    def wav_audio(self, name: str, seconds: float) -> Path:
        path = self.cfg.recorder_folder / name
        frames = int(seconds * 16_000)
        with wave.open(str(path), "wb") as handle:
            handle.setnchannels(1)
            handle.setsampwidth(2)
            handle.setframerate(16_000)
            handle.writeframes(b"\0\0" * frames)
        stamp = time.time_ns()
        os.utime(path, ns=(stamp, stamp))
        return path

    def latest_row(self):
        return self.db.conn.execute("SELECT * FROM recordings ORDER BY id DESC LIMIT 1").fetchone()

    def write_cli_config(self, root: Path, *, notify: bool = False, unmount: bool = True) -> Path:
        config = root / "config.toml"
        config.write_text(
            f'''[paths]\nmount_point = "{root / "recorder"}"\nvault = "{root / "vault"}"\nstate_root = "{root / "state"}"\n'''
            '''[import]\nstable_wait_seconds = 0\n'''
            '''[transcription]\nbackend = "mock"\nmodel = "tiny"\nretry_model = "tiny"\nhybrid_retry = false\nvad_enabled = false\n'''
            '''[summarization]\nenabled = false\nmodel = "mock"\n'''
            f'''[behavior]\nnotify = {str(notify).lower()}\nprevent_sleep = false\nunmount_on_success = {str(unmount).lower()}\n''',
            encoding="utf-8",
        )
        return config

    def test_end_to_end_and_idempotency(self) -> None:
        self.audio("VOICE 001.mp3")
        imported = self.pipeline.import_new()
        self.assertEqual(imported["ready"], 1)
        processed = self.pipeline.process_pending()
        self.assertEqual(processed, {"completed": 1, "failed": 0})
        row = self.latest_row()
        self.assertEqual(row["status"], "completed")
        self.assertTrue(Path(row["transcript_path"]).exists())
        self.assertTrue(Path(row["summary_path"]).exists())
        self.assertTrue(Path(row["note_path"]).exists())
        self.assertEqual(self.pipeline.import_new()["found"], 0)
        self.assertEqual(self.pipeline.process_pending(), {"completed": 0, "failed": 0})

    def test_scan_uses_apple_metadata_header_not_filename_alone(self) -> None:
        self.audio("._named-sidecar.mp3", self.APPLEDOUBLE_HEADER)
        self.audio("renamed-sidecar.mp3", self.APPLEDOUBLE_HEADER)
        self.audio("._legitimate-recording.mp3", b"ID3legitimate audio")

        candidates = at.scan_recorder(self.cfg, self.db)

        self.assertEqual([item.path.name for item in candidates], ["._legitimate-recording.mp3"])

    def test_existing_failed_apple_metadata_becomes_ignored(self) -> None:
        source = self.audio("._old-recording.mp3", self.APPLEDOUBLE_HEADER)
        stat = source.stat()
        record_id = self.db.add_detected(source, stat, at.source_fingerprint(source, stat))
        local = self.cfg.failed_dir / "renamed-old-recording.mp3"
        local.write_bytes(self.APPLEDOUBLE_HEADER)
        self.db.update(
            record_id, status="failed", failed_stage="transcribe",
            last_error="old failure", local_path=str(local),
        )

        self.assertEqual(at.ignore_known_macos_metadata(self.db, self.logger), 1)
        row = self.db.get(record_id)
        self.assertEqual(row["status"], "ignored")
        self.assertIsNone(row["failed_stage"])
        self.assertIn("AppleSingle/AppleDouble", row["last_error"])

    def test_blank_transcript_with_silent_signal_is_terminal_no_speech(self) -> None:
        self.cfg.summary_enabled = False
        self.audio("silent.mp3")
        self.pipeline.import_new()
        row = self.latest_row()
        transcriber = SequenceTranscriber([""])
        self.pipeline.transcriber = transcriber

        quality = at.AudioQuality(1.0, -80.0, -70.0, 0.0, 0.0, ("too-little-active-audio",))
        with mock.patch.object(at, "ffprobe_duration", return_value=1.0), \
                mock.patch.object(at, "analyze_audio_quality", return_value=quality):
            self.assertEqual(self.pipeline.process_record(int(row["id"])), "no_speech")

        row = self.db.get(int(row["id"]))
        self.assertEqual(row["status"], "no_speech")
        self.assertEqual(row["no_text_retries"], 0)
        self.assertEqual(Path(row["local_path"]).parent, self.cfg.archive_dir)
        self.assertEqual(transcriber.calls, 1)

    def test_audible_blank_transcript_gets_one_enhanced_retry(self) -> None:
        self.cfg.summary_enabled = False
        self.audio("quiet-speech.mp3")
        self.pipeline.import_new()
        row = self.latest_row()
        transcriber = SequenceTranscriber(["", " recovered speech"])
        self.pipeline.transcriber = transcriber

        def passthrough(path, cfg):
            return contextlib.nullcontext(path)

        with mock.patch.object(at, "ffprobe_duration", return_value=1.0), \
                mock.patch.object(at, "analyze_audio_quality", return_value=at.AudioQuality(1.0, -38.0, -12.0, 0.8, 0.0)), \
                mock.patch.object(at, "preprocessed_audio", side_effect=passthrough), \
                mock.patch.object(at, "Transcriber", return_value=transcriber):
            self.assertEqual(self.pipeline.process_record(int(row["id"])), "completed")

        row = self.db.get(int(row["id"]))
        self.assertEqual(row["no_text_retries"], 1)
        self.assertEqual(row["model"], "medium")
        self.assertEqual(transcriber.models, ["large-v3-turbo", "medium"])
        self.assertIn("recovered speech", Path(row["transcript_path"]).read_text(encoding="utf-8"))

    def test_audible_recording_still_blank_becomes_needs_review_once(self) -> None:
        self.cfg.summary_enabled = False
        self.audio("needs-review.mp3")
        self.pipeline.import_new()
        row = self.latest_row()
        transcriber = SequenceTranscriber(["", ""])
        self.pipeline.transcriber = transcriber

        def passthrough(path, cfg):
            return contextlib.nullcontext(path)

        with mock.patch.object(at, "ffprobe_duration", return_value=1.0), \
                mock.patch.object(at, "analyze_audio_quality", return_value=at.AudioQuality(1.0, -35.0, -8.0, 0.8, 0.0)), \
                mock.patch.object(at, "preprocessed_audio", side_effect=passthrough), \
                mock.patch.object(at, "Transcriber", return_value=transcriber):
            self.assertEqual(self.pipeline.process_record(int(row["id"])), "needs_review")
            self.assertEqual(self.pipeline.process_record(int(row["id"])), "needs_review")

        row = self.db.get(int(row["id"]))
        self.assertEqual(row["status"], "needs_review")
        self.assertEqual(row["no_text_retries"], 1)
        self.assertEqual(Path(row["local_path"]).parent, self.cfg.failed_dir)
        self.assertEqual(transcriber.calls, 2)

    def test_run_notifies_new_count_and_final_summary_but_empty_poll_is_quiet(self) -> None:
        root = Path(self.temp.name) / "notify-run"
        recorder = root / "recorder" / "REC_FILE"
        recorder.mkdir(parents=True)
        (recorder / "notify.mp3").write_bytes(b"audio")
        config = self.write_cli_config(root, notify=True, unmount=False)
        with mock.patch.object(at, "apply_process_priority", return_value=20), \
                mock.patch.object(at, "notify") as notification:
            self.assertEqual(at.main(["--config", str(config), "--quiet", "run"]), 0)
            self.assertEqual([call.args[1] for call in notification.call_args_list], [
                "Transcribing 1 new recording", "Finished: 1 completed",
            ])
            notification.reset_mock()
            self.assertEqual(at.main(["--config", str(config), "--quiet", "run"]), 0)
            notification.assert_not_called()

    def test_dry_run_is_quiet_and_failure_notifies(self) -> None:
        self.cfg.notify = True
        self.audio("quiet-failure.mp3")
        with mock.patch.object(at, "notify") as notification:
            self.pipeline.import_new(dry_run=True)
            notification.assert_not_called()
            self.pipeline.import_new()
            notification.reset_mock()
            self.pipeline.summarizer = FailingSummarizer()
            self.pipeline.process_pending()
            notification.assert_called_once()
            self.assertEqual(notification.call_args.args[0], "Auto Transcription Error")
            self.assertIn("Summary failed for quiet-failure.mp3", notification.call_args.args[1])

    def test_duplicate_content_with_different_filename(self) -> None:
        self.audio("one.mp3", b"same")
        self.pipeline.import_new()
        self.pipeline.process_pending()
        self.audio("renamed.mp3", b"same")
        imported = self.pipeline.import_new()
        self.assertEqual(imported["duplicate"], 1)
        self.assertEqual(self.db.counts()["duplicate"], 1)

    def test_reused_filename_with_new_content(self) -> None:
        path = self.audio("reused.mp3", b"first")
        self.pipeline.import_new()
        self.pipeline.process_pending()
        path.write_bytes(b"second and different")
        newer = time.time_ns() + 2_000_000
        os.utime(path, ns=(newer, newer))
        self.assertEqual(self.pipeline.import_new()["ready"], 1)
        self.assertEqual(self.pipeline.process_pending()["completed"], 1)
        self.assertEqual(self.db.counts()["completed"], 2)

    def test_unicode_and_nested_filename(self) -> None:
        self.audio("nested/Voice café 日本語 01.mp3")
        self.assertEqual(self.pipeline.import_new()["ready"], 1)
        self.assertEqual(self.pipeline.process_pending()["completed"], 1)

    def test_summary_failure_resumes_without_retranscription(self) -> None:
        self.audio("summary-failure.mp3")
        self.pipeline.import_new()
        self.pipeline.summarizer = FailingSummarizer()
        self.assertEqual(self.pipeline.process_pending()["failed"], 1)
        row = self.latest_row()
        self.assertEqual(row["failed_stage"], "summary")
        self.assertTrue(Path(row["transcript_path"]).exists())
        self.assertEqual(Path(row["local_path"]).parent, self.cfg.failed_dir)
        self.assertEqual(row["attempts"], 1)
        self.pipeline.summarizer = at.OllamaSummarizer(self.cfg, self.logger)
        self.assertEqual(self.pipeline.process_record(int(row["id"])), "completed")
        row = self.db.get(int(row["id"]))
        self.assertEqual(row["attempts"], 1)
        self.assertEqual(Path(row["local_path"]).parent, self.cfg.archive_dir)

    def test_interrupted_state_recovery(self) -> None:
        self.audio("interrupted.mp3")
        self.pipeline.import_new()
        row = self.latest_row()
        self.db.update(int(row["id"]), status="transcribing")
        self.assertEqual(at.recover_interrupted(self.db, self.cfg, self.logger), 1)
        self.assertEqual(self.db.get(int(row["id"]))["status"], "ready")

    def test_long_recording_uses_bounded_chunks_and_resumes_checkpoints(self) -> None:
        self.cfg.summary_enabled = False
        self.cfg.chunk_seconds = 1.0
        self.cfg.max_direct_seconds = 1.0
        self.cfg.chunk_overlap_seconds = 0.2
        self.wav_audio("long.wav", 2.5)
        self.pipeline.import_new()
        row = self.latest_row()

        first = CountingTranscriber(fail_on=2, prefix="first")
        self.pipeline.transcriber = first
        self.assertEqual(self.pipeline.process_record(int(row["id"])), "failed")
        checkpoints = list(self.cfg.chunks_dir.rglob("chunk-*.json"))
        self.assertEqual(len(checkpoints), 1)
        self.assertTrue(first.released)

        second = CountingTranscriber(prefix="resumed")
        self.pipeline.transcriber = second
        self.assertEqual(self.pipeline.process_record(int(row["id"])), "completed")
        self.assertEqual(second.calls, 2)
        row = self.db.get(int(row["id"]))
        transcript = Path(row["transcript_path"]).read_text(encoding="utf-8")
        self.assertIn("first-1", transcript)
        self.assertIn("resumed-1", transcript)
        self.assertEqual(len(list(self.cfg.chunks_dir.rglob("chunk-*.json"))), 3)

    def test_overlap_segments_are_kept_by_core_midpoint_once(self) -> None:
        first = at.offset_and_trim_chunk_result({
            "text": "boundary", "language": "en",
            "segments": [{"start": 0.9, "end": 1.1, "text": "boundary"}],
        }, 0.0, 0.0, 1.0)
        second = at.offset_and_trim_chunk_result({
            "text": "boundary", "language": "en",
            "segments": [{"start": 0.1, "end": 0.3, "text": "boundary"}],
        }, 0.8, 1.0, 2.0)
        merged = at.merge_chunk_results([first, second])
        self.assertEqual(len(merged["segments"]), 1)
        self.assertEqual(merged["text"], "boundary")

    def test_large_unknown_duration_is_rejected_before_backend(self) -> None:
        path = self.audio("unknown.mp3", b"x" * 32)
        self.pipeline.import_new()
        row = self.latest_row()
        self.cfg.max_unknown_duration_bytes = 16
        transcriber = CountingTranscriber()
        self.pipeline.transcriber = transcriber
        with mock.patch.object(at, "ffprobe_duration", return_value=None):
            self.assertEqual(self.pipeline.process_record(int(row["id"])), "failed")
        self.assertEqual(transcriber.calls, 0)

    def test_failed_copy_can_be_retried(self) -> None:
        path = self.audio("copy-retry.mp3")
        stat = path.stat()
        fingerprint = at.source_fingerprint(path, stat)
        record_id = self.db.add_detected(path, stat, fingerprint)
        self.db.update(record_id, status="failed", failed_stage="copy", last_error="test")
        self.assertEqual(self.pipeline.import_new()["ready"], 1)

    def test_note_collision_does_not_overwrite(self) -> None:
        self.audio("collision.mp3")
        self.pipeline.import_new()
        row = self.latest_row()
        recorded = dt.datetime.fromtimestamp(row["source_mtime_ns"] / 1_000_000_000).astimezone()
        base = f"{recorded:%Y-%m-%d %H-%M-%S}_collision_{row['sha256'][:8]}"
        collision = self.cfg.notes_folder / f"{base}.md"
        collision.write_text("user content\n", encoding="utf-8")
        self.pipeline.process_pending()
        row = self.db.get(int(row["id"]))
        self.assertEqual(collision.read_text(encoding="utf-8"), "user content\n")
        self.assertNotEqual(Path(row["note_path"]), collision)

    def test_custom_template_and_corrections(self) -> None:
        template = Path(self.temp.name) / "template.txt"
        template.write_text("SUMMARY={{summary}}\nTEXT={{transcript}}\n", encoding="utf-8")
        self.cfg.note_template = template
        self.cfg.corrections = {"Mock": "Corrected"}
        self.audio("template.mp3")
        self.pipeline.import_new()
        self.pipeline.process_pending()
        row = self.latest_row()
        note = Path(row["note_path"]).read_text(encoding="utf-8")
        self.assertIn("SUMMARY=Mock summary", note)
        self.assertIn("TEXT=Corrected transcript", note)

    def test_pipeline_lock_and_stale_lock(self) -> None:
        with at.PipelineLock(self.cfg.lock_dir):
            with self.assertRaises(at.PipelineError):
                with at.PipelineLock(self.cfg.lock_dir):
                    pass
        self.cfg.lock_dir.mkdir()
        (self.cfg.lock_dir / "pid").write_text("99999999\n", encoding="utf-8")
        with at.PipelineLock(self.cfg.lock_dir):
            self.assertTrue(self.cfg.lock_dir.exists())

    def test_reset_state_option_deletes_only_configured_state_root(self) -> None:
        root = Path(self.temp.name) / "resettable-state"
        (root / "Chunks" / "hash").mkdir(parents=True)
        (root / "state.sqlite3").write_bytes(b"state")
        self.assertEqual(at.main(["--state-root", str(root), "--reset-state"]), 0)
        self.assertFalse(root.exists())
        self.assertTrue(self.cfg.vault.exists())
        self.assertTrue(self.cfg.recorder_folder.exists())

    def test_reset_state_refuses_protected_path(self) -> None:
        cfg = at.Config(state_root=Path.home())
        with self.assertRaises(at.PipelineError):
            at.reset_application_state(cfg)

    def test_reset_state_refuses_active_pipeline(self) -> None:
        root = Path(self.temp.name) / "locked-state"
        root.mkdir()
        cfg = at.Config(state_root=root, vault=self.cfg.vault, mount_point=self.cfg.mount_point)
        with at.PipelineLock(cfg.lock_dir):
            with self.assertRaises(at.PipelineError):
                at.reset_application_state(cfg)
        self.assertTrue(root.exists())

    def test_chunking_respects_limit(self) -> None:
        chunks = at.OllamaSummarizer.chunks("paragraph\n" * 100, 80)
        self.assertGreater(len(chunks), 1)
        self.assertTrue(all(len(chunk) <= 80 for chunk in chunks))

    def test_ollama_transient_failure_is_retried(self) -> None:
        self.cfg.summary_model = "mistral"
        self.cfg.summary_retry_count = 2
        summarizer = at.OllamaSummarizer(self.cfg, self.logger)
        with mock.patch.object(at.urllib.request, "urlopen", side_effect=[urllib.error.URLError("temporary"), FakeHTTPResponse()]), \
                mock.patch.object(at.time, "sleep"):
            self.assertEqual(summarizer._generate("prompt"), "Recovered summary")

    def test_ollama_model_is_explicitly_unloaded(self) -> None:
        self.cfg.summary_model = "mistral"
        summarizer = at.OllamaSummarizer(self.cfg, self.logger)
        with mock.patch.object(at.urllib.request, "urlopen", return_value=FakeHTTPResponse()) as opened:
            summarizer.release_model()
        request = opened.call_args.args[0]
        payload = json.loads(request.data.decode("utf-8"))
        self.assertEqual(payload["keep_alive"], 0)

    def test_running_ollama_is_reused_without_spawning(self) -> None:
        self.cfg.summary_model = "mistral"
        with mock.patch.object(at, "probe_ollama", return_value={"models": []}), \
                mock.patch.object(at.subprocess, "Popen") as popen:
            self.assertFalse(at.ensure_ollama_running(self.cfg, self.logger))
        popen.assert_not_called()

    def test_missing_ollama_server_is_started_low_priority_and_waited_for(self) -> None:
        self.cfg.summary_model = "mistral"
        process = mock.Mock(pid=4321)
        process.poll.return_value = None
        with mock.patch.object(at, "probe_ollama", side_effect=[urllib.error.URLError("offline"), {"models": []}]), \
                mock.patch.object(at.shutil, "which", return_value="/opt/homebrew/bin/ollama"), \
                mock.patch.object(at.platform, "system", return_value="Darwin"), \
                mock.patch.object(at.subprocess, "Popen", return_value=process) as popen:
            self.assertTrue(at.ensure_ollama_running(self.cfg, self.logger))
        command = popen.call_args.args[0]
        self.assertEqual(command, ["/usr/bin/nice", "-n", "20", "/opt/homebrew/bin/ollama", "serve"])
        self.assertTrue(self.cfg.ollama_log_path.exists())

    def test_ollama_is_not_started_when_summaries_are_disabled(self) -> None:
        self.cfg.summary_enabled = False
        with mock.patch.object(at.subprocess, "Popen") as popen:
            self.assertFalse(at.ensure_ollama_running(self.cfg, self.logger))
        popen.assert_not_called()

    def test_prepare_starts_ollama_before_checking_or_pulling_models(self) -> None:
        self.cfg.summary_model = "mistral"
        transcriber = mock.Mock()
        with mock.patch.object(at, "Transcriber", return_value=transcriber), \
                mock.patch.object(at, "ensure_ollama_running") as ensure, \
                mock.patch.object(at.urllib.request, "urlopen", return_value=InstalledTagsResponse()):
            at.prepare_dependencies(self.cfg, self.logger)
        transcriber.prepare_models.assert_called_once_with()
        ensure.assert_called_once_with(self.cfg, self.logger)

    def test_openai_model_references_are_released(self) -> None:
        transcriber = at.Transcriber(self.cfg, self.logger)
        transcriber.backend = "openai"
        transcriber._openai_models = {"large-v3": object()}
        transcriber.release_models()
        self.assertFalse(hasattr(transcriber, "_openai_models"))

    def test_mlx_transcriptions_run_in_fresh_worker_processes(self) -> None:
        self.cfg.backend = "mlx"
        commands = []

        def fake_run(command, **kwargs):
            commands.append(command)
            response_path = Path(command[-1])
            response_path.write_text(json.dumps({
                "text": "isolated", "language": "en",
                "segments": [{"start": 0.0, "end": 1.0, "text": "isolated"}],
            }), encoding="utf-8")
            return subprocess.CompletedProcess(command, 0, "", "")

        transcriber = at.Transcriber(self.cfg, self.logger)
        with mock.patch.object(at.subprocess, "run", side_effect=fake_run):
            first = transcriber.transcribe(Path("first.wav"), "small")
            second = transcriber.transcribe(Path("second.wav"), "small")
        self.assertEqual(first["text"], "isolated")
        self.assertEqual(second["text"], "isolated")
        self.assertEqual(len(commands), 2)
        self.assertTrue(all(command[1] == str(Path(at.__file__).resolve()) for command in commands))
        self.assertTrue(all(command[2] == "_mlx-worker" for command in commands))
        self.assertNotEqual(commands[0][-2], commands[1][-2])

    def test_hybrid_retry_replaces_low_confidence_region(self) -> None:
        transcriber = at.Transcriber(self.cfg, self.logger)
        result = {
            "text": "bad good",
            "language": "en",
            "segments": [
                {"start": 0.0, "end": 1.0, "text": "bad", "avg_logprob": -1.2, "no_speech_prob": 0.01},
                {"start": 2.0, "end": 3.0, "text": " good", "avg_logprob": -0.1, "no_speech_prob": 0.01},
            ],
        }
        transcriber.transcribe = lambda path, model, clips=None: {
            "text": "better", "language": "en",
            "segments": [{"start": 0.0, "end": 1.1, "text": "better", "avg_logprob": -0.1, "no_speech_prob": 0.01}],
        }
        retried = transcriber.hybrid_retry(Path("unused.mp3"), result)
        self.assertIn("better", retried["text"])
        self.assertIn("good", retried["text"])
        self.assertNotIn("bad", retried["text"])

    def test_operation_timeout(self) -> None:
        with self.assertRaises(at.PipelineError):
            with at.operation_timeout(1, "timed out"):
                time.sleep(2)

    def test_audio_attachment_is_hash_verified_and_embedded(self) -> None:
        self.cfg.copy_audio_to_vault = True
        self.audio("attachment.mp3")
        self.pipeline.import_new()
        self.pipeline.process_pending()
        row = self.latest_row()
        attachments = list((self.cfg.vault / self.cfg.attachments_folder).glob("*.mp3"))
        self.assertEqual(len(attachments), 1)
        self.assertEqual(at.sha256_file(attachments[0]), row["sha256"])
        note = Path(row["note_path"]).read_text(encoding="utf-8")
        self.assertIn("![[Recordings/Audio/", note)

    def test_cleanup_requires_age_and_preserves_notes(self) -> None:
        self.audio("cleanup.mp3")
        self.pipeline.import_new()
        self.pipeline.process_pending()
        row = self.latest_row()
        audio = Path(row["local_path"])
        old = time.time() - 10 * 86400
        os.utime(audio, (old, old))
        self.assertEqual(at.cleanup_archives(self.cfg, self.db, 5, True, self.logger), 0)
        old_iso = dt.datetime.fromtimestamp(old, dt.timezone.utc).isoformat(timespec="seconds")
        self.db.update(int(row["id"]), completed_at=old_iso)
        self.assertEqual(at.cleanup_archives(self.cfg, self.db, 5, True, self.logger), 1)
        self.assertTrue(audio.exists())
        self.assertEqual(at.cleanup_archives(self.cfg, self.db, 5, False, self.logger), 1)
        self.assertFalse(audio.exists())
        self.assertTrue(Path(row["note_path"]).exists())

    def test_launchd_plist_escapes_paths(self) -> None:
        plist = at.launchd_plist(self.cfg, 60, Path("/tmp/a&b.py"), None)
        data = at.plistlib.loads(plist.encode("utf-8"))
        self.assertEqual(data["ProgramArguments"][1], "/tmp/a&b.py")
        self.assertEqual(data["StartInterval"], 60)
        self.assertTrue(data["StartOnMount"])
        self.assertTrue(data["RunAtLoad"])
        self.assertEqual(data["Nice"], 20)
        self.assertEqual(data["EnvironmentVariables"]["PATH"], at.LAUNCHD_PATH)

    def test_activate_launchd_replaces_bootstraps_and_verifies(self) -> None:
        plist = Path(self.temp.name) / "local.auto-transcription.plist"
        plist.write_text("plist", encoding="utf-8")
        results = [
            subprocess.CompletedProcess([], 3, "", "not loaded"),
            subprocess.CompletedProcess([], 0, "", ""),
            subprocess.CompletedProcess([], 0, "service", ""),
        ]
        with mock.patch.object(at.platform, "system", return_value="Darwin"), \
                mock.patch.object(at.subprocess, "run", side_effect=results) as run:
            at.activate_launchd(plist, self.logger)
        service = f"gui/{os.getuid()}/local.auto-transcription"
        self.assertEqual(run.call_args_list[0].args[0], ["/bin/launchctl", "bootout", service])
        self.assertEqual(run.call_args_list[1].args[0], ["/bin/launchctl", "bootstrap", f"gui/{os.getuid()}", str(plist)])
        self.assertEqual(run.call_args_list[2].args[0], ["/bin/launchctl", "print", service])

    def test_install_flag_writes_and_activates_agent_without_command(self) -> None:
        root = Path(self.temp.name) / "install"
        plist = root / "local.auto-transcription.plist"
        with mock.patch.object(at, "apply_process_priority", return_value=20), \
                mock.patch.object(at, "install_launchd", return_value=plist) as write_agent, \
                mock.patch.object(at, "activate_launchd") as activate:
            result = at.main([
                "--state-root", str(root / "state"), "--vault", str(root / "vault"),
                "--mount", str(root / "recorder"), "--install",
            ])
        self.assertEqual(result, 0)
        write_agent.assert_called_once()
        activate.assert_called_once_with(plist, mock.ANY)

    def test_cli_run_is_idempotent(self) -> None:
        root = Path(self.temp.name) / "cli"
        recorder = root / "recorder" / "REC_FILE"
        vault = root / "vault"
        state = root / "state"
        recorder.mkdir(parents=True)
        vault.mkdir(parents=True)
        (recorder / "CLI TEST.mp3").write_bytes(b"cli-audio")
        config = self.write_cli_config(root)
        with mock.patch.object(at, "apply_process_priority", return_value=20), \
                mock.patch.object(at, "unmount_recorder", return_value=False) as unmount:
            self.assertEqual(at.main(["--config", str(config), "--quiet", "run"]), 0)
            self.assertEqual(at.main(["--config", str(config), "--quiet", "run"]), 0)
        self.assertEqual(unmount.call_count, 1)
        with at.StateDB(state / "state.sqlite3") as cli_db:
            self.assertEqual(cli_db.counts(), {"completed": 1})

    def test_cli_import_and_process_share_run_lifecycle(self) -> None:
        root = Path(self.temp.name) / "cli-stages"
        recorder = root / "recorder" / "REC_FILE"
        recorder.mkdir(parents=True)
        (root / "vault").mkdir()
        (recorder / "STAGED.mp3").write_bytes(b"staged-audio")
        config = self.write_cli_config(root, unmount=False)
        with mock.patch.object(at, "apply_process_priority", return_value=20):
            self.assertEqual(at.main(["--config", str(config), "--quiet", "import"]), 0)
            self.assertEqual(at.main(["--config", str(config), "--quiet", "process"]), 0)
        with at.StateDB(root / "state" / "state.sqlite3") as cli_db:
            self.assertEqual(cli_db.counts(), {"completed": 1})
            commands = [row[0] for row in cli_db.conn.execute("SELECT command FROM runs ORDER BY id")]
            self.assertEqual(commands, ["import", "process"])

    def test_compatibility_launcher_handles_empty_arguments_under_nounset(self) -> None:
        root = Path(__file__).resolve().parents[1]
        environment = dict(os.environ, AUTO_TRANSCRIPTION_PYTHON="/bin/echo")
        completed = subprocess.run(
            ["/bin/bash", str(root / "transcribe_and_summarize.sh")],
            text=True, capture_output=True, env=environment, check=True,
        )
        self.assertEqual(completed.stdout.strip(), f"{root / 'auto_transcription.py'} run")
        for command in ("doctor", "maintenance", "database", "runs"):
            completed = subprocess.run(
                ["/bin/bash", str(root / "transcribe_and_summarize.sh"), command],
                text=True, capture_output=True, env=environment, check=True,
            )
            self.assertEqual(completed.stdout.strip(), f"{root / 'auto_transcription.py'} {command}")

    def test_recorder_uuid_is_bound_then_mismatch_is_refused(self) -> None:
        cfg = at.Config(
            mount_point=Path("/Volumes/TEST RECORDER"),
            vault=self.cfg.vault,
            state_root=self.cfg.state_root,
        )
        first = at.VolumeIdentity("AAA-BBB", "disk9s1", "TEST RECORDER")
        with mock.patch.object(Path, "is_dir", return_value=True), \
                mock.patch.object(at, "read_volume_identity", return_value=first):
            identity = at.verify_recorder_identity(cfg, self.db, self.logger)
        self.assertEqual(identity, first)
        self.assertEqual(self.db.metadata_get("recorder_volume_uuid"), "AAA-BBB")

        wrong = at.VolumeIdentity("WRONG-UUID", "disk8s1", "OTHER")
        with mock.patch.object(Path, "is_dir", return_value=True), \
                mock.patch.object(at, "read_volume_identity", return_value=wrong):
            with self.assertRaisesRegex(at.PipelineError, "UUID mismatch"):
                at.verify_recorder_identity(cfg, self.db, self.logger)

    def test_copy_retry_backoff_persists_and_scan_waits_until_due(self) -> None:
        source = self.audio("retry-later.mp3")
        stat = source.stat()
        record_id = self.db.add_detected(source, stat, at.source_fingerprint(source, stat))
        self.cfg.retry_base_seconds = 60
        self.cfg.retry_max_seconds = 600

        before = dt.datetime.now(dt.timezone.utc)
        at.schedule_copy_retry(self.db, record_id, self.cfg, "first")
        first = self.db.get(record_id)
        first_retry = dt.datetime.fromisoformat(first["next_retry_at"])
        self.assertEqual(first["consecutive_failures"], 1)
        self.assertGreaterEqual((first_retry - before).total_seconds(), 59)
        self.assertEqual(at.scan_recorder(self.cfg, self.db), [])

        at.schedule_copy_retry(self.db, record_id, self.cfg, "second")
        second = self.db.get(record_id)
        second_retry = dt.datetime.fromisoformat(second["next_retry_at"])
        self.assertEqual(second["consecutive_failures"], 2)
        self.assertGreaterEqual((second_retry - before).total_seconds(), 119)
        self.db.update(record_id, next_retry_at="2000-01-01T00:00:00")
        self.assertEqual([item.path for item in at.scan_recorder(self.cfg, self.db)], [source])

    def test_audio_headers_and_stability_checks_reject_incomplete_files(self) -> None:
        wav = self.wav_audio("valid.wav", 0.1)
        self.assertTrue(at.audio_header_matches(wav))
        self.assertTrue(at.audio_header_matches(self.audio("valid.mp3", b"ID3\x04\0\0payload")))
        self.assertFalse(at.audio_header_matches(self.audio("invalid.mp3", b"not audio")))

        empty = self.audio("empty.wav", b"")
        candidate = at.Candidate(empty, empty.stat(), "empty")
        with self.assertRaisesRegex(at.PipelineError, "empty"):
            at.wait_until_stable(candidate, 0, checks=3)

    def test_audio_quality_reports_silence_and_clipping(self) -> None:
        path = self.audio("quality.mp3")
        stderr = "\n".join([
            "silence_start: 0.0", "silence_end: 9.9",
            "RMS level dB: -50.0", "Peak level dB: 0.0",
            "Peak count: 1600", "Number of samples: 160000",
        ])
        completed = subprocess.CompletedProcess([], 0, "", stderr)
        with mock.patch.object(at, "ffprobe_duration", return_value=10.0), \
                mock.patch.object(at.shutil, "which", return_value="/opt/homebrew/bin/ffmpeg"), \
                mock.patch.object(at.subprocess, "run", return_value=completed):
            quality = at.analyze_audio_quality(path, self.cfg, self.logger)
        self.assertIsNotNone(quality)
        self.assertAlmostEqual(quality.active_seconds, 0.1)
        self.assertAlmostEqual(quality.clipping_percent, 1.0)
        self.assertIn("too-little-active-audio", quality.warnings)
        self.assertIn("mostly-silent", quality.warnings)
        self.assertIn("clipping", quality.warnings)
        self.assertTrue(quality.is_silent(self.cfg))
        self.assertEqual(at.AudioQuality.from_json(quality.to_json()), quality)
        self.assertEqual(at.silence_intervals(stderr, 10.0), [(0.0, 9.9)])
        self.assertEqual(at.audio_quality_warnings("broken"), ("invalid-quality-metadata",))

    def test_database_backup_export_integrity_and_run_history(self) -> None:
        run_id = self.db.start_run("run", self.cfg, "AAA-BBB")
        self.db.finish_run(run_id, "completed", {
            "found": 2, "imported": 2, "completed": 2, "unmounted": True,
        })
        run = self.db.conn.execute("SELECT * FROM runs WHERE id=?", (run_id,)).fetchone()
        self.assertEqual(run["status"], "completed")
        self.assertEqual(run["found"], 2)
        self.assertEqual(run["unmounted"], 1)
        self.assertIsNotNone(run["peak_rss_bytes"])

        backup = self.db.create_backup("test")
        export = Path(self.temp.name) / "state-export.sql"
        self.assertTrue(backup.is_file())
        self.assertTrue(self.db.export_sql(export).is_file())
        self.assertIn("CREATE TABLE recordings", export.read_text(encoding="utf-8"))
        self.assertEqual(self.db.integrity_check(), (True, "ok"))
        with self.assertRaises(at.PipelineError):
            self.db.create_backup(output=self.cfg.db_path)

    def test_schema_migration_creates_backup_and_current_version(self) -> None:
        path = Path(self.temp.name) / "migration" / "state.sqlite3"
        path.parent.mkdir()
        connection = sqlite3.connect(path)
        connection.execute(
            "CREATE TABLE recordings (id INTEGER PRIMARY KEY, status TEXT, sha256 TEXT, "
            "generated_title TEXT, summary_model TEXT)"
        )
        connection.commit()
        connection.close()
        with at.StateDB(path) as migrated:
            version = migrated.conn.execute("PRAGMA user_version").fetchone()[0]
            columns = {str(row[1]) for row in migrated.conn.execute("PRAGMA table_info(recordings)")}
        self.assertEqual(version, at.DB_SCHEMA_VERSION)
        self.assertIn("transcription_generation", columns)
        self.assertIn("no_text_retries", columns)
        self.assertIn("next_retry_at", columns)
        self.assertIn("audio_quality_json", columns)
        self.assertEqual(len(list((path.parent / "Backups").glob("state-pre-migration-*.sqlite3"))), 1)

    def test_state_maintenance_removes_only_safe_generated_files(self) -> None:
        self.audio("maintenance.mp3")
        self.pipeline.import_new()
        self.pipeline.process_pending()
        completed = self.latest_row()
        digest = completed["sha256"]
        chunks = self.cfg.chunks_dir / digest
        chunks.mkdir()
        (chunks / "chunk.wav").write_bytes(b"generated")
        partial = self.cfg.incoming_dir / "stale.partial"
        partial.write_bytes(b"partial")
        orphan = self.cfg.transcripts_dir / "orphan.txt"
        orphan.write_text("orphan", encoding="utf-8")
        old = time.time() - 10 * 86400
        os.utime(partial, (old, old))
        os.utime(orphan, (old, old))

        metadata = self.cfg.failed_dir / "apple-sidecar.mp3"
        metadata.write_bytes(self.APPLEDOUBLE_HEADER)
        source = self.audio("metadata-source.mp3")
        record_id = self.db.add_detected(source, source.stat(), at.source_fingerprint(source, source.stat()))
        self.db.update(record_id, status="ignored", local_path=str(metadata))
        note = Path(completed["note_path"])

        counts = at.maintain_state(self.cfg, self.db, self.logger)
        self.assertEqual(counts["chunks"], 1)
        self.assertEqual(counts["partials"], 1)
        self.assertEqual(counts["metadata"], 1)
        self.assertEqual(counts["orphans"], 1)
        self.assertFalse(chunks.exists())
        self.assertFalse(partial.exists())
        self.assertFalse(metadata.exists())
        self.assertFalse(orphan.exists())
        self.assertTrue(note.exists())

    def test_uninstall_removes_launch_agent_but_preserves_state(self) -> None:
        home = Path(self.temp.name) / "home"
        plist = home / "Library" / "LaunchAgents" / "local.auto-transcription.plist"
        plist.parent.mkdir(parents=True)
        plist.write_text("agent", encoding="utf-8")
        marker = self.cfg.state_root / "preserved"
        marker.write_text("keep", encoding="utf-8")
        with mock.patch.object(at.platform, "system", return_value="Darwin"), \
                mock.patch.object(at.Path, "home", return_value=home), \
                mock.patch.object(at.subprocess, "run", return_value=subprocess.CompletedProcess([], 0)) as run:
            self.assertTrue(at.uninstall_launchd(self.logger))
        self.assertFalse(plist.exists())
        self.assertTrue(marker.exists())
        self.assertIn("bootout", run.call_args.args[0])


if __name__ == "__main__":
    unittest.main()
