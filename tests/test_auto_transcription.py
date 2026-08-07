from __future__ import annotations

import contextlib
import datetime as dt
import json
import logging
import os
from pathlib import Path
import subprocess
import sqlite3
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


class CountingTranscriber:
    backend = "mock"

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

    def hybrid_retry(self, path, result):
        return result

    def version(self):
        return "test"

    def release_models(self):
        self.released = True


class AutoTranscriptionTests(unittest.TestCase):
    def test_default_profile_is_low_memory_and_fast(self) -> None:
        cfg = at.Config()
        self.assertEqual(cfg.model, "small")
        self.assertEqual(cfg.retry_model, "small")
        self.assertFalse(cfg.hybrid_retry)
        self.assertEqual(cfg.nice_level, 20)
        self.assertTrue(cfg.unmount_on_success)

    def test_process_priority_is_lowered_to_configured_nice_level(self) -> None:
        with mock.patch.object(at.os, "getpriority", side_effect=[0, 20]), \
                mock.patch.object(at.os, "setpriority") as setpriority:
            self.assertEqual(at.apply_process_priority(20), 20)
        setpriority.assert_called_once_with(at.os.PRIO_PROCESS, 0, 20)

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

    def only_nonlegacy_row(self):
        return self.db.conn.execute("SELECT * FROM recordings WHERE source_path <> 'legacy' ORDER BY id DESC LIMIT 1").fetchone()

    def test_end_to_end_and_idempotency(self) -> None:
        self.audio("VOICE 001.mp3")
        imported = self.pipeline.import_new()
        self.assertEqual(imported["ready"], 1)
        processed = self.pipeline.process_pending()
        self.assertEqual(processed, {"completed": 1, "failed": 0})
        row = self.only_nonlegacy_row()
        self.assertEqual(row["status"], "completed")
        self.assertTrue(Path(row["transcript_path"]).exists())
        self.assertTrue(Path(row["summary_path"]).exists())
        self.assertTrue(Path(row["note_path"]).exists())
        self.assertEqual(self.pipeline.import_new()["found"], 0)
        self.assertEqual(self.pipeline.process_pending(), {"completed": 0, "failed": 0})

    def test_run_notifies_new_count_and_final_summary_but_empty_poll_is_quiet(self) -> None:
        root = Path(self.temp.name) / "notify-run"
        recorder = root / "recorder" / "REC_FILE"
        recorder.mkdir(parents=True)
        (recorder / "notify.mp3").write_bytes(b"audio")
        config = root / "config.toml"
        config.write_text(
            f'''[paths]\nmount_point = "{root / "recorder"}"\nvault = "{root / "vault"}"\nstate_root = "{root / "state"}"\n'''
            '''[import]\nstable_wait_seconds = 0\n'''
            '''[transcription]\nbackend = "mock"\nmodel = "tiny"\nretry_model = "tiny"\nhybrid_retry = false\nvad_enabled = false\n'''
            '''[summarization]\nenabled = false\nmodel = "mock"\n'''
            '''[behavior]\nnotify = true\nprevent_sleep = false\nunmount_on_success = false\n''',
            encoding="utf-8",
        )
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
        row = self.only_nonlegacy_row()
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
        row = self.only_nonlegacy_row()
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
        row = self.only_nonlegacy_row()

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
        row = self.only_nonlegacy_row()
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
        row = self.only_nonlegacy_row()
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
        row = self.only_nonlegacy_row()
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
        row = self.only_nonlegacy_row()
        attachments = list((self.cfg.vault / self.cfg.attachments_folder).glob("*.mp3"))
        self.assertEqual(len(attachments), 1)
        self.assertEqual(at.sha256_file(attachments[0]), row["sha256"])
        note = Path(row["note_path"]).read_text(encoding="utf-8")
        self.assertIn("![[Recordings/Audio/", note)

    def test_cleanup_requires_age_and_preserves_notes(self) -> None:
        self.audio("cleanup.mp3")
        self.pipeline.import_new()
        self.pipeline.process_pending()
        row = self.only_nonlegacy_row()
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
        self.assertIn("/tmp/a&amp;b.py", plist)
        self.assertIn("<integer>60</integer>", plist)
        self.assertIn("<key>StartOnMount</key><true/>", plist)
        self.assertIn("<key>RunAtLoad</key><true/>", plist)
        self.assertIn("<key>Nice</key><integer>20</integer>", plist)

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

    def test_legacy_hash_migration(self) -> None:
        legacy = Path(self.temp.name) / ".processed_hashes"
        digest = "a" * 64
        legacy.write_text(digest + "\nnot-a-hash\n", encoding="utf-8")
        self.assertEqual(self.db.migrate_legacy_hashes(legacy, self.logger), 1)
        self.assertEqual(self.db.migrate_legacy_hashes(legacy, self.logger), 0)

    def test_existing_database_adds_transcription_generation_column(self) -> None:
        path = Path(self.temp.name) / "old-state.sqlite3"
        connection = sqlite3.connect(path)
        connection.execute(
            "CREATE TABLE recordings (id INTEGER PRIMARY KEY, status TEXT, sha256 TEXT, "
            "generated_title TEXT, summary_model TEXT)"
        )
        connection.commit()
        connection.close()
        with at.StateDB(path) as migrated:
            columns = {str(row[1]) for row in migrated.conn.execute("PRAGMA table_info(recordings)")}
        self.assertIn("transcription_generation", columns)

    def test_cli_run_is_idempotent(self) -> None:
        root = Path(self.temp.name) / "cli"
        recorder = root / "recorder" / "REC_FILE"
        vault = root / "vault"
        state = root / "state"
        recorder.mkdir(parents=True)
        vault.mkdir(parents=True)
        (recorder / "CLI TEST.mp3").write_bytes(b"cli-audio")
        config = root / "config.toml"
        config.write_text(
            f'''[paths]\nmount_point = "{root / "recorder"}"\nvault = "{vault}"\nstate_root = "{state}"\n'''
            '''[import]\nstable_wait_seconds = 0\n'''
            '''[transcription]\nbackend = "mock"\nmodel = "tiny"\nretry_model = "tiny"\nhybrid_retry = false\nvad_enabled = false\n'''
            '''[summarization]\nenabled = false\nmodel = "mock"\n'''
            '''[behavior]\nnotify = false\nprevent_sleep = false\n''',
            encoding="utf-8",
        )
        with mock.patch.object(at, "apply_process_priority", return_value=20), \
                mock.patch.object(at, "unmount_recorder", return_value=False) as unmount:
            self.assertEqual(at.main(["--config", str(config), "--quiet", "run"]), 0)
            self.assertEqual(at.main(["--config", str(config), "--quiet", "run"]), 0)
        self.assertEqual(unmount.call_count, 2)
        with at.StateDB(state / "state.sqlite3") as cli_db:
            self.assertEqual(cli_db.counts(), {"completed": 1})

    def test_compatibility_launcher_handles_empty_arguments_under_nounset(self) -> None:
        root = Path(__file__).resolve().parents[1]
        environment = dict(os.environ, AUTO_TRANSCRIPTION_PYTHON="/bin/echo")
        completed = subprocess.run(
            ["/bin/bash", str(root / "transcribe_and_summarize.sh")],
            text=True, capture_output=True, env=environment, check=True,
        )
        self.assertEqual(completed.stdout.strip(), f"{root / 'auto_transcription.py'} run")


if __name__ == "__main__":
    unittest.main()
