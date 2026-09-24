"""The OneDrive archive of lesson outputs (docs/archive.md).

Offline — OneDrive is faked with a temp folder shaped like the real one. What these pin is
what the archive exists for: the irreplaceable files (Gemini output, blind labels) always
land next to the right recording, audio never does, and nothing about archiving can take a
pipeline run down.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from jp_lesson_distill import archive, pipeline
from jp_lesson_distill.archive import ARCHIVE_DIR, lesson_date, work_root


@pytest.fixture
def onedrive(tmp_path, monkeypatch):
    """A fake 日本語/Soso tree with one recording, and a work/ to archive from."""
    soso = tmp_path / "OneDrive" / "日本語" / "Soso"
    lesson = soso / "Vol 3" / "L16"
    lesson.mkdir(parents=True)
    (lesson / "Soso_20260930_class_audio.m4a").write_bytes(b"recording")
    monkeypatch.setattr(archive, "SOSO", soso)
    work = tmp_path / "work"
    work.mkdir()
    return work, lesson


def touch(path: Path, text: str = "{}") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    return path


def archived(lesson: Path, date: str = "20260930") -> set[str]:
    root = lesson / ARCHIVE_DIR / date
    return {str(p.relative_to(root)) for p in root.rglob("*") if p.is_file()}


# --- blind labels: the thing that motivated all of this ------------------------

def test_labels_from_a_default_kit_are_archived(onedrive):
    work, lesson = onedrive
    touch(work / "20260930" / "labeling" / "items.jsonl", '{"truth": "student"}')
    archive.archive(work, log=lambda _: None)
    assert "20260930/labeling/items.jsonl" in archived(lesson)


def test_labels_from_a_kit_under_any_work_dir_are_archived(onedrive):
    """`distill label --work-dir work/<anything>` must not need anyone to list <anything>.
    The 8/17 kit was cut under work/hc9-2-verify/, and that name was once hard-coded."""
    work, lesson = onedrive
    touch(work / "some-new-experiment" / "20260930" / "labeling" / "items.jsonl")
    touch(work / "some-new-experiment" / "20260930" / "labeling" / "items.bak1.jsonl")
    archive.archive(work, log=lambda _: None)
    assert {"some-new-experiment/20260930/labeling/items.jsonl",
            "some-new-experiment/20260930/labeling/items.bak1.jsonl"} <= archived(lesson)


def test_the_archive_mirrors_work_so_restoring_is_a_plain_copy(onedrive):
    work, lesson = onedrive
    src = touch(work / "20260930" / "transcript_w02.json", '{"segments": []}')
    archive.archive(work, log=lambda _: None)
    assert (lesson / ARCHIVE_DIR / "20260930" / "20260930" / "transcript_w02.json").read_text() \
        == src.read_text()


# --- what is left out -----------------------------------------------------------

def test_audio_is_never_archived(onedrive):
    """Every audio file in work/ is re-cut from the recording; it is also most of the size."""
    work, lesson = onedrive
    touch(work / "20260930" / "transcript.json")
    for name in ["audio.m4a", "windows/w01.m4a", "labeling/clips/001.m4a", "clip.wav"]:
        touch(work / "20260930" / name, "audio")
    archive.archive(work, log=lambda _: None)
    assert archived(lesson) == {"20260930/transcript.json"}


def test_synthetic_smoke_test_lessons_are_not_archived(onedrive):
    work, _ = onedrive
    touch(work / "19990101" / "transcript.json")
    assert archive.plan(work) == {}


def test_a_lesson_with_no_folder_is_reported_not_guessed(onedrive):
    work, _ = onedrive
    touch(work / "20260101" / "transcript.json")
    done = archive.archive(work, log=lambda _: None)
    assert done.not_archived == ["20260101"] and done.copied == 0


# --- re-running -----------------------------------------------------------------

def test_a_second_run_copies_nothing(onedrive):
    work, _ = onedrive
    touch(work / "20260930" / "moments.json")
    assert archive.archive(work, log=lambda _: None).copied == 1
    again = archive.archive(work, log=lambda _: None)
    assert (again.copied, again.unchanged) == (0, 1)


def test_a_file_that_changed_since_is_copied_again(onedrive):
    """A re-labelled kit must replace the archived answers, not be skipped as 'present'."""
    import os

    work, lesson = onedrive
    items = touch(work / "20260930" / "labeling" / "items.jsonl", "old")
    archive.archive(work, log=lambda _: None)
    items.write_text("new")
    later = items.stat().st_mtime + 10
    os.utime(items, (later, later))
    archive.archive(work, log=lambda _: None)
    assert (lesson / ARCHIVE_DIR / "20260930" / "20260930/labeling/items.jsonl").read_text() == "new"


# --- where things go ------------------------------------------------------------

def test_a_date_folder_anywhere_in_the_path_names_the_lesson():
    assert lesson_date(Path("20260930/transcript.json")) == "20260930"
    assert lesson_date(Path("ref-windowed/w1/20260817/transcript.json")) == "20260817"
    assert lesson_date(Path("hc91-window/w01-MED-a.json")) == "20260817"
    assert lesson_date(Path("thinking-budget-attempt.patch")) is None


def test_a_work_dir_inside_work_is_archived_relative_to_work(tmp_path):
    """So `--work-dir work/hc9-2-verify` restores to work/hc9-2-verify/, not to work/."""
    assert work_root(tmp_path / "work" / "hc9-2-verify") == (tmp_path / "work").resolve()
    assert work_root(tmp_path / "elsewhere") == (tmp_path / "elsewhere").resolve()


def test_the_recording_s_own_folder_wins(onedrive, tmp_path):
    """The pipeline knows where the recording is; no lookup needed, none can go wrong."""
    work, lesson = onedrive
    other = archive.SOSO / "Vol 3" / "L17"
    other.mkdir(parents=True)
    touch(work / "20260930" / "moments.json")
    archive.archive(work, {"20260930"}, folder=other, log=lambda _: None)
    assert (other / ARCHIVE_DIR / "20260930" / "20260930" / "moments.json").exists()
    assert not (lesson / ARCHIVE_DIR).exists()


def test_a_recording_outside_onedrive_names_no_folder(onedrive, tmp_path):
    _, lesson = onedrive
    assert archive.lesson_folder(lesson / "Soso_20260930_class_audio.m4a") == lesson
    assert archive.lesson_folder(tmp_path / "Downloads" / "x.m4a") is None


# --- the pipeline hook ----------------------------------------------------------

def run_that_raises(monkeypatch, calls):
    monkeypatch.setattr(pipeline, "archive_lesson",
                        lambda date, work_dir, recording: calls.append(date))

    def boom(cfg):
        raise RuntimeError("window 3 failed")

    monkeypatch.setattr(pipeline, "_run", boom)


def test_a_run_that_fails_is_still_archived(monkeypatch, tmp_path):
    """Windows 1 and 2 are already paid for; they matter most when window 3 fails."""
    calls: list[str] = []
    run_that_raises(monkeypatch, calls)
    with pytest.raises(RuntimeError, match="window 3"):
        pipeline.run(pipeline.Config(recording=tmp_path / "r.m4a", date="20260930"))
    assert calls == ["20260930"]


def test_no_archive_skips_it(monkeypatch, tmp_path):
    calls: list[str] = []
    run_that_raises(monkeypatch, calls)
    with pytest.raises(RuntimeError):
        pipeline.run(pipeline.Config(recording=tmp_path / "r.m4a", date="20260930",
                                     archive=False))
    assert calls == []


def test_archiving_can_never_fail_a_run(monkeypatch, tmp_path, capsys):
    """OneDrive not mounted, another machine: warn and carry on."""
    def broken(*args, **kwargs):
        raise OSError("OneDrive is not mounted")

    monkeypatch.setattr(archive, "archive", broken)
    archive.archive_lesson("20260930", tmp_path / "work")
    out = capsys.readouterr().out
    assert "not archived" in out and "distill archive --date 20260930" in out


def test_smoke_test_runs_are_not_archived(monkeypatch, tmp_path):
    called = []
    monkeypatch.setattr(archive, "archive", lambda *a, **k: called.append(1))
    archive.archive_lesson("19990101", tmp_path / "work")
    assert called == []
