"""Stage orchestration. Each stage caches its output under work/<date>/ and is
skipped when the output file already exists — delete a stage file to redo it."""

from __future__ import annotations

import datetime as dt
import json
from dataclasses import dataclass
from pathlib import Path

from . import prompts
from .audio import clip_audio, duration_seconds, prep_audio
from .gemini import audio_part, generate, make_client, upload_audio
from .models import (
    Candidate,
    CandidateList,
    Moment,
    MomentsFile,
    Relisten,
    Transcript,
    fmt_ts,
    parse_ts,
)

CLIP_PAD = 15.0  # seconds of context on each side of a candidate


@dataclass
class Config:
    recording: Path
    date: str  # YYYYMMDD
    model: str = "gemini-pro-latest"
    work_dir: Path = Path("work")
    out_dir: Path = Path("/Users/stevensaito/Docs/Vault-GeneralNotes/_raw")
    skip_pass_b: bool = False
    max_moments: int = 40

    @property
    def lesson_date(self) -> str:
        return f"{self.date[:4]}-{self.date[4:6]}-{self.date[6:]}"


def _load(path: Path, schema):
    return schema.model_validate_json(path.read_text())


def _save(path: Path, obj) -> None:
    path.write_text(obj.model_dump_json(indent=2))


def run(cfg: Config) -> Path | None:
    work = cfg.work_dir / cfg.date
    work.mkdir(parents=True, exist_ok=True)
    client = None  # created only if a Gemini stage actually runs

    # prep
    audio = work / "audio.m4a"
    if audio.exists():
        print(f"[prep] cached: {audio}")
    else:
        print(f"[prep] extracting mono audio from {cfg.recording}")
        prep_audio(cfg.recording, audio)
    total = duration_seconds(audio)
    print(f"[prep] duration {fmt_ts(total)}")

    # pass A
    transcript_path = work / "transcript.json"
    if transcript_path.exists():
        print(f"[pass_a] cached: {transcript_path}")
    else:
        client = client or make_client()
        print(f"[pass_a] uploading audio and transcribing with {cfg.model} "
              "(an hour of audio takes 10-20 min; dots = transcript streaming in)")
        f = upload_audio(client, audio)
        transcript: Transcript = generate(client, cfg.model, [f, prompts.PASS_A], Transcript, progress=True)
        _save(transcript_path, transcript)
        print(f"[pass_a] {len(transcript.segments)} segments")
    transcript = _load(transcript_path, Transcript)

    # detect
    candidates_path = work / "candidates.json"
    if candidates_path.exists():
        print(f"[detect] cached: {candidates_path}")
    else:
        client = client or make_client()
        print("[detect] flagging candidate learning moments")
        prompt = prompts.DETECT.format(transcript=transcript.model_dump_json())
        cands: CandidateList = generate(client, cfg.model, [prompt], CandidateList, progress=True)
        _save(candidates_path, cands)
    cands = _load(candidates_path, CandidateList)
    ordered = sorted(cands.candidates, key=lambda c: c.priority, reverse=True)
    if len(ordered) > cfg.max_moments:
        print(f"[detect] {len(ordered)} candidates; keeping top {cfg.max_moments} by priority "
              f"(dropping {len(ordered) - cfg.max_moments} — raise --max-moments to include them)")
        ordered = ordered[: cfg.max_moments]
    else:
        print(f"[detect] {len(ordered)} candidates")

    if cfg.skip_pass_b:
        print("[pass_b] skipped (--skip-pass-b); no moments.json emitted")
        _emit_transcript(cfg, transcript)
        return None

    # pass B
    moments_path = work / "moments.json"
    if moments_path.exists():
        print(f"[pass_b] cached: {moments_path}")
    else:
        client = client or make_client()
        clips_dir = work / "clips"
        clips_dir.mkdir(exist_ok=True)
        moments: list[Moment] = []
        for i, cand in enumerate(ordered, start=1):
            t0, t1 = parse_ts(cand.t_start), parse_ts(cand.t_end)
            clip = clips_dir / f"m{i:02d}.m4a"
            if not clip.exists():
                clip_audio(audio, clip, t0 - CLIP_PAD, min(total, t1 + CLIP_PAD))
            print(f"[pass_b] {i}/{len(ordered)} re-listening {cand.t_start}-{cand.t_end} ({cand.type})")
            r: Relisten = _relisten(client, cfg.model, clip, cand)
            if not r.keep:
                print(f"[pass_b]   rejected: {r.explanation}")
                continue
            moments.append(Moment(
                id=f"m{i:02d}", t_start=t0, t_end=t1, type=r.type,
                student_verbatim=r.student_verbatim, teacher_correction=r.teacher_correction,
                explanation=r.explanation, confidence=r.confidence,
            ))
        out = MomentsFile(
            lesson_date=cfg.lesson_date,
            source_recording=str(cfg.recording),
            model=cfg.model,
            generated_at=dt.datetime.now().astimezone().isoformat(timespec="seconds"),
            moments=moments,
        )
        _save(moments_path, out)
    out = _load(moments_path, MomentsFile)
    print(f"[pass_b] {len(out.moments)} moments confirmed")

    # emit
    cfg.out_dir.mkdir(parents=True, exist_ok=True)
    dest = cfg.out_dir / f"{cfg.date}-moments.json"
    dest.write_text(out.model_dump_json(indent=2))
    _emit_transcript(cfg, transcript)
    print(f"[emit] {dest}")
    return dest


def _relisten(client, model: str, clip: Path, cand: Candidate) -> Relisten:
    prompt = prompts.PASS_B.format(type=cand.type, rationale=cand.rationale, excerpt=cand.excerpt)
    return generate(client, model, [audio_part(clip), prompt], Relisten)


def _emit_transcript(cfg: Config, transcript: Transcript) -> None:
    lines = [f"# 授業の文字起こし {cfg.lesson_date}", "",
             f"Diarized transcript (verbatim — student errors preserved). Source: `{cfg.recording}`", ""]
    for seg in transcript.segments:
        name = "Soso先生" if seg.speaker == "teacher" else "Steven"
        flag = " ⚠️" if seg.uncertain else ""
        lines.append(f"- `[{seg.start}]` **{name}:**{flag} {seg.text}")
    path = cfg.out_dir / f"{cfg.date}-transcript.md"
    cfg.out_dir.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")
    print(f"[emit] {path}")
