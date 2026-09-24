from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

from .archive import archive, archive_lesson, main_work_dir
from .gemini import (
    DEFAULT_MODEL,
    PASS_A_THINKING_LEVEL,
    STREAM_IDLE_TIMEOUT_S,
    THINKING_LEVELS,
)
from .pipeline import Config, run
from .validate import build, dump, play, report, reset


def main() -> None:
    parser = argparse.ArgumentParser(prog="distill", description="Distill a Japanese lesson recording into learning moments")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("run", help="run the full pipeline on a recording")
    p.add_argument("recording", type=Path, help="path to the recording (audio or video; OneDrive sync path is fine)")
    p.add_argument("--date", required=True, help="lesson date, YYYYMMDD")
    p.add_argument("--model", default=DEFAULT_MODEL,
                   help=f"Gemini model (default: {DEFAULT_MODEL}, pinned — the -latest aliases "
                        "move under you, so two runs days apart are not comparable)")
    p.add_argument("--out-dir", type=Path, default=Path("/Users/stevensaito/Docs/Vault-GeneralNotes/_raw"))
    p.add_argument("--work-dir", type=Path, default=Path("work"))
    p.add_argument("--skip-pass-b", action="store_true", help="stop after moment detection (cheap dry run)")
    p.add_argument("--max-moments", type=int, default=40)
    p.add_argument("--window-minutes", type=float, default=20.0,
                   help="length of each Pass A transcription window (0 = one call for the whole "
                        "recording; only safe under ~20 min)")
    p.add_argument("--overlap-seconds", type=float, default=30.0,
                   help="overlap between consecutive windows; the merge splits it at its midpoint")
    p.add_argument("--window-attempts", type=int, default=2,
                   help="how many times to transcribe a window that fails the sanity gates "
                        "(1 = never retry; the least-bad attempt is kept either way)")
    p.add_argument("--pass-a-thinking", default=PASS_A_THINKING_LEVEL.lower(),
                   choices=[*THINKING_LEVELS, "model-default"],
                   help="how hard the model may think while TRANSCRIBING (default: "
                        f"{PASS_A_THINKING_LEVEL.lower()}, measured — 'low' is faster and gets "
                        "the SPEAKER wrong, see ADR-0007). Detect and Pass B always use the "
                        "model default")
    p.add_argument("--stream-timeout", type=float, default=STREAM_IDLE_TIMEOUT_S,
                   help="seconds of silence before a streaming call is abandoned and re-rolled; "
                        "this is a gap between chunks, not a budget for the whole response "
                        f"(default: {STREAM_IDLE_TIMEOUT_S:.0f})")
    p.add_argument("--no-archive", action="store_true",
                   help="don't copy this lesson's outputs to OneDrive when the run ends "
                        "(for experiments; see docs/archive.md)")

    # distill label — build a blind listening kit for diarization ground truth (jld-dli)
    lab = sub.add_parser("label", help="sample lines and cut blind clips to label by ear")
    lab.add_argument("recording", type=Path, nargs="?",
                     help="the recording, used only to find Z_kaiwa_scripts_*.md beside it")
    lab.add_argument("--date", required=True, help="lesson date, YYYYMMDD")
    lab.add_argument("--work-dir", type=Path, default=Path("work"))
    lab.add_argument("--compare", type=Path,
                     help="another run's transcript.json; lines the two disagree on become "
                          "the contested stratum")
    lab.add_argument("--random", type=int, default=20, dest="n_random")
    lab.add_argument("--contested", type=int, default=20, dest="n_contested")
    lab.add_argument("--scripted", type=int, default=10, dest="n_scripted")
    lab.add_argument("--script", type=Path,
                     help="dialogue script to match against (default: Z_kaiwa_scripts_*.md beside "
                          "the recording); any markdown table or plain lines will do")
    lab.add_argument("--seed", type=int, default=0, help="sampling seed (same seed = same kit)")
    lab.add_argument("--reset", action="store_true",
                     help="clear every answer and relabel from scratch (the old answers are "
                          "backed up next to the kit)")
    lab.add_argument("--play", action="store_true",
                     help="start listening straight away (resumable; also `distill label --play` alone)")

    sc = sub.add_parser("score", help="score Pass A against the labels you recorded")
    sc.add_argument("--date", required=True, help="lesson date, YYYYMMDD")
    sc.add_argument("--work-dir", type=Path, default=Path("work"))
    sc.add_argument("--against", type=Path,
                    help="a repaired transcript.json to compare against Pass A, paired (McNemar)")
    sc.add_argument("--dump", action="store_true", help="print the labelled truth as JSON")

    # distill archive — normally automatic; this is for catching up by hand
    ar = sub.add_parser("archive", help="copy lesson outputs to OneDrive next to each recording "
                                        "(runs automatically after `distill run`)")
    ar.add_argument("--date", nargs="*", help="only these lesson dates, YYYYMMDD (default: all)")
    ar.add_argument("--work", type=Path, help="the work/ to archive (default: the main checkout's)")
    ar.add_argument("--dry-run", action="store_true", help="say what would be copied, copy nothing")

    args = parser.parse_args()
    dates = (args.date or []) if args.command == "archive" else [args.date]
    if not all(re.fullmatch(r"\d{8}", d) for d in dates):
        parser.error("--date must be YYYYMMDD")

    try:
        if args.command == "archive":
            done = archive(args.work or main_work_dir(), set(dates) or None, dry_run=args.dry_run)
            print(f"{'would copy' if args.dry_run else 'copied'} {done.copied} files; "
                  f"{done.unchanged} already archived and up to date")
            if done.not_archived:
                sys.exit(f"not archived (no lesson folder in OneDrive): {', '.join(done.not_archived)}")
            return
        if args.command == "score":
            dump(args.date, args.work_dir) if args.dump else report(args.date, args.work_dir, args.against)
            return
        if args.command == "label":
            if args.recording is not None and not args.recording.exists():
                parser.error(f"recording not found: {args.recording}")
            if args.reset:
                reset(args.date, args.work_dir)
            elif args.recording is not None or not args.play:
                build(args.recording, args.date, args.work_dir, args.compare,
                      args.n_random, args.n_contested, args.n_scripted, args.seed,
                      script_path=args.script)
            if args.play:
                play(args.date, args.work_dir)
                # The answers are the irreplaceable part of a kit; don't leave them waiting
                # for the next pipeline run to be copied.
                archive_lesson(args.date, args.work_dir, args.recording)
            return

        if not args.recording.exists():
            parser.error(f"recording not found: {args.recording}")
        run(Config(
            recording=args.recording, date=args.date, model=args.model,
            work_dir=args.work_dir, out_dir=args.out_dir,
            skip_pass_b=args.skip_pass_b, max_moments=args.max_moments,
            window_minutes=args.window_minutes, overlap_seconds=args.overlap_seconds,
            window_attempts=args.window_attempts, stream_timeout=args.stream_timeout,
            pass_a_thinking=(None if args.pass_a_thinking == "model-default"
                             else args.pass_a_thinking.upper()),
            archive=not args.no_archive,
        ))
    except KeyboardInterrupt:
        sys.exit(130)


if __name__ == "__main__":
    main()
