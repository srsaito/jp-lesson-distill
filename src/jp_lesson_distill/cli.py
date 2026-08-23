from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

from .pipeline import Config, run
from .validate import build, dump, play, report, reset


def main() -> None:
    parser = argparse.ArgumentParser(prog="distill", description="Distill a Japanese lesson recording into learning moments")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("run", help="run the full pipeline on a recording")
    p.add_argument("recording", type=Path, help="path to the recording (audio or video; OneDrive sync path is fine)")
    p.add_argument("--date", required=True, help="lesson date, YYYYMMDD")
    p.add_argument("--model", default="gemini-pro-latest",
                   help="Gemini model (default: the -latest Pro alias; pin a dated model for reproducibility)")
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

    args = parser.parse_args()
    if not re.fullmatch(r"\d{8}", args.date):
        parser.error("--date must be YYYYMMDD")

    try:
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
            return

        if not args.recording.exists():
            parser.error(f"recording not found: {args.recording}")
        run(Config(
            recording=args.recording, date=args.date, model=args.model,
            work_dir=args.work_dir, out_dir=args.out_dir,
            skip_pass_b=args.skip_pass_b, max_moments=args.max_moments,
            window_minutes=args.window_minutes, overlap_seconds=args.overlap_seconds,
            window_attempts=args.window_attempts,
        ))
    except KeyboardInterrupt:
        sys.exit(130)


if __name__ == "__main__":
    main()
