from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

from .pipeline import Config, run


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

    args = parser.parse_args()
    if not re.fullmatch(r"\d{8}", args.date):
        parser.error("--date must be YYYYMMDD")
    if not args.recording.exists():
        parser.error(f"recording not found: {args.recording}")

    try:
        run(Config(
            recording=args.recording, date=args.date, model=args.model,
            work_dir=args.work_dir, out_dir=args.out_dir,
            skip_pass_b=args.skip_pass_b, max_moments=args.max_moments,
            window_minutes=args.window_minutes, overlap_seconds=args.overlap_seconds,
        ))
    except KeyboardInterrupt:
        sys.exit(130)


if __name__ == "__main__":
    main()
