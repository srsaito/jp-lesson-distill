# jp-lesson-distill

Distill Japanese lesson recordings (1-on-1 with Soso先生) into learning moments — spoken mistakes, verbal corrections, hesitations, verbally-introduced material — as input to flashcard generation. See `CLAUDE.md` for the charter and `docs/architecture.md` for the design.

## Setup

```sh
uv sync
export GEMINI_API_KEY=...   # already in Steven's shell env
```

ffmpeg is bundled via the `imageio-ffmpeg` wheel — no system install needed (a system ffmpeg is used if present).

## Usage

```sh
# full pipeline on a recording (any local path, incl. OneDrive sync folder)
uv run distill run ~/OneDrive/…/lesson.mov --date 20260707

# options
#   --model gemini-flash-latest   cheaper model (default: gemini-pro-latest)
#   --out-dir PATH             emit destination (default: General vault _raw/)
#   --work-dir PATH            stage cache (default: ./work)
#   --skip-pass-b              stop after detection (cheap dry run)
#   --max-moments N            cap Pass B clips (default 40)
```

Stages cache under `work/<date>/` (`audio.m4a`, `transcript.json`, `candidates.json`, `moments.json`, `clips/`). Re-runs skip completed stages; delete a stage file to redo it — e.g. delete `candidates.json` after editing the detection prompt without re-transcribing the hour.

Output lands in the General vault `_raw/` as `YYYYMMDD-moments.json` + `YYYYMMDD-transcript.md`, then `/distill-jp-lesson` (a General-vault Claude skill) fuses them with the lesson note and generates Anki cards via FlashGen.

## Verification

`scripts/make_test_lesson.sh` synthesizes a mini-lesson with macOS `say` (Kyoko/Samantha voices) containing a planted student error (「学校で行きます」, particle mistake), so the full pipeline can be smoke-tested end-to-end without a real recording:

```sh
./scripts/make_test_lesson.sh          # writes work/test/test_lesson.m4a
uv run distill run work/test/test_lesson.m4a --date 19990101 --out-dir work/test/out
```

Expected: the planted mistake survives verbatim in a `correction` moment (で → に), and `work/test/out/19990101-moments.json` validates against the schema.
