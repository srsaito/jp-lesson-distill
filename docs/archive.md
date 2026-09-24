# Where the lesson data lives (and the archive in OneDrive)

The repo is public, so it never holds lesson content. Three places do, and they have
different jobs:

| place | holds | backed up? |
|---|---|---|
| `日本語/Soso/Vol N/L##/Soso_<YYYYMMDD>_class*.m4a` (OneDrive) | the recordings — canonical (ADR-0004) | yes, OneDrive |
| `work/` in the **main** checkout (`~/Dev/jp-lesson-distill/work/`, gitignored) | everything the pipeline derives: audio cuts, Gemini output, listening kits | **no** — one laptop |
| `日本語/Soso/Vol N/L##/distill-archive/<YYYYMMDD>/` (OneDrive) | the part of `work/` that cannot be regenerated | yes, OneDrive |
| `tests/fixtures/` (this repo) | text-free summaries of measurements — labels, timings, counts | yes, git |

## What the archive keeps, and why only that

`scripts/archive_work.py` copies every **non-audio** file for a lesson into the archive,
next to that lesson's recording. Audio is left out on purpose: every audio file in `work/`
(`audio.m4a`, `windows/`, `clips/`) is cut from the recording and can be cut again.

What it keeps cannot be regenerated identically:

- **Gemini output** — Pass A windows (`transcript_w<NN>.json`) and the merged
  `transcript.json`, `candidates.json`, `moments.json`, failed attempts
  (`*.attempt<N>.json`) and partials (`pass_a_partial_*.json`), and the Gemini 3.5
  Transcribe trial responses (`transcribe35/`). Gemini is nondeterministic and billed: a
  re-run is a *different* transcript, not the same one again — and several ADRs rest on
  comparing specific runs.
- **Blind diarization labels** — `hc9-2-verify/20260817/labeling/items.jsonl` holds the 41
  hand labels behind the 89% figure (`jld-dli`), *with* the utterance text. The committed
  `tests/fixtures/diarization_truth_20260817.json` is the text-free version of the same labels.
- The measurement runs that ADRs cite: `hc9-2-verify/` (the verified windowed run of 8/17),
  `ref-windowed/` (the hand-staged reference that ground truth scored 0 for 16), and
  `hc91-window/` (the thinking-level arms behind ADR-0007).

## Layout

Each lesson date folder **mirrors `work/`**, so paths are identical on both sides:

```
日本語/Soso/Vol 3/L13/distill-archive/
    README.md                         ← what this is, for someone browsing OneDrive
    20260817/
        20260817/transcript.json       = work/20260817/transcript.json
        hc9-2-verify/20260817/labeling/items.jsonl
        hc9-2-verify/20260817/transcribe35/w01.raw.json
        hc91-window/w01-MED-a.json
        ref-windowed/w1/20260817/transcript.json
```

## Using it

```bash
uv run python scripts/archive_work.py --dry-run          # what would be copied
uv run python scripts/archive_work.py                    # archive every lesson in work/
uv run python scripts/archive_work.py --date 20260930    # just one lesson
```

Run it after each lesson's pipeline run. It is safe to repeat: a file is copied only when
it is missing from the archive or newer in `work/`. It works from a worktree too — it
archives the **main** checkout's `work/`, which is where the data is.

**To restore a lesson** (new machine, deleted `work/`), copy the date folder back over
`work/` — the pipeline then treats every archived stage as cached and skips it:

```bash
rsync -a "$HOME/Library/CloudStorage/OneDrive-個人用/日本語/Soso/Vol 3/L13/distill-archive/20260817/" work/
```

Then re-cut the audio the stages need by running the pipeline on the recording as usual.

## Rules

- The archive holds what was said in the lessons. It stays in private OneDrive and never
  goes into this repo, whose fixtures are text-free by design.
- A lesson with no recording in OneDrive is matched by its vault outputs instead
  (e.g. `Vol 3/L7/20260312-moments.json`). If neither exists the script says
  `NOT archived` — it never guesses a folder.
- Synthetic smoke-test lessons (`work/1999xxxx`) are not archived.
