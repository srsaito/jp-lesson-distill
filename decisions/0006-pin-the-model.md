---
status: accepted
date: 2026-09-17
---

# 0006 — The default model is a pinned id, not a `-latest` alias

Supersedes the model-default clause of [[0002-gemini-pass-a-and-b]]. Everything else in 0002 —
Gemini for both passes, the prompts, the ±15 s clips — still stands.

## Context
ADR-0002 chose the **`gemini-pro-latest` alias** as the default, for a concrete reason: a pinned
`gemini-2.5-pro` default had already rotted once mid-build when 2.5 Pro was retired for new API
users. The alias was the pragmatic choice for a personal pipeline nobody maintains full-time.

Since then the alias moved — and the cost of that showed up as a measurement we could not
interpret. Two Pass A runs over the **same** 2026-08-17 lesson, three days apart (2026-08-18 and
2026-08-21), agree on the words and disagree on **who spoke** for a whole stretch of the lesson:
145 disagreements over 417 matched lines, concentrated in 08:00–20:00 where agreement is 35%,
which is *below chance* and therefore an inversion rather than noise (`jld-dli`). Blind ground
truth later showed the windowed run was right 14 times out of 16 on the contested lines and the
other run 0 — so one of the two runs was substantially worse, and the leading explanation we
could not rule out was simply **that the two runs were different models**.

Nothing on disk could answer that question. `moments.json` records `model: "gemini-pro-latest"`
for both, which is not the name of a model. A probe on 2026-09-17 resolved the alias to
`gemini-3.1-pro-preview`; the same alias served Gemini 2.5 Pro when this pipeline was written.

## Drivers
- A transcript is a measurement. A measurement whose instrument is unrecorded cannot be compared
  with another, and this project's whole quality story — 89% diarization, coverage gates,
  calibrated loop thresholds — is built out of comparisons between runs.
- Alias rot is real (2.5 Pro, above), so the pin must be revisitable cheaply, not permanent.
- Card generation downstream reads `moments.json`; the field is already there, it just needs to
  mean something.

## Options
1. **Pin the default to an explicit id; keep `--model` for anything else.**
2. Keep the alias and record the resolved version somewhere. Honest, but it documents drift
   instead of preventing it: runs still are not comparable, they are merely known not to be.
3. Pin, and have the code refuse an alias outright. Removes the escape hatch on the day the
   pinned id is retired — which is precisely the failure ADR-0002 was written around.

## Decision
**Option 1, plus the recording half of option 2.** `DEFAULT_MODEL = "gemini-3.1-pro-preview"` in
`gemini.py`, and it is the only Gemini model id written down anywhere in the source — the CLI
and `Config` both import it, so a default cannot drift between them (pinned by a test).

Every call also reports the version the API actually served, and says so **only when it differs
from the id that was asked for**. With the pin that line never prints; with `--model
gemini-pro-latest` it names the resolution. Drift becomes visible rather than inferred.

`--model` still takes anything, including the aliases, so a retired pin is a one-flag problem.

## Consequences
- `moments.json` now names an actual model, so two lessons' outputs are comparable and a future
  "did the model change?" question is answerable from the artifacts.
- The pinned id is a **preview** id (`gemini-3.1-pro-preview`, version `3.1-pro-preview-01-2026`)
  because there is no GA 3.x Pro on this API today. Preview ids are exactly the kind that get
  retired, so this ADR expects to be revisited; the upgrade is deliberate, which is the point.
- Upgrading now costs one line and one re-measurement of whatever a comparison depends on,
  rather than happening silently between two lessons.
- It does **not** retroactively explain the 2026-08 diarization divergence. The hypothesis is
  still untested — the two runs are gone — but the next such divergence will have the model
  written down on both sides.
