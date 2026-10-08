---
status: accepted
date: 2026-10-08
---

# 0009 — Moments carry the board lines near them; the board stays out of the prompts

## Context
ADR-0008 gave every lesson a board timeline: what Soso先生 typed and wrote on the shared screen, and when. The obvious next step was to show it to detect and Pass B, on the theory that a typed correction is evidence of an error, and that a correction typed but never spoken would otherwise produce no moment (`jld-02g`).

That was built and measured on the two lessons with a board and no レジュメ, 2026-09-28 and 2026-10-05: detect with the board notes appended, against the original run and against a second no-board run of detect, which shows how much detect varies by itself.

| | 9/28 | 10/5 |
|---|---|---|
| candidates: original / no-board re-run / with board | 17 / 14 / 16 | 19 / 24 / 18 |
| original candidates found again by the no-board re-run | 14 of 17 | 18 of 19 |
| original candidates found again with the board | 17 of 17 | 16 of 19 |
| candidates only the board run found | 0 | 1 (also in the no-board re-run) |
| candidates whose rationale cites the board | 11 of 16 | 1 of 18 |

On 9/28 Pass B also ran with the board lines in its prompt: 16 moments against 17, and the teacher's correction came out the same on every moment the two runs share.

## What it showed
- **The board found no moment the audio had missed.** He says aloud what he types (あげる/くれる, つけ直す, 電源, 二文目, コンロ), so the transcript already carries it.
- **The differences between runs are the size of detect's own run-to-run variation**, in both directions. Two lessons cannot separate a small effect from that.
- **Pass B's corrections did not change.**
- What the board does add is the *written* form, with reading and gloss: 「できもの bump」, 「内視鏡 ないしきょう endoscope」, 「凍らせる こおらせる to freeze」. On 9/28, 12 of 16 moments have such lines within reach.

## Decision
- **`Moment` gains `board: list[BoardNote]`** (`t` in seconds, `text`), default empty: the board lines first seen from 30 s before the exchange to 90 s after it. Typing trails the speech it answers, hence the lopsided window. It is attached by time alone, after the moments exist, with no model involved.
- **The board is not shown to detect or Pass B.** The prompts are as they were. A test pins that, and says what would justify changing it.
- Order of a run: the moments and transcript are emitted first, then the screen is read (10–20 minutes), then `moments.json` is written once more with the board lines. A run that loses its board (no video, a failure) leaves the first, complete, file in place.
- Text comes from Gemini's reading of the slide where it matches the line Vision timed, so a moment gets お義母さん, not Vision's お茶母さん.

This extends the ADR-0003 contract with an optional field; a `moments.json` without it remains valid, and old files load unchanged.

## Consequences
- The vault skill has the teacher's spelling, reading and gloss next to each moment without joining two files by timestamp itself.
- The window is generous, so a moment can pick up a neighbour's vocabulary line or a slide label (「16-12」). The consumer is Claude, which can ignore those; a wrong line dropped silently would be worse than an extra one.
- Lines typed in one burst share a time (ADR-0008), which is another reason the window is wide.
- **Reopen this if** a lesson turns up where he corrects in writing without saying it — a listening-heavy lesson, or one where he types while Steven keeps talking. The experiment is commit `c46e5cf` on the branch that introduced this ADR; its prompts are there to restore.
- Found on the way: Pass B keeps no per-clip cache, so one stalled clip late in a run throws away every re-listen before it (`jld-8rb`).
