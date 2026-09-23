---
status: accepted
date: 2026-09-17
---

# 0007 — Pass A sets an explicit thinking level, and it is `medium`, not `low`

## Context
`jld-hc9.1` proposed sending `thinking_config=ThinkingConfig(thinking_level='low')` on Pass A,
with the reasoning that **transcription is mechanical**: hear the words, write them down, and
thinking tokens come out of the same 65,536-token budget as the answer (a 59-minute call once
spent 62,911 tokens thinking and had 2,563 left for the transcript, ADR/issue `jld-hc9.3.4`).
Detect and Pass B would keep the model default, since they genuinely reason.

Measuring it changed the answer. Two things came out of one afternoon on 2026-09-17, both on
`gemini-3.1-pro-preview` (the pin, [[0006-pin-the-model]]), same prompt, same temperature 0.2,
and — for the real numbers — the same window 1 (00:00–20:00) of the 2026-08-17 lesson that
blind ground truth exists for.

**A level is not a small dial, and the model's own default is not where the issue assumed.**
On a 20-minute window the model default spends **11,507 thinking tokens** before answering —
first byte at 71 s, complete at 109 s. `medium` spends 9,262 (first byte 59 s). `low` spends so
little the API declines to report a count at all, and starts streaming in **4 s**. So `medium`
is roughly the status quo minus a little, while `low` is a genuine departure from every run this
project has ever measured, including the one that scored 89% against blind ground truth.

**`low` measurably damages diarization.** The address-form canary (`jld-86b`) is this project's
only check on *who spoke*, and on window 1 it reads:

| level | stream | first byte | thinking | answer | segments | coverage | address forms |
|---|---|---|---|---|---|---|---|
| `low` | 53 s | 4.4 s | unreported | 8,500 | 186 | 100% | **3/5** |
| `low` | 50 s | 4.1 s | unreported | 7,483 | 158 | 100% | **2/5** |
| `low` | 52 s | 3.6 s | unreported | 8,070 | 174 | 100% | **1/5** |
| `medium` | 98 s | 59.2 s | 9,262 | 7,075 | 145 | 100% | **4/5** |
| model default | 109 s | 71.1 s | 11,507 | — | — | — | (patient probe, not scored) |
| *2026-08-20 reference run* | — | — | — | — | 192 | 100% | *5/5* |

The `low` misses are not subtle and not a segmentation artifact: they are short, cleanly split
lines that can only be the student — 「そそ先生は、あ、韓国に行ったことがありますか？」,
「私はもう予定はありませんけど、あー、妻が予定があります。」 — labelled `teacher`. Six of
fifteen cue lines wrong across three runs, against 5/5 for the run that later scored 89% against
blind ground truth. The one `medium` miss is a truncated fragment (「に行って、で、奥さんは仕事して、」),
which is the merged-turn defect `jld-beb`, not an attribution error.

Everything else was equal, which is what makes the canary readable: coverage 100% at every
level, density inside the known-good 4.6–9.6/min band, and no sign of the fluency drift the
prompt exists to prevent (student filler rate 13.0–18.7 per 1,000 characters against the
reference's 13.7, and student character counts within 6%).

## Drivers
- Pass A has to finish unattended on a 20-minute window, and its cost has to be *legible* —
  a level the pipeline sets is a level the pipeline can log and compare.
- Diarization at 89% is the most expensively established number in this repo (41 blind hand
  labels). A default that quietly spends it is worse than a slower one.
- Thinking competes with the transcript for one output budget, so the level cannot simply be
  raised without watching what it costs.
- The level must be measurable per call, or a regression like this one is invisible.

## Options
1. `low`, as the issue proposed. Fastest, and it is the arm the canary says is wrong.
2. **`medium`.** Roughly 50 s more per window, 9,262 thinking tokens (14% of the output budget),
   and the canary reads as consistent.
3. No level (today's behaviour). Not disqualified on quality — it is what produced the
   reference run — but it is the most expensive of the three and it is unmeasurable: the
   pipeline cannot report a level it did not set, and cannot notice when the model's default
   moves. It is the status quo this issue exists to replace.
4. `minimal`. Not an option at all — this model answers it with
   `400 Thinking level MINIMAL is not supported for this model`.
5. `high`. Not measured; there is no evidence a defect above `medium` needs it, and it buys more
   of the thing that competes with the answer.

## Decision
**Option 2.** `PASS_A_THINKING_LEVEL = "MEDIUM"`, sent as `thinking_level` and never alongside
`thinking_budget` (which is a 400 together and a silent no-op on 3.x). `--pass-a-thinking`
exposes `low|medium|high|model-default` for experiments; detect and Pass B send no level.

Every call logs its thinking/answer token split, and warns when thinking exceeds 25% of the
output budget — the share at which it is measurably competing with the transcript rather than
merely being a large number. `medium` sits at 14%.

The issue's premise was half right: transcription *is* mechanical, but **deciding who spoke is
not**. It is inference over voice, address forms and conversational role, and it is the first
thing to degrade when the model is told to think less.

## Consequences
- A lesson costs roughly 5 minutes of wall clock across three windows instead of ~3 at `low`,
  against ~30 for the single hour-long call this replaced. Thinking adds ~28k tokens an hour,
  which is not visible against the $100/mo cap.
- **The measurement is one window of one lesson, scored by a 5-cue canary.** That is enough to
  reject `low`, which is what this decision does, and it is not enough to call `medium` the best
  level. `jld-hc9.1` leaves a follow-up to settle it with `distill label` — blind clips and real
  ground truth — which is the instrument this repo already built for exactly this question.
- `medium` produced **145 segments where the reference produced 192**, and its one canary miss
  was a merged fragment. Under-splitting is `jld-beb`, the largest measured defect in Pass A, so
  the level may be trading one defect for another; the follow-up measures both.
- **Some of what looks like a stall is a call that is still thinking, and aborting it is still
  right.** A patient probe at `medium` (no retry, 1500 s tolerance) produced its first byte at
  **347.8 s** — past the 300 s watchdog, so the pipeline would have aborted and re-rolled it.
  That sounds like the watchdog is too tight until you look at what the patient call returned:
  **5,769 characters against 15,787** for the healthy `medium` run on the same window, about a
  third, which is the shape of the silent compression ADR-0005 was written about. So a slow
  first byte did not buy a good transcript here, and a re-roll is cheaper than waiting for a bad
  one. One observation, and the probe did not keep the transcript to score — but it is evidence
  against *raising* the timeout, which was the obvious reading an hour earlier.
- The intermittent stall itself is not a thinking-level effect and this decision does not fix
  it. It is the same `jld-hc9.3` phenomenon and `_with_retry` is still what handles it.
- The first-byte figures are the calibration `jld-hc9.3.3` was waiting for, and they are
  level-dependent: **4 s at `low`, 59 s at `medium`, 71 s at the model default**, against a
  300 s idle timeout. Noted on that issue rather than changed here.
