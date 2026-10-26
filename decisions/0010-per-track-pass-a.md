---
status: proposed
date: 2026-10-17
---

# 0010 — Pass A per Zoom participant track when the tracks exist

**Proposed, not accepted.** The structural measurements are in and favour per-track transcription. The measurement the spike's decision rule names as deciding it, verbatim fidelity judged by ear, is cut and waiting to be listened to (`jld-tlc`). Accept, amend or reject this ADR when that is scored.

## Context
Since 2026-10-08 Zoom writes one file per participant beside the mix: Steven's mic, Soso先生's mic, and his shared computer sound (the textbook playback). All three share the mix's clock and length. Zoom's noise suppression leaves each track at exact digital silence (−120 dB) while that person is quiet.

Pass A on the mix (A0) mislabels whole stretches (`jld-ec6`: 67% agreement with the tracks on 10/14, inverted for 45–56 min), merges two speakers into one segment (`jld-beb`), and transcribes playback as the teacher (`jld-lg6`). Relabelling A0 from track energy (A1, no extra Gemini call) fixes clear-cut labels, which took 10/14 from 12 to 22 moments. But A1 cannot split a merged turn or re-time a start.

The spike (`scripts/spike_per_track.py`) compared, over the same window of the mix:

- **A0**: mixed-file Pass A as it ran.
- **A1**: A0 relabelled by energy (the 10/14 rule).
- **A2**: the teacher track first, then Steven's track with the teacher's text as context, then the share track's active spans as played audio. Merged by start, each start snapped to its own track's energy onset.
- **A2c**: A2 with Steven's track transcribed without the context.

The prompts were single-speaker variants of `PASS_A`'s verbatim rules. Model, thinking level (`medium`) and `PASS_A` itself were unchanged.

## What it measured (10/14 window 3 = 39:00–56:35; 10/12 window 3 = 39:00–59:00; 10/25 whole lesson, 4 windows)

| | 10/14 | 10/12 | 10/25 |
|---|---|---|---|
| A0 segments the energy rule cannot decide (merged or mis-cut) | 22 of 69 | 41 of 101 | 91 of 366 |
| …of those, split by A2 into turns from 2+ tracks | 9 | 18 | 24 |
| segment starts on their own speaker's voice (−1..+2 s): A0 / A2 | 60/69 / 146/152 | 89/101 / 237/237 | 343/366 / 412/417 |
| segments: A0 / A2 / A2c | 69 / 152 / 122 | 101 / 237 / 290 | 366 / 417 / 442 |
| playback transcribed as its own lines (share track) | 14 lines in 2 spans | 33 lines in 10 spans | 12 lines in 3 spans |
| moments, fresh detect + Pass B per window: A0 / A1 / A2 / A2c | 8 / 8 / 9 / 6 | 8 / 6 / 11 / 10 | 35 / 32 / 33 / 30 |
| lines with no sound on their own track (hallucinated) in A2 | 6 | 3 | 4 |

- **10/25, the first whole lesson, ties on moments.** Its mixed transcript was better than 10/12's and 10/14's: 87% label agreement with the tracks, and a typical start 1.2 s from the energy onset against 4–5 s. So it is the lesson where per-track transcription has the least to fix. Each arm has about 10 moments the other lacks. Most are the same exchange timed more than 25 s apart, and the rest look like detect variance. Running detect twice and taking the union would find more than switching transcripts does.
- **Timestamps.** A0's starts are wrong by 14–25 s in places late in the window. On 10/14, a line Steven begins at 47:20, right after the teacher stops at 47:17, is placed at 46:59 inside a teacher turn. Pass B clips ±15 s, so that is the size of error that voids a re-listen. A2's starts are snapped to energy at 0.1 s.
- **Merged turns.** A1 still carries Steven's long hesitation at 48:48 inside a `teacher` segment, and a teacher explanation labelled `student`, because the energy rule leaves mixed segments alone. A2 separates both.
- **Playback.** The 会話 that A0 wrote as one 14-line `teacher` block comes back from the share track as the dialogue, line by line, under its own label. This is `jld-lg6` solved for lessons with tracks.
- **New moments are real exchanges.** On 10/12, A2 adds からに／に, 一人暮らし, 正しい／ただし and 誰もいない. These are exchanges A0 merged or mislabelled.
- **Moment counts on one window are within detect's own variance.** The whole-lesson 12 vs 22 does not reproduce at window scale (8 vs 8 on 10/14). The count is weak evidence either way.
- **Hallucination is real and cheap to catch.** The teacher track got invented うん lines every 4 s while Steven spoke (10/14 40:14–40:27, track at −89 to −120 dB). On 10/14, Steven's track got two invented interjections, in the context arm only. Every such line sits on digital silence, so an energy check removes it.
- **Soft speech is not lost, but it is quiet.** Steven's soft はい／うーん／あー sit at −33 to −55 dB on his own track. The −45 dB threshold from the relabel rule misses them. "Any sound" is anything above about −70 dB.
- **Context cuts both ways (verbatim, unresolved).** With the teacher's text as context (A2), one clear false start on 10/12 was smoothed away (A2c 「食べるものもいまし…あー、言いました」, A2 「食べるものも、あー、言いました」). Without context (A2c), Steven's lines came back lumped into long segments on 10/14 (47 vs 77), on a call that reported no thinking at all and answered in 4 s. On 10/12 A2c instead produced the most segments of any arm. One run each cannot separate the arm from run-to-run variance.
- **Cost.** Three calls per window instead of one: teacher 51–58 s, then Steven 66–69 s waiting on it, then one call per playback span (24 s on 10/14; 381 s on 10/12, including one stalled call).

## Decision (proposed)
When all three tracks are present and well-formed, transcribe per track, provided the verbatim kit does not show A2 or A2c losing to A1 on Steven's lines. The structural case is already made: speaker by construction, turns split, starts on the right voice, playback separated.

Whatever the kit says:

- **⭐ REQUIREMENT: the mixed-recording path stays a fully supported fallback** (Steven, 2026-10-26). Per-track transcription is an upgrade for lessons that have the tracks, never a dependency. The pipeline must still produce `moments.json` from the mix alone, with the quality it has today: windowed Pass A, the sanity gates, the address-form check, and the energy relabel (A1) when tracks exist but cannot be transcribed. It falls back when:
  - **No tracks:** lessons before 2026-10-08, or Zoom's per-participant setting turned off.
  - **Equipment problems:** a track missing, empty, corrupt or unreadable; a length that does not match the mix (within 2 s); a role that cannot be identified by exact display-name prefix; or a mic track that is silent for the whole lesson. A silent share track is normal (no playback) and only warns.
  - **A partial set:** any mic track missing means the mixed path for the whole lesson, not a guess at the missing speaker.
  - **Per-track transcription failing** on a window after its retries: that window falls back to the mixed path, and the merged transcript says which windows did.
  - **Asked for:** a CLI flag forces the mixed path (and another forces per-track), for regression checks and comparisons.
  
  The run log always says which path each window took and why. The fallback path stays under test: the unit tests and the smoke test cover it (the synthetic lesson has no tracks, so the smoke test exercises it on every run), and removing it needs a new ADR.
- **Drop any per-track line whose own track is silent** (max below about −70 dB within −2..+3 s of its start).
- **Transcribe the share track over its active spans only, as played audio.** Warn when it is silent all lesson.
- **Warn when a track's silence is not digital zero.** Everything above was measured with suppression on.

Open until the kit is scored: **context or no context for Steven's track.** If neither is clearly better, prefer A2c. It is one call fewer on the critical path, and it cannot copy the teacher.

## Consequences
- If accepted: `jld-ec6`'s verify/repair step becomes the fallback-path check; `jld-beb` and `jld-lg6` are re-scoped to the mixed-file path.
- The transcript schema needs a third speaker value (`played`) or a flag. Today the spike passes playback to detect as a prefixed teacher line. That is a `models.py` change, so it needs its own ADR (ADR-0003).
- Per-track segments carry an `end` from energy for free. Pass B could clip to real turn boundaries instead of ±15 s.
- Lessons before 2026-10-08 never get this; their quality stays where A1 puts it.
