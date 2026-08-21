---
status: accepted
date: 2026-08-20
---

# 0005 — Pass A transcribes the recording in overlapping windows

## Context
Pass A originally sent the whole ~60-minute lesson to Gemini in one call. Three failure modes showed up on real lessons (GH #1/#2, epic `jld-hc9`): the response loops on a repeated phrase and never terminates; it truncates; or — worst — it **silently compresses**. The 2026-08-17 run is the cautionary case: valid JSON, 394 plausible segments, and the last timestamp at 49:59 of a 56:00 recording, with whole explanations missing from the middle. Nothing in the output says it went wrong, and Pass B then clips ±15 s around timestamps that drifted.

A hand-staged experiment — cutting the same lesson into three ~20-minute pieces and transcribing each separately (`work/ref-windowed/w{1,2,3}/`) — produced 382 segments covering 00:00–55:55 with no compressed stretches. Windowing is the fix; the question was where it belongs.

## Drivers
- Coverage and density must be verifiable, and a per-window failure must be retryable on its own.
- Timestamps are load-bearing: Pass B's clips are only as good as Pass A's absolute times.
- `transcript.json` feeds detect, Pass B and emit; changing its shape would ripple through all of them.
- Manual staging (cut audio by hand, run three times, paste results together) is not a workflow.

## Options
1. **Window inside Pass A**, merge to the same `transcript.json` — later stages unchanged.
2. Keep single-pass and add retries/sanity gates — retries re-roll the same 60-minute dice, and a compressed response passes schema validation.
3. Make windows a first-class pipeline stage with per-window `transcript.json` files, and teach detect/Pass B to read a list — larger blast radius for no gain, since nothing downstream cares about window boundaries.
4. Split by silence (VAD) rather than a fixed clock — avoids cutting mid-sentence, but adds a dependency and a variable-length failure mode for a problem the overlap already solves.

## Decision
**Option 1.** `audio.py` cuts the prepped audio into windows (default 20 min, 30 s overlap, ffmpeg stream copy) under `work/<date>/windows/`; Pass A transcribes each one separately and caches it as `work/<date>/transcript_w<NN>.json`, so a failed or bad window is retried by deleting that one file. Every window's timestamps are shifted by its offset to absolute recording time, and the windows are stitched with a **midpoint rule**: in each overlap region, window N owns everything before the midpoint and window N+1 everything after. The merged result is written to `transcript.json` — the same file, the same schema, so detect, Pass B and emit are untouched.

`--window-minutes` / `--overlap-seconds` expose the knobs; `--window-minutes 0` restores the single call, which is what a sub-20-minute recording gets anyway.

## Consequences
- One 60-minute call becomes three ~20-minute calls: three uploads and three chances to fail, each cheap to retry, none able to lose the middle of the lesson unnoticed.
- The 30 s overlap costs a little duplicated audio; the midpoint rule means neither copy of a straddling utterance is emitted twice, and a same-speaker/same-text repeat inside the seam is dropped as a de-dup safety net.
- A window boundary can cut a turn in half, so an utterance may appear split across the seam. The overlap gives the model the surrounding context in one window or the other, which is why the seam is 30 s and not 0.
- Segments a window timestamps beyond its own span are discarded rather than trusted, so a hallucinated tail cannot poison the merge. The final window gets a drift allowance (10% of the window, at least 60 s) past the end of the recording, because nothing follows it and models routinely overshoot their own span by tens of seconds — the first real windowed run put window 1's last segment at 20:49 of a 20:00 window.
- Coverage becomes checkable: the last merged timestamp should land near the audio duration. Turning that into an enforced gate is `jld-hc9.4`.
