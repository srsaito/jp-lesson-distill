---
status: accepted
date: 2026-07-09
---

# 0001 — Phase 1 is audio-only; video/whiteboard fusion deferred

## Context
Lessons produce both video (Zoom screen: Soso先生's whiteboard, textbook screenshots) and audio. The whiteboard content largely becomes the PDF class notes, which `/ingest-jp-lesson` already turns into cards.

## Drivers
- The unique value of recordings is **spoken** material the PDF misses: mistakes corrected verbally but not written down, and mistakes never corrected.
- The whiteboard is mostly redundant with the PDF; video's marginal value is lower than audio's.
- One hour of video is much more expensive and complex to process than one hour of audio (~115k Gemini tokens).

## Options
1. Audio-first, video later.
2. Design and build audio + video fusion together (Gemini single-pass video, or keyframe extraction + Claude over an interleaved timeline).

## Decision
**Option 1.** Ship the audio pipeline; treat video as Phase 2. The `moments.json` contract carries timestamps, so board-state snapshots can join the same timeline later without reworking the boundary.

## Consequences
- Faster to a working system; prompts/taxonomy get tuned on real lessons before video complicates things.
- Phase 2 will choose between Gemini native video (simplest) and ffmpeg keyframes + perceptual-hash dedupe + VLM reading of board states (transparent, debuggable). Timestamps remain the join key.
