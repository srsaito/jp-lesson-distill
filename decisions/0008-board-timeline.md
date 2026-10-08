---
status: accepted
date: 2026-10-08
---

# 0008 — The shared screen is read into a board timeline, beside `moments.json`

## Context
ADR-0001 deferred video because "the whiteboard is mostly redundant with the PDF". Two things turned out differently. The レジュメ PDF arrives a day or more after class and sometimes not at all (2026-09-28, 2026-10-05), so lesson notes for those days were written from audio alone. And the レジュメ is nothing more than the final state of each slide Soso先生 shared, so the recording already contains it, minutes after class ends.

Video was also tried for the other thing it might have fixed, who is speaking (`jld-2nl`). Mouth movement in the Zoom webcam tiles is right on long turns and worse than Pass A on short ones, so it is not used. This decision is about the screen only.

## Drivers
- The notes should not wait for the PDF.
- Pass A is fragile under added instructions (ADR-0005); nothing here may touch that call.
- The Gemini Project has a $100/month cap.
- The vault is in git, so whatever lands there has to be small.

## Options
1. **Feed the video to Gemini in Pass A.** About 1M input tokens an hour at a resolution that can read the text, roughly $5 a lesson, and it loads the one call that is already fragile.
2. **Read the screen locally, send only the settled slides to Gemini.** ffmpeg at one frame a second, read a frame when the screen has changed and then held still, macOS Vision for the text and its position, Gemini once per distinct slide.
3. **Local only.** Free, but Vision garbles kanji under furigana (地震 → 地態) and misses most pen strokes. Enlarging the frame does not help.

## Decision
**Option 2.** `board.py`, run at the end of every `distill run` (after the moments are emitted, warn-only, `--no-board` to skip) and by itself as `distill board <video> --date …`. The video is the recording itself, or `…_class_video.mp4` beside `…_class_audio.m4a`; a lesson with no video has no board. It runs for every lesson, PDF or not: the PDF is only the final state, and the order and timing of what was typed is lost in it.
- `read` — one frame a second; a frame is read when it differs from the last one read and has been still for two seconds. Regions that move constantly are ignored, because some recordings lay the webcam tiles over a corner of the screen. Writes `work/<date>/board/ocr.jsonl` and the frames it read.
- `build` — groups the reads into slides (a slide he returns to is the same slide), separates what was on a slide when it appeared from what was added while it was up, and stamps each added line with when it first appeared. Gemini transcribes each slide's final frame, cached per slide, at `thinking_level=LOW` (`--no-gemini` on `distill board` turns that off). The result is written for the vault.

The repo↔vault boundary gains two optional siblings of `YYYYMMDD-moments.json`:
- `YYYYMMDD-board.md` — per slide: its image, its text, the teacher's notes on it, handwriting, and the timed lines.
- `YYYYMMDD-board/` — one image per slide, 1280 px wide, about 1 MB a lesson.

`moments.json` itself is unchanged, so ADR-0003's schema stands. A lesson with no board files is still a complete hand-off.

## Consequences
- The レジュメ's content is available as soon as the recording is. When the PDF does arrive it remains the authority; the board is the stand-in.
- Cost is about ten to twenty small Gemini calls a lesson (cents), plus 10–20 minutes of local reading.
- macOS only: Vision comes through `ocrmac`, a dependency on Darwin alone. Elsewhere the board stage warns and the run carries on.
- A full `distill run` is 10–20 minutes longer.
- Timestamps on added lines are when the reader first saw them after the screen settled, so lines typed in one burst share a time.
- Web pages he shares are kept as a title and an image, not transcribed.
- Not done yet: the board is not given to detect or Pass B as context (`jld-333`'s follow-up). `work/<date>/board/frames/` is 20–45 MB a lesson and is copied by the OneDrive archive like any other non-audio file.
