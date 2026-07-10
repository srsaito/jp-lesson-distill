---
status: accepted
date: 2026-07-09
---

# 0002 — Gemini for Pass A (full transcript) and Pass B (clip re-listen)

## Context
STT models are trained toward fluent text, so they auto-correct learner errors — precisely the signal this project needs to preserve. Japanese worsens this: kana/kanji normalization hides pronunciation-level mistakes. The pipeline therefore needs an **instructable listener** that can be told "do NOT fix the student's errors."

## Drivers
- Verbatim error preservation is the core requirement; Claude has no audio input.
- Diarization (teacher vs student), timestamps, JA+EN code-switching.
- Vendor/plumbing simplicity for a personal project.

## Options
1. **Gemini for both Pass A and Pass B** — one vendor; promptable diarization/timestamps/verbatim; also the only realistic Phase-2 video model.
2. ElevenLabs Scribe (verbatim-focused STT, word timestamps, diarization, ~$0.40/hr) for Pass A + Gemini for Pass B — best pure transcript, two vendors.
3. Local Whisper (mlx-whisper) for Pass A — free/private, but worst at preserving errors; needs pyannote for diarization.
4. Dual-engine diff (Scribe + Whisper): disagreement regions auto-flag mistake candidates — most signal, most plumbing.

## Decision
**Option 1.** Default model is the **`gemini-pro-latest` alias** (CLI `--model` to swap or pin) — a pinned `gemini-2.5-pro` default already rotted once during development (retired for new API users), so the alias is the safer default for a personal pipeline; pin a dated model only if reproducibility matters for a comparison. Pass A prompts for diarized, timestamped, verbatim-biased transcription; Pass B re-listens to ~30 s clips per candidate moment with an explicit preserve-errors instruction.

## Consequences
- One API key (`GEMINI_API_KEY`), one SDK (`google-genai`), and a clear upgrade path to Phase-2 video.
- Pass A timestamps are coarser than dedicated STT word timestamps — acceptable because Pass B clips use ±15 s windows.
- Options 2 and 4 remain plausible upgrades if Pass A transcript quality proves limiting; revisit after real-lesson runs.
- Class audio is sent to Google's API (accepted for this project).
