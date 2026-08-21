# jp-lesson-distill

Distill Japanese lesson recordings (1-on-1 with Soso先生) into learning moments — spoken mistakes, verbal corrections, hesitations, verbally-introduced material — as input to flashcard generation. See `CLAUDE.md` for the charter and `docs/architecture.md` for the design.

## Setup

```sh
uv sync
echo 'GEMINI_API_KEY=...' > .env   # the jp-lesson-distill project's key (billed, Tier 1)
```

Key resolution: a `GEMINI_API_KEY` in the repo's `.env` (gitignored) wins over the shell environment, and is passed to the SDK explicitly so a stray `GOOGLE_API_KEY` can never take precedence. This keeps the pipeline on its own billed Google Cloud project while the shell's key stays free-tier for other experiments. Without a `.env`, the shell's `GEMINI_API_KEY` is used.

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
#   --window-minutes N         Pass A window length (default 20; 0 = one call)
#   --overlap-seconds N        overlap between windows (default 30)
```

Stages cache under `work/<date>/` (`audio.m4a`, `windows/`, `transcript_w<NN>.json`, `transcript.json`, `candidates.json`, `moments.json`, `clips/`). Re-runs skip completed stages; delete a stage file to redo it — e.g. delete `candidates.json` after editing the detection prompt without re-transcribing the hour, or delete a single `transcript_w<NN>.json` to redo one bad window.

Pass A never sends the whole hour in one call — it cuts the audio into ~20-minute windows overlapping by 30 s, transcribes each on its own, and merges them back into one `transcript.json`, splitting each overlap at its midpoint (ADR-0005). A single hour-long call loops, truncates, or silently compresses: valid JSON with the middle of the lesson quietly missing.

Output lands in the General vault `_raw/` as `YYYYMMDD-moments.json` + `YYYYMMDD-transcript.md`, then `/distill-jp-lesson` (a General-vault Claude skill) fuses them with the lesson note and generates Anki cards via FlashGen.

## Verification

`scripts/make_test_lesson.sh` synthesizes a mini-lesson with macOS `say` (Kyoko/Samantha voices) containing a planted student error (「学校で行きます」, particle mistake), so the full pipeline can be smoke-tested end-to-end without a real recording:

```sh
./scripts/make_test_lesson.sh          # writes work/test/test_lesson.m4a
uv run distill run work/test/test_lesson.m4a --date 19990101 --out-dir work/test/out
```

Expected: the planted mistake survives verbatim in a `correction` moment (で → に), and `work/test/out/19990101-moments.json` validates against the schema.

## Design notes: why an LLM does the listening

(Companion to `docs/architecture.md` and ADR-0002. The model is the `gemini-pro-latest` alias, not a pinned version.)

### Instructable listener vs. dedicated transcriber

Dedicated STT systems (Whisper, ElevenLabs Scribe, Deepgram) are single-purpose models: audio in, fluent text out, behavior baked in at training time. Gemini is an LLM that accepts audio as input tokens — transcription is just another instruction-following task, so the prompt genuinely changes how it decodes. That one property drives everything below.

### Preserving mistakes instead of correcting them

Every speech recognizer carries a language-model prior — a learned expectation that speech is fluent. In dedicated STT that prior is fused into the decoder: say 「学校**で**行きました」 and beam search weighs the acoustics against "nobody says that," so the fluent 「学校に行きました」 often wins. There is no knob to turn this off; you can't tell Whisper anything.

With Gemini the fluency prior still exists but is *steerable*: the Pass A prompt instructs it to keep wrong particles and conjugations, render mispronunciations in kana as heard, and flag garbled audio as uncertain rather than guess-correcting. Instruction-following competes with the fluency prior — and mostly wins. Because "mostly" isn't "always" over a full hour, the architecture doesn't trust Pass A alone: Pass B re-listens to a short padded clip with a stronger adversarial instruction ("this is a learner; the error IS the payload; quote him exactly, then separately state the correction"). A 30-second clip plus an explicit instruction focuses the model far better than one heroic pass. Bonus no STT can offer: the model understands Japanese grammar, so the same call that transcribes the error also explains it and supplies the fix.

### Diarization by role, not just by voice

Dedicated diarization (pyannote, Scribe's built-in) clusters voice embeddings acoustically and outputs anonymous "Speaker 1 / Speaker 2" labels. Gemini fuses acoustic evidence (pitch, timbre, pace) with semantic evidence — the prompt says the teacher is a native speaker who corrects and the student is an English-native learner who makes mistakes and code-switches. Fluency, accent, who echoes whom with corrections: that's how it assigns meaningful *role* labels directly, no mapping step.

### Timestamps: estimated, not measured

Pass A segments and detected candidates carry `"MM:SS"` timestamps that Gemini emits as part of its structured output. These are **model-estimated positions** inferred from progress through the audio token stream (~32 tokens/second), not decoder-aligned word timings — typically accurate to a few seconds, with drift growing the deeper into a call they are. Windowed Pass A bounds that: every window's clock restarts at zero and only ~20 minutes of drift can accumulate before the merge re-anchors it to absolute time. The pipeline never trusts them at fine granularity: Pass B clips are padded ±15 s, and nothing downstream needs better than "somewhere in this window."

### Trade-off

The runner-up (ADR-0002) was ElevenLabs Scribe, a dedicated verbatim-focused STT. On pure transcript mechanics it wins: word-level timestamps, less long-audio hallucination risk. The moment-centric design absorbs those weaknesses — Pass A only *locates* moments, Pass B restores precision where it counts — and in exchange we get the preserve-errors instruction, role-aware diarization, grammar explanations, one vendor/one key, and the only realistic path to Phase 2 (whiteboard video) from the same model. If Pass A transcript quality proves to be the weak link on real lessons, the documented upgrade is a dual-engine diff: Scribe anchors a precise timeline, Gemini keeps doing the understanding, and disagreement regions become free mistake-candidate flags.
