# jp-lesson-distill — Hot State

## Snapshot (2026-07-09)
Project bootstrapped from a design session in the ML vault: charter, ADRs 0001–0004, architecture doc, and the Phase-1 audio pipeline implemented (`distill` CLI: prep → Pass A transcript → moment detection → Pass B re-listen → emit to vault `_raw/`). Vault-side `/distill-jp-lesson` skill created in the General vault.

## Where we left off
- Pipeline verified end-to-end against the synthetic mini-lesson (README `## Verification`): diarization 7/7 turns, both planted errors (「学校でいきました」で→に correction, uncorrected 「面白いでした」) captured verbatim, correctly typed and explained.
- Default model is the `gemini-pro-latest` alias — a pinned `gemini-2.5-pro` was already retired for new API users during development (ADR-0002).
- Not yet run on a real 1-hour lesson recording.

## Next actions
- [ ] Run on a real recording from OneDrive (suggest the 2026-07-07 lesson — its note is rich) and inspect `moments.json` quality: does it re-find corrections documented in `20260707 Lesson.md`, and find new ones?
- [ ] Run `/distill-jp-lesson` in the General vault on that output; create a couple of cards end-to-end.
- [ ] Tune prompts/taxonomy based on the first real lesson (expect iteration on Pass A diarization and detection precision).

## Open questions / decisions pending
- Pass A model economics: `gemini-2.5-pro` quality vs `flash` cost on a full hour — measure on the first real run (swap with `--model`).
- Whether Pass B needs wider clips (±15 s default) for context-dependent grammar errors.
- Phase 2: video/whiteboard fusion (keyframe extraction vs Gemini single-pass video) — deliberately deferred.
