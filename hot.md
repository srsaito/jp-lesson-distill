# jp-lesson-distill — Hot State

## Snapshot (2026-07-09)
Project bootstrapped from a design session in the ML vault: charter, ADRs 0001–0004, architecture doc, and the Phase-1 audio pipeline implemented (`distill` CLI: prep → Pass A transcript → moment detection → Pass B re-listen → emit to vault `_raw/`). Vault-side `/distill-jp-lesson` skill created in the General vault.

## Where we left off
- **First real-lesson run complete (2026-07-07 recording, 1:02:56):** 348 Pass-A segments, 18 candidates, 18 confirmed moments → `_raw/20260707-moments.json` + `-transcript.md` in the General vault, awaiting `/distill-jp-lesson`. Quality is strong: it re-found the PDF's 作られる→作れる and 焦点を当てる corrections (dedupe targets) and surfaced ~15 moments NOT in the PDF (e.g. noun+けど without だ, 高いではない, サッカーに上手, 上手だかどうか, 勝ちません past-tense, フランスで→を旅行, コンロ/注視 pronunciations).
- Ops lesson learned the hard way: non-streaming generate_content on an hour of audio dies with `[Errno 54] Connection reset` (idle connection during 10-20 min of silent generation). Fixed in `gemini.py`: **streamed generation** + 30-min HTTP timeout + 3-attempt retry on transient httpx errors. Full run ≈ 30 min wall-clock on `gemini-pro-latest`.
- Earlier: synthetic mini-lesson verification passed (diarization 7/7, planted errors captured verbatim). `gemini-2.5-pro` retired for new API users mid-build → default is the `gemini-pro-latest` alias (ADR-0002).

## Next actions
- [ ] Run `/distill-jp-lesson` in the General vault on the 20260707 output; create the first delta cards via FlashGen.
- [ ] Spot-check a few moments against the actual audio timestamps (are t_start/t_end accurate enough for clip review?).
- [ ] Tune prompts if the card session reveals noise (false uncorrected-errors, missed hesitations — note: 0 hesitation moments were flagged this run; check if the detect prompt undersells them).
- [ ] Consider `--pass-a-model` flag (cheap Flash Pass A + Pro Pass B) if per-lesson cost/latency matters.

## Open questions / decisions pending
- Pass A model economics: `gemini-2.5-pro` quality vs `flash` cost on a full hour — measure on the first real run (swap with `--model`).
- Whether Pass B needs wider clips (±15 s default) for context-dependent grammar errors.
- Phase 2: video/whiteboard fusion (keyframe extraction vs Gemini single-pass video) — deliberately deferred.
