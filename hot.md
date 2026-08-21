# jp-lesson-distill — Hot State

## API key / billing setup (2026-07-10, settled)
- This pipeline uses its **own Google Cloud project ("Gemini Project"**, Tier 1 paid, **$100/mo spend cap**) via `GEMINI_API_KEY` in the repo's gitignored `.env` — which overrides shell env (the shell's key belongs to the **FlashGen project, $20/mo cap**; both under the same billing account, $250 account cap). Verified working.
- The 7/9–7/10 "rate limiting" mystery: an auto-set experimental **$3.87/mo spend cap** on the FlashGen project — hit mid-run and surfaced as 429s. Not free-tier RPM (the account was Tier 1 all along).

## Snapshot (2026-07-09)
Project bootstrapped from a design session in the ML vault: charter, ADRs 0001–0004, architecture doc, and the Phase-1 audio pipeline implemented (`distill` CLI: prep → Pass A transcript → moment detection → Pass B re-listen → emit to vault `_raw/`). Vault-side `/distill-jp-lesson` skill created in the General vault.

## Where we left off (2026-08-20 — windowed Pass A, jld-hc9.2)
- **Pass A now windows the recording internally** (ADR-0005): ~20 min windows / 30 s overlap, one Gemini call each, cached as `work/<date>/transcript_w<NN>.json`, merged into the usual `transcript.json` by shifting each window's timestamps and splitting every overlap at its midpoint. `--window-minutes` / `--overlap-seconds` on the CLI; `0` = the old single call. Detect/Pass B/emit untouched.
- **Verified on the 2026-08-17 recording (56:00):** 437 segments, last at 55:56, monotonic, no duplicates, no empty 5-minute stretch — against 394 segments stopping at 49:59 for the old single-pass run of the same lesson. All 313 reference lines from the hand-staged windows (`work/ref-windowed/`) are present in the merge. Wall clock **~5 min** for all three windows (the single hour-long call was ~30). Output lives in `work/hc9-2-verify/20260817/` (better than the compressed `work/20260817/transcript.json` — treat that one as the cautionary sample, not a reference).
- `tests/` exists now: `uv run pytest` (12 tests, offline, uses `work/ref-windowed/` when present).
- **Two things the run exposed, both already filed:** (1) Gemini sometimes never sends a first chunk on an audio+`response_schema` call — reproduced 5 of 6 times on a 24 s synthetic window while a text call answered in 2.5 s, so the stream just sits in an SSL read until the 30-min timeout → that is `jld-hc9.3`'s watchdog, and probably `jld-hc9.1`'s `thinking_level` too. (2) Two independent Pass A runs over the same lesson agree on the words but only **71%** on teacher-vs-student attribution → filed as its own issue.

## Where we left off (2026-07-10)
- **First real-lesson run complete (2026-07-07 recording, 1:02:56):** 348 Pass-A segments, 18 candidates, 18 confirmed moments → `_raw/20260707-moments.json` + `-transcript.md` in the General vault, awaiting `/distill-jp-lesson`. Quality is strong: it re-found the PDF's 作られる→作れる and 焦点を当てる corrections (dedupe targets) and surfaced ~15 moments NOT in the PDF (e.g. noun+けど without だ, 高いではない, サッカーに上手, 上手だかどうか, 勝ちません past-tense, フランスで→を旅行, コンロ/注視 pronunciations).
- Ops lesson learned the hard way: non-streaming generate_content on an hour of audio dies with `[Errno 54] Connection reset` (idle connection during 10-20 min of silent generation). Fixed in `gemini.py`: **streamed generation** + 30-min HTTP timeout + 3-attempt retry on transient httpx errors. Full run ≈ 30 min wall-clock on `gemini-pro-latest`.
- Earlier: synthetic mini-lesson verification passed (diarization 7/7, planted errors captured verbatim). `gemini-2.5-pro` retired for new API users mid-build → default is the `gemini-pro-latest` alias (ADR-0002).

## Next actions
- [ ] `jld-hc9.3` (stream watchdog) is now the top of the epic — the stall above will bite a real lesson eventually.
- [ ] Run `/distill-jp-lesson` in the General vault on the 20260707 output; create the first delta cards via FlashGen.
- [ ] Spot-check a few moments against the actual audio timestamps (are t_start/t_end accurate enough for clip review?).
- [ ] Tune prompts if the card session reveals noise (false uncorrected-errors, missed hesitations — note: 0 hesitation moments were flagged this run; check if the detect prompt undersells them).
- [ ] Consider `--pass-a-model` flag (cheap Flash Pass A + Pro Pass B) if per-lesson cost/latency matters.

## Open questions / decisions pending
- Pass A model economics: `gemini-2.5-pro` quality vs `flash` cost on a full hour — measure on the first real run (swap with `--model`).
- Whether Pass B needs wider clips (±15 s default) for context-dependent grammar errors.
- Phase 2: video/whiteboard fusion (keyframe extraction vs Gemini single-pass video) — deliberately deferred.
