# jp-lesson-distill — Hot State

## API key / billing setup (2026-07-10, settled)
- This pipeline uses its **own Google Cloud project ("Gemini Project"**, Tier 1 paid, **$100/mo spend cap**) via `GEMINI_API_KEY` in the repo's gitignored `.env` — which overrides shell env (the shell's key belongs to the **FlashGen project, $20/mo cap**; both under the same billing account, $250 account cap). Verified working.
- The 7/9–7/10 "rate limiting" mystery: an auto-set experimental **$3.87/mo spend cap** on the FlashGen project — hit mid-run and surfaced as 429s. Not free-tier RPM (the account was Tier 1 all along).

## Snapshot (2026-07-09)
Project bootstrapped from a design session in the ML vault: charter, ADRs 0001–0004, architecture doc, and the Phase-1 audio pipeline implemented (`distill` CLI: prep → Pass A transcript → moment detection → Pass B re-listen → emit to vault `_raw/`). Vault-side `/distill-jp-lesson` skill created in the General vault.

## Where we left off (2026-08-23 — diarization ground-truth kit, jld-dli)
- **Decision: the repair pass lives in the repo**, not the vault (code doesn't belong in the vault). It stays consistent with ADR-0003 by reading source materials but never writing vault artifacts. Cheap, because the materials sit next to the recording in OneDrive and the convention holds across all 14 lesson folders: `Z_L##.pdf` (textbook, `_au` suffix in L1–L10), `Z_kaiwa_scripts_L##.md` (dialogue scripts, already markdown with a 話者 column), `Z_audio_L##/*.mp3` (the textbook audio itself).
- **`distill label` / `distill score` are built** (`labeling.py` + `validate.py`, 21 tests). Blind, shuffled, stratified sampling → clips under `work/<date>/labeling/` → interactive listening (`--play`, resumable, saves after every keystroke, no network) → per-stratum accuracy, confusion by direction, and paired McNemar against a repaired transcript.
- **A kit for 2026-08-17 is cut and waiting: 41 clips** in `work/hc9-2-verify/20260817/labeling/` — 20 contested (where the windowed run and the hand-staged reference disagree), 20 random, 1 scripted. ~15 minutes of listening. Nothing is labelled yet; the repair pass is deliberately not built until there is ground truth to build against.
- Only 1 scripted clip because `Z_kaiwa_scripts_L13.md` covers 会話13-07/08/09 (the 2026-07-29 lesson) while 0817 reviewed 会話13-11, whose script is only in `Z_L13.pdf`. `--script` takes any file, so adding 13-11 to the .md (or extracting it) fixes this without code changes.
- The labelling options are `teacher` / `student` / `played` (textbook audio) / `other` (a live third voice) / `mixed` (the segment text runs two turns together — a segmentation defect, excluded from accuracy but counted) / `unsure`. Steven caught the gap: the 会話 has 利用者・司書・客・係員 in it, and if you are only offered two participants there is nowhere to put a voice that is neither. `played` and `other` are graded and are always errors for Pass A.
- **New issue `jld-lg6`, blocking dli:** played textbook audio is a third speaker. Soso先生 plays each dialogue twice and it lands on `teacher` — timestamps documented in `Z_kaiwa_scripts_L13.md` for the 0729 lesson, verbatim-matched against the source mp3s.

## Where we left off (2026-08-21 — Pass A sanity gates, jld-hc9.4)
- **`quality.py` gates every Pass A window** before accepting it, and re-rolls a failing window at a higher temperature (`--window-attempts`, default 2). Failed attempts are kept as `transcript_w<NN>.attempt<N>.json`; if all attempts fail, the least-bad one is promoted loudly instead of killing the run. Every window now prints one stats line, so degradation is visible in the log.
- **The threshold in the original issue was wrong and the measurement says so.** Segments-per-minute does NOT detect degradation: the known-bad 2026-08-17 single-pass run is 7.0/min, *inside* the 4.6–9.6/min range of known-good windows (chars/min is no better: bad 177, good 162–205). What separates them is **coverage** (89% vs 100%), **dead time** (8.1 min of audio behind under 1 char/s vs 0 min) and **max span** (361 s in one segment vs ≤74 s). Density is now logged, never gated.
- **Duplicates only warn.** In the bad run they are the failure mechanism — four consecutive lines from 11:13–13:04 re-emitted verbatim at 20:12–21:03, i.e. the model lost its place and replayed an earlier block over eight minutes of real audio. But the 2026-07-07 run's 11 duplicates are three passes over one textbook dialogue (「木村さん、ネットでパソコン買ったことありますか？」), a genuine drill. They look structurally identical; coverage and dead time are what tell them apart.
- 25 offline tests (`uv run pytest`). The 2026-07-07 run is *not* degraded, despite failing both thresholds the issue originally proposed — its 590-char segment is 3.5 chars/s of continuous speech, merely under-split.

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
- [ ] `jld-dli` (diarization repair) is designed but unbuilt: cue anchors (「スティーブンさん」/「先生」/「奥さん」/「妻」) + phase-block detection + disfluency evidence, with the textbook script (`Z_L13.pdf` has 会話13-11 with 利用者/司書 role labels) as an optional bonus, never a dependency. Open design question: whether the repo gets an optional `--textbook-pdf` input or the repair moves vault-side where the lesson note already maps the roleplay stretches.
- [ ] Run `/distill-jp-lesson` in the General vault on the 20260707 output; create the first delta cards via FlashGen.
- [ ] Spot-check a few moments against the actual audio timestamps (are t_start/t_end accurate enough for clip review?).
- [ ] Tune prompts if the card session reveals noise (false uncorrected-errors, missed hesitations — note: 0 hesitation moments were flagged this run; check if the detect prompt undersells them).
- [ ] Consider `--pass-a-model` flag (cheap Flash Pass A + Pro Pass B) if per-lesson cost/latency matters.

## Open questions / decisions pending
- Pass A model economics: `gemini-2.5-pro` quality vs `flash` cost on a full hour — measure on the first real run (swap with `--model`).
- Whether Pass B needs wider clips (±15 s default) for context-dependent grammar errors.
- Phase 2: video/whiteboard fusion (keyframe extraction vs Gemini single-pass video) — deliberately deferred.
