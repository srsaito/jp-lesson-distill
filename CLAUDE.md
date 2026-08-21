# Workstream: jp-lesson-distill

Distill Steven's Japanese lesson recordings (1-on-1 with Soso先生, ~1 hr) into **learning moments** — spoken mistakes, corrections, hesitations, verbally-introduced material — and feed extra flashcard candidates into the existing notes→cards flow. The recordings contain moments the PDF class notes miss: mistakes Soso先生 corrected verbally but didn't write down, and mistakes he didn't correct at all.

## Architecture (Phase 1 = audio only)

Core insight: **STT models auto-correct learner errors — the exact signal we want.** So the pipeline is moment-centric with a re-listen pass. Gemini (audio-native, instructable) is the ears; Claude is the brain.

- **Pass A** — Gemini transcribes the full recording: diarized (teacher/student), timestamped, prompted verbatim.
- **Detect** — Gemini text pass over the transcript flags candidate moments (correction / uncorrected-error / hesitation / new-item).
- **Pass B** — ffmpeg clips ~30 s around each candidate; Gemini re-listens per clip: "transcribe the student verbatim, do NOT fix errors; state the error and the correction."
- **Pass C** (not in this repo) — Claude fuses moments with the lesson note and generates cards, via the `/distill-jp-lesson` skill in the General vault.

Details: `docs/architecture.md`. Phase 2+ (video/whiteboard fusion, dual-engine STT diff) is future work — see ADRs.

## Current state (auto-loaded)
@hot.md
@decisions/README.md

## Boundaries & contracts
- **Repo↔vault contract is `moments.json`** (schema in ADR-0003 and `src/jp_lesson_distill/models.py`). This repo's job ends when `YYYYMMDD-moments.json` + `YYYYMMDD-transcript.md` land in `/Users/stevensaito/Docs/Vault-GeneralNotes/_raw/`.
- **Card generation lives in the General vault**, not here: skill `.claude/commands/distill-jp-lesson.md`, card rules `_wiki/日本語の授業/CLAUDE.md` (furigana, TTS, deck/tags), FlashGen MCP. Do not duplicate card rules in this repo.
- **Recordings are canonical in OneDrive** (ADR-0004). The CLI takes any local path (incl. the OneDrive sync folder). Never commit audio to this repo; never store recordings in the Obsidian vaults.

## Running
- `uv run distill run <recording> --date YYYYMMDD` — full pipeline. Stage outputs cache under `work/YYYYMMDD/`; re-runs skip completed stages (delete a stage file to redo it). Pass A windows the recording internally (`--window-minutes` / `--overlap-seconds`, ADR-0005) and caches each window as `transcript_w<NN>.json` — delete one to redo just that window.
- Requires `GEMINI_API_KEY` in the environment. ffmpeg resolves from the `imageio-ffmpeg` wheel (no system install needed).

## How this workstream is organized
- **`hot.md`** — current state / where we left off. Snapshot, not append-log; update at end of session.
- **`decisions/`** — ADRs, one per file, MADR-lite. Capture firm decisions there; open threads in `hot.md`.
- **`docs/`** — architecture and design notes.
- Related vault material: lesson notes `_wiki/日本語の授業/` (General vault); recording-setup notes in Vault-ML under `Coding/MacBook Setup/`.

## Engineering guide (for building/maintaining the transcription pipeline)

### Code map (`src/jp_lesson_distill/`)
| module | role | rule of thumb |
|---|---|---|
| `cli.py` | `distill run` argparse front-end | flags only; no logic |
| `pipeline.py` | stage orchestration + per-stage caching under `work/<date>/`; windowed Pass A and the overlap merge | every stage = one cached file; idempotent; re-run skips done stages |
| `gemini.py` | google-genai wrapper: upload, **streamed** structured generation, retry | the only module that talks to the API |
| `audio.py` | ffmpeg helpers (prep, window split, clip, duration) | bundled `imageio-ffmpeg` binary; no system install assumed; no ffprobe |
| `prompts.py` | the three stage prompts | the product is the *student's errors* — every prompt says "do not fix" |
| `models.py` | Pydantic schemas (= Gemini `response_schema`) + `moments.json` contract | changing `Moment`/`MomentsFile` means an ADR (ADR-0003) |

### Dev loop
- `uv sync` once; `uv run distill run <rec> --date YYYYMMDD` to run. `--skip-pass-b` is the cheap dry run; `--work-dir` isolates experiments.
- **Smoke test without a real lesson:** `scripts/make_test_lesson.sh` synthesizes a ~1-min lesson with planted errors under `work/test/` (macOS `say`). Run it before touching prompts or schemas.
- **Offline fixtures** (gitignored, local only): `work/ref-windowed/w{1,2,3}/<date>/transcript.json` are known-good windowed transcripts (`offset.txt` = seconds to add to reach absolute time); `work/20260729/pass_a_partial.attempt1.json` is a real repetition-loop sample. Unit-test parsing, merging, de-dup, loop detection and sanity gates against these — **never against the live API**.
- Tests: `uv run pytest` (`tests/`, pytest is in the `dev` dependency group). Anything that calls Gemini is an integration test and must be opt-in (env flag), not part of the default run.
- Quality gates before closing an issue: smoke test passes; unit tests pass; one real-lesson run if the change touches Pass A.

### Gemini rules (learned the expensive way — see GH #1/#2, epic `jld-hc9`)
- **Never transcribe a full hour in one call.** Pass A does this for you (ADR-0005): ~20 min windows, ~30 s overlap, per-window transcription, offset-shifted timestamps, overlap split at its midpoint. Single-pass 60-min calls loop, truncate, or — worst — *silently compress*: valid JSON, plausible density, whole explanations missing. `--window-minutes 0` restores the single call; it exists for regression checks on short recordings, not for lessons.
- **Always stream** (`generate_content_stream`); a non-streaming hour-long call gets its idle connection reset. Add a per-chunk inactivity watchdog — the HTTP timeout does not fire on a stream that trickles.
- **Thinking config:** `gemini-pro-latest` is a moving alias (now Gemini 3.x). Use `thinking_level` (`low` for transcription); `thinking_budget` is silently ignored on 3.x. Never send both. Pin the model explicitly and upgrade deliberately.
- Thinking and answer share the 65,536-token output window. Log `thoughts_token_count` — and treat `None` as "unknown", not zero.
- Structured output (`response_schema`) + disfluent audio is a known repetition-loop trigger, and this corpus is *made of* disfluencies. Detect loops on the stream tail and abort in seconds; retry that window at a higher temperature.
- Retry 429/5xx and transient httpx errors with backoff; retries are per window, never per hour.
- Cost: the Gemini Project has a **$100/mo cap**. An hour of audio ≈ 30 min wall-clock on Pro. Don't burn lessons on experiments — use the synthetic lesson or a single window.

### Transcription invariants (what "correct" means here)
- **Verbatim beats fluent.** A transcript that fixes the student's grammar is wrong even if it reads better. Mispronunciations are rendered in kana as heard.
- **Timestamps are load-bearing:** Pass B clips ±15 s around Pass A timestamps, so a timestamp error silently voids the verbatim re-listen. Segment starts must be monotonic, per-turn, and ≤ ~30 s apart in speech; a conversational hour yields roughly 600–800 segments — far fewer means compression, not a quiet lesson.
- **Diarization is binary and named:** `teacher` (Soso先生) / `student` (Steven). Spot-check lines like 「スティーブンさんはどうですか？」 land on the teacher.
- `uncertain: true` is a feature, not a failure — garbled spots are candidate mistake sites and flow into detect.
- Japanese text: normal kana/kanji orthography, 「」 for quotes, no furigana markup in this repo (that belongs to the vault's card rules).

### Don'ts
- Don't commit anything under `work/`, audio of any kind, or `.env`.
- Don't add card-generation logic here (lives in the General vault).
- Don't "fix" a hang by raising timeouts — find out why the stream stalled.
- Don't trust a Pass A run because it validated; check coverage (last timestamp ≈ audio length) and density.


<!-- BEGIN BEADS INTEGRATION v:1 profile:minimal hash:ca08a54f -->
## Beads Issue Tracker

This project uses **bd (beads)** for issue tracking. Run `bd prime` to see full workflow context and commands.

### Quick Reference

```bash
bd ready              # Find available work
bd show <id>          # View issue details
bd update <id> --claim  # Claim work
bd close <id>         # Complete work
```

### Rules

- Use `bd` for ALL task tracking — do NOT use TodoWrite, TaskCreate, or markdown TODO lists
- Run `bd prime` for detailed command reference and session close protocol
- Use `bd remember` for persistent knowledge — do NOT use MEMORY.md files

## Session Completion

**When ending a work session**, you MUST complete ALL steps below. Work is NOT complete until `git push` succeeds.

**MANDATORY WORKFLOW:**

1. **File issues for remaining work** - Create issues for anything that needs follow-up
2. **Run quality gates** (if code changed) - Tests, linters, builds
3. **Update issue status** - Close finished work, update in-progress items
4. **PUSH TO REMOTE** - This is MANDATORY:
   ```bash
   git pull --rebase
   bd dolt push
   git push
   git status  # MUST show "up to date with origin"
   ```
5. **Clean up** - Clear stashes, prune remote branches
6. **Verify** - All changes committed AND pushed
7. **Hand off** - Provide context for next session

**CRITICAL RULES:**
- Work is NOT complete until `git push` succeeds
- NEVER stop before pushing - that leaves work stranded locally
- NEVER say "ready to push when you are" - YOU must push
- If push fails, resolve and retry until it succeeds
<!-- END BEADS INTEGRATION -->
