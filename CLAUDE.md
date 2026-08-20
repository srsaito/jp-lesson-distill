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
- `uv run distill run <recording> --date YYYYMMDD` — full pipeline. Stage outputs cache under `work/YYYYMMDD/`; re-runs skip completed stages (delete a stage file to redo it).
- Requires `GEMINI_API_KEY` in the environment. ffmpeg resolves from the `imageio-ffmpeg` wheel (no system install needed).

## How this workstream is organized
- **`hot.md`** — current state / where we left off. Snapshot, not append-log; update at end of session.
- **`decisions/`** — ADRs, one per file, MADR-lite. Capture firm decisions there; open threads in `hot.md`.
- **`docs/`** — architecture and design notes.
- Related vault material: lesson notes `_wiki/日本語の授業/` (General vault); recording-setup notes in Vault-ML under `Coding/MacBook Setup/`.


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
