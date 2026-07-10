---
status: accepted
date: 2026-07-09
---

# 0003 — `moments.json` is the repo↔vault boundary

## Context
Card generation needs three things that live in the General vault: the lesson notes, Steven's card rules (`_wiki/日本語の授業/CLAUDE.md`, auto-loaded there), and the FlashGen MCP config. Duplicating those into this repo would drift.

## Drivers
- Card rules maintained in exactly one place.
- The pipeline should be testable and runnable without touching Anki.
- The vault skill should not care how moments were produced (audio today, video later).

## Options
1. Pipeline ends at a structured `moments.json`; a vault-side skill (`/distill-jp-lesson`) does fusion + cards.
2. Self-contained repo: wire FlashGen MCP here and import card rules by path.
3. Whole flow in the repo, vault only receives the updated lesson note.

## Decision
**Option 1.** The pipeline emits into the General vault `_raw/` (the vault's ingestion landing zone):
- `YYYYMMDD-moments.json`
- `YYYYMMDD-transcript.md` (human-readable, diarized, timestamped)

Schema (source of truth: `src/jp_lesson_distill/models.py`):

```json
{
  "lesson_date": "YYYY-MM-DD",
  "source_recording": "/path/to/recording",
  "model": "gemini-2.5-pro",
  "generated_at": "ISO-8601",
  "moments": [
    {
      "id": "m01",
      "t_start": 754.0,
      "t_end": 783.0,
      "type": "correction | uncorrected-error | hesitation | new-item",
      "student_verbatim": "…exactly what Steven said, errors preserved…",
      "teacher_correction": "…or null…",
      "explanation": "what the error was / why this moment matters",
      "confidence": 0.0
    }
  ]
}
```

`t_start`/`t_end` are seconds from recording start.

## Consequences
- `/distill-jp-lesson` (General vault) consumes the file, dedupes against PDF-derived cards, runs the interactive FlashGen flow.
- Phase-2 video output can extend the schema (e.g. a `board_state` field) without moving the boundary.
- Files in `_raw/` are transient: absorbed then cleaned per vault rules (deletion always confirmed with Steven).
