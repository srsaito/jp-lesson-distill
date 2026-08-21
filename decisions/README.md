# Decision Records (ADRs)

Architecture Decision Records for jp-lesson-distill. One decision per file, `NNNN-slug.md`. Lightweight MADR format: **Context → Drivers → Options → Decision → Consequences**. `status` is one of `proposed | accepted | superseded`.

| ADR | Decision | Status |
|---|---|---|
| [[0001-audio-first-scope]] | Phase 1 processes audio only; video/whiteboard deferred | Accepted |
| [[0002-gemini-pass-a-and-b]] | Gemini for both full-lesson transcription and clip re-listen | Accepted |
| [[0003-moments-json-contract]] | `moments.json` is the repo↔vault boundary | Accepted |
| [[0004-recordings-canonical-onedrive]] | Recordings stay canonical in OneDrive; nothing stored in vaults or repo | Accepted |
| [[0005-windowed-pass-a]] | Pass A windows the recording internally (~20 min, 30 s overlap) and merges to one transcript | Accepted |
