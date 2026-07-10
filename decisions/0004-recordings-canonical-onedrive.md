---
status: accepted
date: 2026-07-09
---

# 0004 — Recordings are canonical in OneDrive; never stored in vaults or repo

## Context
Class recordings (Zoom/QuickTime capture, ~1 hr) are large binaries. The Obsidian vaults are git-backed markdown stores; this repo is code.

## Drivers
- Vaults must stay lean (auto-backup commits would balloon with media).
- One canonical home for recordings that already exists in Steven's workflow.

## Decision
Recordings live in **OneDrive** (canonical). The CLI accepts any local path — including the OneDrive sync folder directly, so no copy step is required. A transient copy in the vault `_raw/` is allowed as a staging convenience but is deleted after absorption (with confirmation). Audio never lands in this repo; `work/` (derived audio, clips, stage caches) is gitignored.

## Consequences
- No automated OneDrive watcher for now; runs are invoked manually per lesson with a path.
- If a recording routine gap shows up (classes not consistently captured), that's a process fix, not a pipeline fix.
