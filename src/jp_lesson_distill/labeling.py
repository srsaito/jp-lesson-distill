"""Ground truth for diarization: sample lines, cut blind clips, score what you hear.

Pass A's teacher/student labels have never been measured against reality — two runs
of the same lesson agree only 71% of the time, and neither one is known to be right
(jld-dli). This module builds a listening kit and scores it.

Three rules the design exists to enforce:

1. **Blind.** The listener sees the clip and the text, never the model's label.
   Seeing "teacher" is enough to make you hear a teacher.
2. **Shuffled.** Clips are presented out of order, because a lesson alternates
   speakers and a listener working in time order will infer rather than hear.
3. **Stratified.** Contested lines are rare, so a random sample measures the base
   rate and says almost nothing about the cases a repair pass has to get right.
   Random, contested and scripted lines are sampled and scored separately.

"Correct" means WHO PHYSICALLY SPOKE, and four things can be speaking:

- `teacher` / `student` — including when they act. A student reading the librarian's
  part in a roleplay is `student`; the words belong to 利用者, the voice is Steven's.
- `played` — the textbook audio. Soso先生 plays each 会話 in class, so the voices of
  客/係員/利用者/司書 are actors on a recording, not anyone in the room (jld-lg6).
- `other` — a live third voice: someone else in the room, a phone, another recording.

`played` and `other` are always errors for Pass A, whose schema can only say teacher
or student. Counting them is the point: it measures how much of the transcript is
speech that no participant produced.
"""

from __future__ import annotations

import json
import math
import random
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path

from .models import Segment, fmt_ts, parse_ts

GRADED = ("teacher", "student", "played", "other")
TRUTH_VALUES = (*GRADED, "unsure")

# Address forms pin a speaker regardless of voice: only the teacher says 「スティーブンさん」
# or 「奥さん」 (someone else's wife), only the student says 「先生」 as an address or
# 「妻」 (his own wife). Sparse — about nine lines an hour — but decisive where they land.
MARKERS: tuple[tuple[str, str], ...] = (
    ("スティーブン", "teacher"),
    ("先生", "student"),
    ("奥さん", "teacher"),
    ("妻", "student"),
)

CLIP_LEAD_IN = 2.0   # s before the line starts — enough to catch the voice, not the last turn
CLIP_MAX = 12.0      # s of a line is plenty to identify who is speaking
MATCH_WINDOW = 45.0  # s: how far apart two runs may time the same utterance


# --- text similarity (kana/kanji, so character bigrams beat word tokens) ---

def _grams(text: str) -> set[str]:
    t = re.sub(r"[\s、。，．,.！？!?「」『』…]", "", text)
    return {t[i:i + 2] for i in range(len(t) - 1)}


def similarity(a: str, b: str) -> float:
    """Overlap of the shorter string's bigrams — tolerant of one run merging turns."""
    ga, gb = _grams(a), _grams(b)
    if not ga or not gb:
        return 0.0
    return len(ga & gb) / min(len(ga), len(gb))


def jaccard(a: str, b: str) -> float:
    ga, gb = _grams(a), _grams(b)
    if not ga or not gb:
        return 0.0
    return len(ga & gb) / len(ga | gb)


# --- the kit ---

@dataclass
class Item:
    id: str
    stratum: str          # random | contested | scripted
    t_start: float
    text: str
    speaker: str          # Pass A's label — withheld from the listener
    clip: str
    order: int            # shuffled presentation order
    compare_speaker: str | None = None
    reason: str = ""      # why this line was sampled
    truth: str | None = None  # filled in by listening

    def to_json(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=False)


@dataclass
class Kit:
    date: str
    items: list[Item] = field(default_factory=list)

    @property
    def labelled(self) -> list[Item]:
        return [i for i in self.items if i.truth]

    def save(self, path: Path) -> None:
        path.write_text("\n".join(i.to_json() for i in self.items) + "\n")

    @classmethod
    def load(cls, path: Path, date: str) -> Kit:
        items = [Item(**json.loads(line)) for line in path.read_text().splitlines() if line.strip()]
        return cls(date=date, items=items)


def match(segment: Segment, pool: list[Segment], window: float = MATCH_WINDOW) -> Segment | None:
    """The same utterance in another run of the same lesson, if it is there."""
    t = parse_ts(segment.start)
    near = [s for s in pool if abs(parse_ts(s.start) - t) <= window]
    if not near:
        return None
    best = max(near, key=lambda s: similarity(segment.text, s.text))
    return best if similarity(segment.text, best.text) >= 0.5 else None


def marker_conflict(segment: Segment) -> str | None:
    """The line contains an address form its label contradicts."""
    for marker, implied in MARKERS:
        if marker in segment.text:
            return None if segment.speaker == implied else f"says 「{marker}」 but labelled {segment.speaker}"
    return None


def parse_script(markdown: str) -> list[str]:
    """Utterances from a Z_kaiwa_scripts_L##.md table (| 話者 | 発話 |).

    Best-effort by design: the file is hand-maintained prose plus tables, and a lesson
    that never reached the dialogue simply yields nothing.
    """
    lines = []
    for row in markdown.splitlines():
        row = row.strip()
        if not row.startswith("|") or row.count("|") < 3:
            continue
        cells = [c.strip() for c in row.strip("|").split("|")]
        if len(cells) < 2 or not cells[1] or set(cells[1]) <= set("-: "):
            continue
        if cells[0] in ("話者", "会話"):  # header row
            continue
        if len(_grams(cells[1])) >= 4:
            lines.append(cells[1])
    return lines


def find_script(recording: Path) -> Path | None:
    """Z_kaiwa_scripts_L##.md sits next to the recording in OneDrive; absence is normal."""
    matches = sorted(recording.parent.glob("Z_kaiwa_scripts_*.md"))
    return matches[0] if matches else None


def build_items(segments: list[Segment], *, compare: list[Segment] | None = None,
                script: list[str] | None = None, n_random: int = 20, n_contested: int = 20,
                n_scripted: int = 10, seed: int = 0, min_chars: int = 8) -> list[Item]:
    """Pick the lines to listen to, stratum by stratum, longest-odds first.

    Contested and scripted are drawn before random so a line that qualifies for a
    scarce stratum is not wasted on the plentiful one.
    """
    rng = random.Random(seed)
    usable = [s for s in segments if len(s.text) >= min_chars]
    taken: set[int] = set()
    picked: list[tuple[Segment, str, str, str | None]] = []  # segment, stratum, reason, compare label

    contested: list[tuple[int, Segment, str, str | None]] = []
    for i, seg in enumerate(usable):
        if compare is not None:
            other = match(seg, compare)
            if other is not None and other.speaker != seg.speaker:
                contested.append((i, seg, f"other run says {other.speaker}", other.speaker))
                continue
        conflict = marker_conflict(seg)
        if conflict:
            contested.append((i, seg, conflict, None))
    rng.shuffle(contested)
    for i, seg, reason, other in contested[:n_contested]:
        taken.add(i)
        picked.append((seg, "contested", reason, other))

    if script:
        scripted = [(i, s) for i, s in enumerate(usable)
                    if i not in taken and max((jaccard(s.text, line) for line in script), default=0) >= 0.6]
        rng.shuffle(scripted)
        for i, seg in scripted[:n_scripted]:
            taken.add(i)
            picked.append((seg, "scripted", "matches the textbook dialogue", None))

    rest = [(i, s) for i, s in enumerate(usable) if i not in taken]
    rng.shuffle(rest)
    for i, seg in rest[:n_random]:
        picked.append((seg, "random", "", None))

    picked.sort(key=lambda p: parse_ts(p[0].start))
    order = list(range(len(picked)))
    rng.shuffle(order)
    return [
        Item(
            id=f"{fmt_ts(parse_ts(seg.start)).replace(':', '')}-{n:03d}",
            stratum=stratum,
            t_start=parse_ts(seg.start),
            text=seg.text,
            speaker=seg.speaker,
            compare_speaker=other,
            reason=reason,
            clip=f"clips/{n:03d}.m4a",
            order=order[n],
        )
        for n, (seg, stratum, reason, other) in enumerate(picked)
    ]


# --- scoring ---

@dataclass
class Score:
    stratum: str
    n: int
    correct: int

    @property
    def accuracy(self) -> float:
        return self.correct / self.n if self.n else 0.0


def score(items: list[Item], labels: dict[str, str] | None = None) -> tuple[list[Score], dict[str, int]]:
    """Accuracy per stratum plus a confusion matrix, over the labelled items only.

    `labels` overrides the transcript's own label per item id — that is how a repaired
    transcript is scored against the same ground truth.
    """
    graded = [i for i in items if i.truth in GRADED]
    strata: dict[str, Score] = {}
    confusion: dict[str, int] = {}
    for item in graded:
        guess = (labels or {}).get(item.id, item.speaker)
        s = strata.setdefault(item.stratum, Score(item.stratum, 0, 0))
        s.n += 1
        s.correct += guess == item.truth
        if guess != item.truth:
            confusion[f"{item.truth} -> {guess}"] = confusion.get(f"{item.truth} -> {guess}", 0) + 1
    scores = sorted(strata.values(), key=lambda s: s.stratum)
    total = Score("ALL", sum(s.n for s in scores), sum(s.correct for s in scores))
    return scores + [total], confusion


def mcnemar(items: list[Item], before: dict[str, str], after: dict[str, str]) -> tuple[int, int, float]:
    """Paired comparison of two labelings on the same clips.

    Returns (fixed, broken, p). Only the disagreements carry information: of the
    fixed+broken lines where the two labelings differ, an exact two-sided binomial
    test asks whether the split is more lopsided than coin flips would give.
    """
    fixed = broken = 0
    for item in items:
        if item.truth not in GRADED:
            continue
        was = before.get(item.id, item.speaker) == item.truth
        now = after.get(item.id, item.speaker) == item.truth
        fixed += now and not was
        broken += was and not now
    n = fixed + broken
    if n == 0:
        return 0, 0, 1.0
    k = min(fixed, broken)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / (2 ** n)
    return fixed, broken, min(1.0, 2 * tail)
