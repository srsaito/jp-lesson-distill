"""jld-b18: are the moments real? A listening kit over one run's moments, judged by ear.

Pass B confirmed every candidate it was given on 10/26 (20 of 20) and on the extra detect runs
(40 of 40), so nothing in the pipeline tells a real moment from a false one. This measures it.
Each item plays the clip Pass B heard (`clips/mNN.m4a`) and shows what the moment claims: the
student's words and the correction. The listener does not see which detect run found it; the
score breaks the answers down by that, and flags pairs that look like the same exchange.

  uv run python scripts/moment_kit.py kit   --work work/e2e-1026/20261026
  uv run python scripts/moment_kit.py play  --work work/e2e-1026/20261026
  uv run python scripts/moment_kit.py score --work work/e2e-1026/20261026
"""

from __future__ import annotations

import argparse
import json
import random
import subprocess
from collections import Counter
from pathlib import Path

from jp_lesson_distill.labeling import similarity
from jp_lesson_distill.models import CandidateList, MomentsFile, fmt_ts
from jp_lesson_distill.pipeline import same_candidate

KEYS = {
    "r": "real",       # he made this error (or this really is new to him), and the words are right
    "w": "wrong-words",  # a real moment, but the text is not what he said
    "f": "false",      # no error there, or nothing worth a card
    "?": "unsure",
}
PROMPT = """
For each clip, is the moment real?
  [r] real: you said this (errors and all), and it is worth a card
  [w] real moment, but the words shown are not what you said
  [f] false: there is no error here, or it is not worth a card
  [?] can't tell        [p] replay   [b] back   [q] save and quit
A note can follow the key after a space, e.g.  w I said 旅行とき, not 旅行の時
"""


def kit(work: Path, seed: int) -> None:
    moments = MomentsFile.model_validate_json((work / "moments.json").read_text()).moments
    ordered = sorted(CandidateList.model_validate_json((work / "candidates.json").read_text()).candidates,
                     key=lambda c: c.priority, reverse=True)
    runs = sorted(work.glob("candidates_run*.json"))
    found_by = [CandidateList.model_validate_json(p.read_text()).candidates for p in runs]
    items = []
    for m in moments:
        n = int(m.id[1:])
        cand = ordered[n - 1]
        which = [i + 1 for i, run in enumerate(found_by) if any(same_candidate(cand, c) for c in run)]
        items.append({"id": m.id, "t": m.t_start, "clip": f"clips/{m.id}.m4a", "type": m.type,
                      "said": m.student_verbatim, "correction": m.teacher_correction,
                      "explanation": m.explanation, "runs": which, "answer": None, "note": ""})
    random.Random(seed).shuffle(items)
    out = work / "moment_kit.jsonl"
    if out.exists():
        raise SystemExit(f"{out} exists; delete it to cut a new kit (it may hold answers)")
    out.write_text("".join(json.dumps(i, ensure_ascii=False) + "\n" for i in items))
    print(f"{len(items)} moments in {out}; run `play` to listen")


def _load(work: Path) -> list[dict]:
    return [json.loads(x) for x in (work / "moment_kit.jsonl").read_text().splitlines()]


def _save(work: Path, items: list[dict]) -> None:
    (work / "moment_kit.jsonl").write_text("".join(json.dumps(i, ensure_ascii=False) + "\n" for i in items))


def play(work: Path) -> None:
    items = _load(work)
    todo = [i for i in items if not i["answer"]]
    print(f"{len(todo)} of {len(items)} left.{PROMPT}")
    pos = 0
    while 0 <= pos < len(todo):
        it = todo[pos]
        print(f"\n[{pos + 1}/{len(todo)}] {fmt_ts(it['t'])}  ({it['type']})")
        print(f"   you said:   {it['said']}")
        if it["correction"]:
            print(f"   correction: {it['correction']}")
        print(f"   why:        {it['explanation']}")
        while True:
            subprocess.run(["afplay", str(work / it["clip"])], check=False)
            ans = input("   real? ").strip()
            key, _, note = ans.partition(" ")
            if key == "p":
                continue
            if key == "q":
                _save(work, items)
                print("saved")
                return
            if key == "b":
                pos = max(0, pos - 1)
                break
            if key in KEYS:
                it["answer"], it["note"] = KEYS[key], note
                _save(work, items)
                pos += 1
                break
            print("   r / w / f / ? / p / b / q")
    _save(work, items)
    print("\ndone. `score` tallies it.")


def score(work: Path) -> None:
    items = _load(work)
    done = [i for i in items if i["answer"]]
    print(f"{len(done)}/{len(items)} answered: {dict(Counter(i['answer'] for i in done))}")
    first = [i for i in done if 1 in i["runs"]]
    later = [i for i in done if 1 not in i["runs"]]
    for name, group in (("found by run 1", first), ("found only by later runs", later)):
        print(f"  {name}: {dict(Counter(i['answer'] for i in group))}")
    for i in sorted(done, key=lambda i: i["t"]):
        if i["note"] or i["answer"] != "real":
            print(f"   {fmt_ts(i['t'])} {i['id']} {i['answer']:11s} {i['said'][:30]} | {i['note']}")
    # pairs that look like one exchange: close in time, similar words
    for a in items:
        for b in items:
            if a["id"] < b["id"] and abs(a["t"] - b["t"]) <= 60 and similarity(a["said"], b["said"]) >= 0.3:
                print(f"   possible duplicate: {a['id']} / {b['id']} at {fmt_ts(a['t'])}")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("stage", choices=["kit", "play", "score"])
    p.add_argument("--work", type=Path, required=True)
    p.add_argument("--seed", type=int, default=18)
    a = p.parse_args()
    {"kit": lambda: kit(a.work, a.seed), "play": lambda: play(a.work), "score": lambda: score(a.work)}[a.stage]()


if __name__ == "__main__":
    main()
