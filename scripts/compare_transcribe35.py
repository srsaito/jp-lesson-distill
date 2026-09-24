"""Experiment (2026-09-24): Gemini 3.5 Transcribe vs Pass A (gemini-3.1-pro-preview).

Not part of the pipeline. Two questions, because a dedicated STT model can win one and
lose the other:
  1. WHO spoke — does acoustic diarization beat Pass A's inferred teacher/student labels?
  2. WHAT was said — does its `verbatim` mode keep Steven's errors, or "fix" them the way
     the repo's founding assumption says STT models do?

    uv run python scripts/compare_transcribe35.py fetch 20260923 20260916 20260817
    uv run python scripts/compare_transcribe35.py compare

fetch reuses the pipeline's cached window files and writes one raw response per window to
work/<date>/transcribe35/ (resumable: an existing file is skipped). Diarization is capped at
30 min per request, so the 20-min windows fit. The SDK (google-genai 2.11) drops the
word annotations, so this calls the Interactions REST endpoint directly.
"""

from __future__ import annotations

import json
import os
import re
import sys
import time
from collections import Counter
from pathlib import Path

import httpx
from google import genai

from jp_lesson_distill.gemini import _dotenv_key
from jp_lesson_distill.labeling import GRADED, similarity
from jp_lesson_distill.models import MomentsFile, Transcript, parse_ts

MODEL = "gemini-3.5-transcribe"
URL = "https://generativelanguage.googleapis.com/v1beta/interactions"
STRIDE = 20 * 60 - 30  # pipeline default: 20-min windows, 30 s overlap

WORK = Path("work")
SOURCES = {  # date -> (dir holding windows/ and Pass A output, Pass A transcript)
    "20260923": (WORK / "20260923", WORK / "20260923/transcript.json"),
    "20260916": (WORK / "20260916", WORK / "20260916/transcript.json"),
    "20260817": (WORK / "hc9-2-verify/20260817", WORK / "hc9-2-verify/20260817/transcript.json"),
}
TRUTH_20260817 = Path("tests/fixtures/diarization_truth_20260817.json")


def source(date: str) -> tuple[Path, Path]:
    """Explicit entries above, else the pipeline's own cache layout work/<date>/."""
    return SOURCES.get(date) or (WORK / date, WORK / date / "transcript.json")

# Address forms that pin a speaker (mirrors labeling.MARKERS, which only covers
# teacher/student labels, not spk ids).
TEACHER_CUES = ("奥さん",)
# 3.5T hears 「スティーブさん」 as 「スリーブさん」 (9/23 09:40), so match the name loosely.
NAME = re.compile(r"[スステ][ィーリ]{0,3}ブ(ン)?さん")
STUDENT_CUES = ("先生", "妻")


# --- fetch -------------------------------------------------------------------

def fetch(date: str) -> None:
    base, _ = source(date)
    out = base / "transcribe35"
    out.mkdir(exist_ok=True)
    # The repo's own project key (.env), as the pipeline uses — NOT the shell's, which is
    # FlashGen's project. The first run of this script (2026-09-24) billed FlashGen by mistake.
    key = _dotenv_key() or os.environ["GEMINI_API_KEY"]
    client = genai.Client(api_key=key)
    for w in sorted((base / "windows").glob("w*.m4a")):
        dest = out / f"{w.stem}.raw.json"
        if dest.exists():
            print(f"[{date}] {w.stem} cached")
            continue
        f = client.files.upload(file=str(w))
        while f.state and f.state.name == "PROCESSING":
            time.sleep(2)
            f = client.files.get(name=f.name)
        body = {"model": MODEL,
                "input": [{"type": "audio", "uri": f.uri, "mime_type": f.mime_type}],
                "generation_config": {"transcription_config": {
                    "language_codes": ["ja-JP", "en-US"],
                    "mode": {"type": "verbatim", "diarization_mode": "speaker",
                             "timestamp_granularities": ["word"]}}}}
        for attempt in range(12):
            t = time.time()
            r = httpx.post(URL, headers={"x-goog-api-key": key},
                           json=body, timeout=900)
            if r.status_code == 429:
                m = re.search(r"retry in (\d+)s", r.text)
                wait = int(m.group(1)) + 10 if m else 70
                print(f"[{date}] {w.stem} 429 (per-minute token cap), waiting {wait}s", flush=True)
                time.sleep(wait)
                continue
            r.raise_for_status()
            dest.write_text(json.dumps(r.json(), ensure_ascii=False))
            print(f"[{date}] {w.stem} ok in {time.time() - t:.1f}s", flush=True)
            break
        else:
            raise SystemExit(f"[{date}] {w.stem}: still rate-limited after 12 tries")


# --- parse -------------------------------------------------------------------

def words(date: str) -> list[dict]:
    """All word annotations, absolute time, speaker id qualified by window (w01:spk:0)."""
    base, _ = source(date)
    out = []
    for raw in sorted((base / "transcribe35").glob("w*.raw.json")):
        i = int(raw.stem[1:3])
        offset = (i - 1) * STRIDE
        # the pipeline splits each overlap at its midpoint; do the same
        lo = 0.0 if i == 1 else 15.0
        d = json.loads(raw.read_text())
        for step in d.get("steps", []):
            for c in step.get("content", []) or []:
                for a in c.get("annotations", []) or []:
                    if a.get("type") != "word_info":
                        continue
                    s = float(a["start_offset"].rstrip("s"))
                    if s < lo:
                        continue
                    out.append({"t": offset + s, "text": a["text"],
                                "spk": f"{raw.stem.split('.')[0]}:{a.get('speaker', '?')}"})
    out.sort(key=lambda w: w["t"])
    # drop overlap duplicates: keep the earlier window's words up to the next window's cut
    dedup, last = [], -1.0
    for w in out:
        if w["t"] >= last:
            dedup.append(w)
            last = w["t"]
    return dedup


def turns(ws: list[dict]) -> list[dict]:
    """Group consecutive same-speaker words into turns."""
    ts: list[dict] = []
    for w in ws:
        if ts and ts[-1]["spk"] == w["spk"] and w["t"] - ts[-1]["end"] < 3.0:
            ts[-1]["text"] += w["text"]
            ts[-1]["end"] = w["t"]
        else:
            ts.append({"start": w["t"], "end": w["t"], "spk": w["spk"], "text": w["text"]})
    return ts


def marker_mapping(ts: list[dict]) -> dict[str, str]:
    """spk id -> teacher/student from address forms alone (no ground truth, no Pass A)."""
    votes: dict[str, Counter] = {}
    for t in ts:
        v = votes.setdefault(t["spk"], Counter())
        v["teacher"] += sum(t["text"].count(c) for c in TEACHER_CUES) + len(NAME.findall(t["text"]))
        v["student"] += sum(t["text"].count(c) for c in STUDENT_CUES)
    mapping = {}
    for window in sorted({s.split(":")[0] for s in votes}):
        spks = sorted(s for s in votes if s.startswith(window))
        scored = {s: votes[s]["teacher"] - votes[s]["student"] for s in spks}
        if len(spks) >= 2 and any(scored.values()):
            order = sorted(spks, key=lambda s: -scored[s])
            mapping[order[0]] = "teacher"
            for s in order[1:]:
                mapping[s] = "student"
        # no cue in this window -> leave unmapped; reported as such
    return mapping


def best_mapping(ts, spks, truth):
    """Per window, whichever of the two teacher/student assignments agrees best with truth."""
    out = {}
    for window in sorted({s.split(":")[0] for s in spks}):
        ids = sorted(s for s in spks if s.startswith(window + ":"))
        best = None
        for flip in (False, True):
            m = {s: ("teacher" if (k == 0) ^ flip else "student") for k, s in enumerate(ids)}
            ok = sum(1 for t, lab in truth if m.get(speaker_at(ts, t)) == lab)
            if best is None or ok > best[0]:
                best = (ok, m)
        out.update(best[1])
    return out


def speaker_at(ts: list[dict], t0: float, span: float = 4.0) -> str | None:
    """Majority spk id over the words of [t0, t0+span]."""
    c = Counter()
    for t in ts:
        if t["end"] >= t0 and t["start"] <= t0 + span:
            c[t["spk"]] += 1
    return c.most_common(1)[0][0] if c else None


def passa_at(segs, t0: float) -> str | None:
    prev = None
    for s in segs:
        if parse_ts(s.start) > t0 + 1.0:
            break
        prev = s
    return prev.speaker if prev else None


# --- compare -----------------------------------------------------------------

# Content-derived attributions for 2026-09-23 (who went to Shizuoka / who had the biopsy),
# made on 2026-09-23 while reading the transcript. Claude's judgement, NOT blind ears.
CONTENT_TRUTH_20260923 = [
    (5, "teacher"), (20, "student"), (127, "teacher"), (180, "student"), (262, "teacher"),
    (378, "teacher"), (440, "student"), (570, "teacher"), (586, "student"), (667, "teacher"),
    (684, "student"), (706, "teacher"), (747, "student"), (812, "student"), (926, "teacher"),
    (1023, "teacher"), (1063, "student"), (1089, "teacher"), (1179, "student"), (1301, "student"),
    (1644, "teacher"), (1673, "student"), (1777, "student"), (2123, "teacher"), (2572, "student"),
    (2665, "student"), (2978, "student"), (3300, "student"), (3664, "student"),
]

ERRORS = {  # substrings that only exist if the error was transcribed as spoken
    "20260923": ["釣るの魚", "眠いの感じ", "新しいの", "重いなもの", "いない", "技術、お", "もっとくんな",
                 "数回のところ", "まだ逃げません", "普通のこと", "親さん"],
    "20260916": ["逃げれ", "待つる", "捨てれ", "着ります", "きれ", "集まろ", "やろう", "仕事にの", "暑すぎますと",
                 "通じました", "始めます"],
}


def compare() -> None:
    for date in ("20260817", "20260916", "20260923"):
        base, pa_path = source(date)
        if not (base / "transcribe35").exists():
            continue
        ws = words(date)
        ts = turns(ws)
        mapping = marker_mapping(ts)
        pa = Transcript.model_validate_json(pa_path.read_text()).segments
        spks = sorted({t["spk"] for t in ts})
        print(f"\n=== {date} ===")
        print(f"words {len(ws)}, turns {len(ts)} ({len(ts) / (ts[-1]['end'] / 60):.1f}/min) vs Pass A "
              f"{len(pa)} segments; speakers {Counter(t['spk'].split(':', 1)[1] for t in ts)}")
        print(f"marker mapping: {mapping or '{}'}; unmapped: {[s for s in spks if s not in mapping]}")
        latin_t35 = sum(len(re.findall(r'[A-Za-z]{3,}', t['text'])) for t in ts)
        latin_pa = sum(len(re.findall(r'[A-Za-z]{3,}', s.text)) for s in pa)
        print(f"English words kept: 3.5T {latin_t35} vs Pass A {latin_pa}")

        def t35_label(t0, oracle=None):
            s = speaker_at(ts, t0)
            if s is None:
                return None
            m = oracle if oracle is not None else mapping
            return m.get(s)

        if date == "20260817":
            items_raw = json.loads(TRUTH_20260817.read_text())["items"]
            graded = [r for r in items_raw if r["truth"] in GRADED]

            def acc(get):
                ok = n = 0
                for r in graded:
                    lab = get(r["t_start"])
                    n += 1
                    ok += lab == r["truth"]
                return ok, n
            # oracle: best of the two mappings per window, against the truth (an upper bound)
            oracle = best_mapping(ts, spks, [(r["t_start"], r["truth"]) for r in graded])
            pa_ok, n = acc(lambda t0: passa_at(pa, t0))
            print(f"vs 41 blind hand labels (graded n={n}; mixed/unsure excluded):")
            print(f"  Pass A 3.1 Pro        {pa_ok}/{n} = {pa_ok / n:.0%}")
            mk_ok, _ = acc(lambda t0: t35_label(t0))
            print(f"  3.5 Transcribe, marker-mapped  {mk_ok}/{n} = {mk_ok / n:.0%}")
            or_ok, _ = acc(lambda t0: t35_label(t0, oracle))
            print(f"  3.5 Transcribe, best mapping   {or_ok}/{n} = {or_ok / n:.0%}  (upper bound)")
            played = [r for r in items_raw if r["truth"] in ("played", "other")]
            for r in played:
                print(f"  truth={r['truth']} @{r['t_start']:.0f}s -> 3.5T spk {speaker_at(ts, r['t_start'])}")

        if date == "20260923":
            oracle = best_mapping(ts, spks, CONTENT_TRUTH_20260923)
            or_ok = sum(t35_label(t, oracle) == lab for t, lab in CONTENT_TRUTH_20260923)
            n = len(CONTENT_TRUTH_20260923)
            pa_ok = sum(passa_at(pa, t) == lab for t, lab in CONTENT_TRUTH_20260923)
            mk_ok = sum(t35_label(t) == lab for t, lab in CONTENT_TRUTH_20260923)
            print(f"vs {n} content-derived attributions (Claude's reading, not ears):")
            print(f"  Pass A 3.1 Pro   {pa_ok}/{n}")
            print(f"  3.5 Transcribe   {mk_ok}/{n} (marker-mapped)")
            print(f"  3.5 Transcribe   {or_ok}/{n} (best mapping, 1 choice per window)")
            for t, lab in CONTENT_TRUTH_20260923:
                a, b = passa_at(pa, t), t35_label(t, oracle)
                if a != lab or b != lab:
                    print(f"   {t // 60:02d}:{t % 60:02d} truth={lab:7} PassA={a} 3.5T={b}")

        if date in ERRORS:
            t35_text = "".join(t["text"] for t in ts)
            pa_text = "".join(s.text for s in pa)
            print("error strings present (3.5T / PassA):")
            for e in ERRORS[date]:
                print(f"  {e:10} {'✓' if e in t35_text else '✗'} / {'✓' if e in pa_text else '✗'}")
            mf = base / "moments.json"
            if mf.exists():
                moms = MomentsFile.model_validate_json(mf.read_text()).moments
                better = worse = 0
                for m in moms:
                    lo, hi = m.t_start - 20, m.t_end + 20
                    a = max((similarity(m.student_verbatim, s.text) for s in pa
                             if lo <= parse_ts(s.start) <= hi), default=0)
                    b = max((similarity(m.student_verbatim, t["text"]) for t in ts
                             if lo <= t["start"] <= hi), default=0)
                    better += b > a + 0.05
                    worse += b < a - 0.05
                print(f"  moments: 3.5T closer to Pass B verbatim on {better}, further on {worse}, "
                      f"of {len(moms)}")

        # agreement map with Pass A, per minute, where both have a label
        row = []
        for minute in range(int(ts[-1]["end"] // 60) + 1):
            t0 = minute * 60 + 30
            a, b = passa_at(pa, t0), t35_label(t0)
            row.append("." if b is None else ("=" if a == b else "X"))
        print("per-minute agreement with Pass A (= agree, X disagree, . unmapped):")
        print("  " + "".join(row))


# --- speakers: the question "is it consistently better at WHO spoke?" -------------

def passa_mapping(ts, pa) -> dict[str, str]:
    """Name each 3.5T speaker id by majority vote of Pass A's label at its words' times.

    Independent of the address forms, so those stay a fair test for both systems. It can
    only be as right as Pass A is on average over an id's words, which is the point: an id
    that Pass A mostly calls teacher is the teacher unless the cue lines say otherwise."""
    votes: dict[str, Counter] = {}
    for t in ts:
        lab = passa_at(pa, t["start"])
        if lab:
            votes.setdefault(t["spk"], Counter())[lab] += len(t["text"])
    return {k: v.most_common(1)[0][0] for k, v in votes.items()}


def implied(text: str) -> str | None:
    if NAME.search(text) or any(c in text for c in TEACHER_CUES):
        return "teacher"
    if any(c in text for c in STUDENT_CUES):
        return "student"
    return None


def speakers(dates: list[str]) -> None:
    for date in dates:
        base, pa_path = source(date)
        if not (base / "transcribe35").exists():
            continue
        ts = turns(words(date))
        pa = Transcript.model_validate_json(pa_path.read_text()).segments
        m = passa_mapping(ts, pa)
        per_window = Counter(t["spk"].split(":")[0] for t in {t["spk"]: t for t in ts}.values())
        # dropped speech: a >45 s hole in 3.5T where Pass A has words
        drops = []
        for a, b in zip(ts, ts[1:]):
            if b["start"] - a["end"] > 45:
                chars = sum(len(s.text) for s in pa if a["end"] < parse_ts(s.start) < b["start"])
                if chars > 80:
                    drops.append(f"{int(a['end'] // 60)}:{int(a['end'] % 60):02d}+{b['start'] - a['end']:.0f}s({chars}ch)")
        # address-form lines, each system scored on its own text
        pa_cue = [(s.speaker, implied(s.text)) for s in pa if implied(s.text)]
        t_cue = [(m.get(t["spk"]), implied(t["text"])) for t in ts if implied(t["text"])]
        pa_ok = sum(a == b for a, b in pa_cue)
        t_ok = sum(a == b for a, b in t_cue)
        # disagreement sampled every 10 s where 3.5T has speech
        samples = []
        for t0 in range(0, int(ts[-1]["end"]), 10):
            sid = speaker_at(ts, t0, span=3)
            if sid and m.get(sid):
                samples.append((t0, passa_at(pa, t0), m[sid]))
        dis = [x for x in samples if x[1] != x[2]]
        print(f"\n=== {date} === speaker ids per window {dict(per_window)}")
        print(f"  dropped speech: {drops or 'none'}")
        print(f"  address-form lines right: Pass A {pa_ok}/{len(pa_cue)}   3.5T {t_ok}/{len(t_cue)}")
        print(f"  disagree on {len(dis)}/{len(samples)} 10-s samples ({len(dis) / max(1, len(samples)):.0%})")
        # runs of >=3 consecutive disagreeing samples = a stretch worth judging by content
        run: list = []
        for x in samples + [(None, None, None)]:
            if x[0] is not None and x[1] != x[2] and (not run or x[0] - run[-1][0] <= 10):
                run.append(x)
                continue
            if len(run) >= 3:
                a0, a1 = run[0][0], run[-1][0] + 10
                print(f"  -- stretch {a0 // 60}:{a0 % 60:02d}-{a1 // 60}:{a1 % 60:02d}")
                for s_ in pa:
                    st = parse_ts(s_.start)
                    if a0 - 5 <= st <= a1:
                        print(f"     PA  {s_.start} {s_.speaker:7} {s_.text[:70]}")
                for t in ts:
                    if a0 - 5 <= t["start"] <= a1:
                        print(f"     35T {int(t['start'] // 60):02d}:{int(t['start'] % 60):02d} {m.get(t['spk'], '?'):7} {t['text'][:70]}")
            run = [x] if x[0] is not None and x[1] != x[2] else []


if __name__ == "__main__":
    cmd, *args = sys.argv[1:]
    if cmd == "fetch":
        for d in args:
            fetch(d)
    elif cmd == "compare":
        compare()
    elif cmd == "speakers":
        speakers(args)
