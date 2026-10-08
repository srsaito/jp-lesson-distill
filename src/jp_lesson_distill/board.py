"""Board timeline (jld-333): read the shared screen of a lesson video into timestamped notes.

Soso先生 types vocabulary and corrections on the shared screen during class and
sends the レジュメ a day or more later. This reads the screen straight from the
Zoom recording: which slide was up when, and which text appeared when (ADR-0008).

`read` (slow, once, local) looks at one frame a second, runs macOS Vision text
recognition whenever the screen has changed and then held still, and writes
work/<date>/board/ocr.jsonl plus the frames it read. `build` (quick, re-runnable)
turns that into board.json and board.md: one section per slide, the slide's own
text, then what was typed onto it and when. With Gemini on, each slide's final
frame is also transcribed once and cached — Vision cannot read kanji under
furigana or pen strokes. `run_board` is both, as `distill run` and `distill board`
call it.
"""
from __future__ import annotations

import json
import re
import subprocess
from difflib import SequenceMatcher
from pathlib import Path

VIDEO_SUFFIXES = (".mp4", ".mov", ".mkv", ".webm")

SCREEN_W, SCREEN_H = 1920, 1080  # the shared screen; recordings with thumbnails are wider
DOWN = 4  # compare frames at 480x270
PIXEL_DELTA = 24  # grey levels before a pixel counts as changed
CHANGED = 0.0004  # share of pixels: about one typed character
STILL = 0.0002  # ...and less than this between consecutive seconds counts as holding still
CUT = 0.15  # share of pixels that makes it a different slide
MIN_GAP_S = 3  # between two reads, so a wandering pointer can't flood the reader
BLOCK = 30  # px at the compared size: the grid for spotting webcam tiles
BUSY = 0.35  # a block that moved in this share of recent seconds is video, not board

FURIGANA_MAX_H = 0.029  # of frame height; typed and slide body text is taller
KANA_ONLY = re.compile(r"^[぀-ヿー\s]+$")
WORDY = re.compile(r"[぀-ヿ一-鿿A-Za-z0-9]")
WEB = re.compile(r"www\.|https?:|\.(com|jp|org|net)/|Google ?(マップ|Maps|検索)|Japanese Lesson \d+")  # browser chrome
EMIT_W = 1280  # px: slide images for the vault are for reading, and the vault is in git
PASSING_S = 3  # a slide on screen no longer than this, with nothing typed on it, was only passed through


# --- read: video -> ocr.jsonl -------------------------------------------------

def _frames(video: Path):
    import imageio_ffmpeg
    import numpy as np

    cmd = [
        imageio_ffmpeg.get_ffmpeg_exe(), "-loglevel", "error", "-i", str(video),
        "-vf", f"fps=1,crop='min(iw,{SCREEN_W})':'min(ih,{SCREEN_H})':0:0,scale={SCREEN_W}:{SCREEN_H}",
        "-pix_fmt", "rgb24", "-f", "rawvideo", "-",
    ]
    size = SCREEN_W * SCREEN_H * 3
    with subprocess.Popen(cmd, stdout=subprocess.PIPE) as proc:
        while True:
            buf = proc.stdout.read(size)
            if len(buf) < size:
                return
            yield np.frombuffer(buf, np.uint8).reshape(SCREEN_H, SCREEN_W, 3)


def _is_red(frame, x: float, y: float, w: float, h: float) -> bool:
    """Is the ink in this box red? Soso先生 fills in answers and corrections in red."""
    import numpy as np

    box = frame[int(y * SCREEN_H):int((y + h) * SCREEN_H) + 1, int(x * SCREEN_W):int((x + w) * SCREEN_W) + 1].astype(np.int16)
    if box.size == 0:
        return False
    flat = box.reshape(-1, 3)
    ink = np.abs(flat - np.median(flat, axis=0)).sum(axis=1) > 120
    if ink.sum() < 20:
        return False
    r, g, b = flat[ink].T
    return float(((r > g + 60) & (r > b + 60)).mean()) > 0.5


def _ocr(frame) -> list[dict]:
    from ocrmac import ocrmac
    from PIL import Image

    found = ocrmac.OCR(Image.fromarray(frame), language_preference=["ja-JP", "en-US"],
                       recognition_level="accurate").recognize()
    lines = []
    for text, conf, (x, y_bottom, w, h) in found:
        y = 1 - y_bottom - h  # Vision measures from the bottom edge
        lines.append({"text": text, "conf": round(conf, 2), "x": round(x, 4), "y": round(y, 4),
                      "w": round(w, 4), "h": round(h, 4), "red": _is_red(frame, x, y, w, h)})
    return sorted(lines, key=lambda l: (l["y"], l["x"]))


def read(video: Path, date: str, work_dir: Path) -> None:
    import numpy as np
    from PIL import Image

    out = work_dir / date / "board"
    (out / "frames").mkdir(parents=True, exist_ok=True)

    def small(frame):
        return frame[::DOWN, ::DOWN].mean(axis=2).astype(np.int16)

    # Some recordings lay the two webcam tiles over a corner of the shared screen
    # (2026-10-05) instead of beside it. A face moves nearly every second, so the
    # screen would never count as still. Blocks that keep moving are left out.
    bh, bw = SCREEN_H // DOWN // BLOCK, SCREEN_W // DOWN // BLOCK
    busy = np.zeros((bh, bw))
    live = np.ones((bh * BLOCK, bw * BLOCK), bool)  # pixels that count

    def moved(a, b) -> float:
        return float((np.abs(a - b) > PIXEL_DELTA)[live].mean())

    def note_motion(a, b) -> None:
        nonlocal busy, live
        hit = (np.abs(a - b) > PIXEL_DELTA).reshape(bh, BLOCK, bw, BLOCK).mean(axis=(1, 3)) > 0.01
        busy = 0.97 * busy + 0.03 * hit
        live = np.kron(busy < BUSY, np.ones((BLOCK, BLOCK), bool)).astype(bool)

    records: list[dict] = []

    def take(t: int, frame, cut: bool) -> None:
        lines = []
        for l in _ocr(frame):
            cy = min(int((l["y"] + l["h"] / 2) * bh), bh - 1)
            cx = min(int((l["x"] + l["w"] / 2) * bw), bw - 1)
            if busy[cy, cx] < BUSY:  # not a name label sitting on a webcam tile
                lines.append(l)
        records.append({"t": t, "cut": cut, "lines": lines})
        Image.fromarray(frame).save(out / "frames" / f"{t:05d}.jpg", quality=70)

    prev = prev_small = ref_small = None
    prev_still, last_read = False, -MIN_GAP_S
    prev_read = True  # was the previous frame the one we read last?
    for t, frame in enumerate(_frames(video)):
        s = small(frame)
        if ref_small is None:
            take(t, frame, True)
            ref_small, last_read, prev_read = s, t, True
        else:
            note_motion(s, prev_small)
            step = moved(s, prev_small)
            if step > CUT and not prev_read and moved(prev_small, ref_small) > CHANGED:
                # the slide is about to change: keep the last look at the old one
                take(t - 1, prev, False)
                ref_small, last_read = prev_small, t - 1
            still = step < STILL
            drift = moved(s, ref_small)
            prev_read = False
            if still and prev_still and drift > CHANGED and t - last_read >= MIN_GAP_S:
                take(t, frame, drift > CUT)
                ref_small, last_read, prev_read = s, t, True
            prev_still = still
        prev, prev_small = frame, s
        if t % 300 == 0:
            print(f"[board]   {t // 60:3d} min, {len(records)} frames read", flush=True)
    if prev is not None and not prev_read:
        take(t, prev, False)
    (out / "ocr.jsonl").write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in records) + "\n")
    print(f"[board] {len(records)} frames read over {t // 60} min -> {out / 'ocr.jsonl'}")


# --- build: ocr.jsonl -> board.json, board.md ----------------------------------

def _furigana(line: dict) -> bool:
    return line["h"] < FURIGANA_MAX_H and bool(KANA_ONLY.match(line["text"]))


def _same_text(a: str, b: str) -> bool:
    return a == b or SequenceMatcher(None, a, b).ratio() >= 0.8


def _find(line: dict, tracks: list[dict]) -> dict | None:
    """The line this is another look at: same place on the screen, or the same words moved."""
    for tr in tracks:
        if abs(tr["y"] - line["y"]) < 0.015 and abs(tr["x"] - line["x"]) < 0.03:
            return tr
    for tr in tracks:
        if len(line["text"]) >= 4 and _same_text(tr["text"], line["text"]):
            return tr
    return None


def _overlap(texts: list[str], lines: list[dict]) -> float:
    """Share of `texts` that are still on screen in `lines`."""
    if not texts:
        return 0.0
    now = [l["text"] for l in lines]
    return sum(any(_same_text(t, n) for n in now) for t in texts) / len(texts)


def _stamp(t: float) -> str:
    return f"{int(t) // 60:02d}:{int(t) % 60:02d}"


_client = None


def _gemini_slide(out: Path, t: int) -> list[dict]:
    """Gemini's reading of one slide, cached.

    Vision garbles kanji that sit under furigana (地震 -> 地態) and mostly misses
    pen strokes. Every slide goes through, not only the furigana-heavy ones: the
    free-talk board has no furigana and is where he draws.
    """
    cache = out / "slides" / f"{t:05d}.json"
    if cache.exists():
        return json.loads(cache.read_text())
    from . import gemini, prompts
    from .models import SlideText

    global _client
    if _client is None:
        _client = gemini.make_client()
    image = gemini.image_part(out / "frames" / f"{t:05d}.jpg")
    # LOW: this is reading, not deciding who spoke, and a slide's thinking would cost more than its answer
    got = gemini.generate(_client, gemini.DEFAULT_MODEL, [image, prompts.SLIDE], SlideText, thinking_level="LOW")
    lines = [l.model_dump() for l in got.lines]
    cache.parent.mkdir(exist_ok=True)
    cache.write_text(json.dumps(lines, ensure_ascii=False, indent=1))
    return lines


def _emit(date: str, out: Path, board: dict, md: str, out_dir: Path) -> None:
    """Put the board next to moments.json: one note, and only the slide images it shows, made smaller."""
    from PIL import Image

    images = out_dir / f"{date}-board"
    images.mkdir(parents=True, exist_ok=True)
    for sc in board["scenes"]:
        name = Path(sc["final_frame"]).name
        im = Image.open(out / "frames" / name)
        im.resize((EMIT_W, im.height * EMIT_W // im.width), Image.LANCZOS).save(images / name, quality=70)
    (out_dir / f"{date}-board.md").write_text(md.replace("](frames/", f"]({date}-board/"))
    size = sum(f.stat().st_size for f in images.iterdir()) / 1e6
    print(f"[emit] {out_dir / f'{date}-board.md'} with {len(board['scenes'])} images ({size:.1f} MB)")


def build(date: str, work_dir: Path, min_seen: int = 2, use_gemini: bool = False,
          out_dir: Path | None = None) -> dict:
    """`min_seen` drops text read in fewer frames than that: one-frame flicker, not writing."""
    out = work_dir / date / "board"
    records = [json.loads(l) for l in (out / "ocr.jsonl").read_text().splitlines() if l.strip()]
    scenes: list[dict] = []
    current: dict | None = None
    last_t = 0
    for rec in records:
        lines = [l for l in rec["lines"] if not _furigana(l) and l["text"].strip()]
        t = rec["t"]
        on_screen = [tr["text"] for tr in current["tracks"] if tr["last_seen"] == last_t] if current else []
        kept = _overlap(on_screen, lines)
        new = current is None or (rec["cut"] and kept < 0.6) or (len(on_screen) >= 4 and kept < 0.3)
        if new:
            if current:
                current["visits"][-1][1] = last_t
            # back to a slide we have seen before? Then carry on with it: the board
            # he returns to still holds everything typed on it earlier.
            back = None
            for sc in scenes:
                was = [tr["text"] for tr in sc["tracks"] if tr["seen"] >= min_seen or tr["slide"]]
                if len(was) >= 3 and _overlap(was, lines) >= 0.6:
                    back = sc
                    break
            if back:
                current = back
                current["visits"].append([t, t])
            else:
                current = {"id": len(scenes) + 1, "visits": [[t, t]], "tracks": [], "first": t}
                scenes.append(current)
        first_look = t == current["first"]
        for l in lines:
            tr = _find(l, current["tracks"])
            if tr is None:
                current["tracks"].append({**l, "t_first": t, "t_changed": t, "last_seen": t, "seen": 1,
                                          "slide": first_look, "looks": [(l["text"], l["conf"])]})
                continue
            if tr["last_seen"] != t:
                tr["seen"] += 1
            tr["last_seen"] = t
            if l["text"] != tr["text"]:
                tr["looks"].append((l["text"], l["conf"]))
                # a line being typed gets longer; a line read badly through the pointer gets less sure
                if len(l["text"]) > len(tr["text"]) or l["conf"] >= tr["conf"]:
                    if not _same_text(tr["text"], l["text"]) or len(l["text"]) != len(tr["text"]):
                        tr["t_changed"] = t
                    tr.update(text=l["text"], conf=l["conf"], red=l["red"] or tr["red"])
        current["visits"][-1][1] = t
        last_t = t
    if current:
        current["visits"][-1][1] = last_t

    # Text that sits at one spot on most slides is not on the slides: it is laid
    # over them (the Zoom name labels, when the webcam tiles cover a corner).
    # Matched loosely and by place: a 4-character name is read three different ways.
    short = [(sc["id"], tr) for sc in scenes for tr in sc["tracks"] if len(tr["text"]) <= 12]

    def kin(a: dict, b: dict) -> bool:
        return (abs(a["x"] - b["x"]) < 0.04 and abs(a["y"] - b["y"]) < 0.04
                and SequenceMatcher(None, a["text"], b["text"]).ratio() >= 0.5)

    overlay = []
    for _, tr in short:
        on = {sid for sid, other in short if kin(tr, other)}
        if len(on) >= 4 and len(on) > len(scenes) / 2:
            overlay.append(tr)
    for sc in scenes:
        sc["tracks"] = [tr for tr in sc["tracks"] if not any(tr is o for o in overlay)]

    board = {"date": date, "scenes": []}
    md = [f"# Board timeline {date}", "",
          "Read from the shared screen of the lesson recording. Each image is a slide as Soso先生 left it,",
          "which is what the レジュメ page will be. Timed lines come from on-device text recognition —",
          "expect the odd misread character. Times are minutes:seconds into the recording. **Bold** = red ink.", ""]
    events = []
    for sc in scenes:
        tracks = [tr for tr in sc["tracks"] if tr["seen"] >= min_seen or tr["last_seen"] == sc["visits"][-1][1]]
        slide = sorted((tr for tr in tracks if tr["slide"]), key=lambda tr: (tr["y"], tr["x"]))
        typed = []
        for tr in sorted((tr for tr in tracks if not tr["slide"]), key=lambda tr: (tr["t_first"], tr["y"])):
            # stray marks (a lone bracket, the speaker icon) and a line he moved or we met again after a scroll
            if len(WORDY.findall(tr["text"])) >= 2 and not any(_same_text(tr["text"], seen["text"]) for seen in typed):
                typed.append(tr)
        shown_s = sum(b - a for a, b in sc["visits"])
        if (not typed and (shown_s <= PASSING_S or len(slide) < 3)) or (len(slide) + len(typed) < 3 and shown_s < 30):
            continue  # flicked past on the way to another slide, or only the Zoom name tiles
        title = next((tr["text"] for tr in slide + typed if len(tr["text"]) >= 4), (slide + typed)[0]["text"])[:40]
        shown = ", ".join(f"{_stamp(a)}–{_stamp(b)}" for a, b in sc["visits"])
        # the slide as he left it — which is what the レジュメ page will be
        final = f"frames/{sc['visits'][-1][1]:05d}.jpg"
        board["scenes"].append({
            "id": sc["id"], "title": title, "visits": sc["visits"], "final_frame": final,
            "slide": [{"text": tr["text"], "red": tr["red"]} for tr in slide],
            "typed": [{"t": tr["t_first"], "t_done": tr["t_changed"], "text": tr["text"], "red": tr["red"]} for tr in typed],
        })
        if any(WEB.search(tr["text"]) for tr in sc["tracks"]):
            # he is showing a web page or a map: worth knowing it happened, not worth transcribing
            board["scenes"][-1].update(kind="web", slide=[], typed=[])
            md += [f"## {sc['id']}. (web page) {title}", f"*on screen {shown}*", "", f"![]({final})", ""]
            continue
        t_final = sc["visits"][-1][1]
        if use_gemini:
            read = _gemini_slide(out, t_final)
            printed = [l["text"] for l in read if l["kind"] == "printed"]
            title = next((x for x in printed if len(x) >= 4), title)[:40]
            board["scenes"][-1].update(title=title, reader="gemini", slide=[{"text": x, "red": False} for x in printed],
                                       notes=[l["text"] for l in read if l["kind"] == "typed"],
                                       handwritten=[l["text"] for l in read if l["kind"] == "handwritten"])
            md += [f"## {sc['id']}. {title}", f"*on screen {shown}*", "", f"![]({final})", ""]
            for head, kind in [("On the slide:", "printed"), ("Soso先生's notes on it, as he left them:", "typed"),
                               ("Handwritten:", "handwritten")]:
                rows = [l["text"] for l in read if l["kind"] == kind]
                if rows:
                    md += [head, ""] + [f"- {x}" for x in rows] + [""]
        else:
            md += [f"## {sc['id']}. {title}", f"*on screen {shown}*", "", f"![]({final})", ""]
            if slide:
                md += ["On the slide:", ""] + [f"- {'**' + tr['text'] + '**' if tr['red'] else tr['text']}" for tr in slide] + [""]
        if typed:
            md += ["Added during class:", ""]
            md += [f"- `{_stamp(tr['t_first'])}` {'**' + tr['text'] + '**' if tr['red'] else tr['text']}" for tr in typed] + [""]
        events += [(tr["t_first"], sc["id"], tr["text"]) for tr in typed]
    (out / "board.json").write_text(json.dumps(board, ensure_ascii=False, indent=1))
    (out / "board.md").write_text("\n".join(md) + "\n")
    print(f"[board] {len(board['scenes'])} slides, {len(events)} lines added during class, from {len(records)} frames")
    print(f"[board] {out / 'board.md'}")
    if out_dir:
        _emit(date, out, board, "\n".join(md) + "\n", out_dir)
    return board


def notes(board: dict) -> list[dict]:
    """Every line added to a slide during class, in time order: `{"t": seconds, "text": …}`.

    This is what detect and Pass B are shown. The time comes from Vision, which saw the
    line appear; where Gemini read the same line off the slide's final frame its text is
    used instead, because Vision misreads characters (お義母さん came out お茶母さん).
    """
    out = []
    for sc in board["scenes"]:
        if sc.get("kind") == "web":
            continue
        clean = [*sc.get("notes", []), *sc.get("handwritten", []), *(l["text"] for l in sc.get("slide", []))]
        for line in sc["typed"]:
            squeezed = re.sub(r"\s", "", line["text"])
            best = max(clean, key=lambda c: SequenceMatcher(None, squeezed, re.sub(r"\s", "", c)).ratio(), default=None)
            if best and SequenceMatcher(None, squeezed, re.sub(r"\s", "", best)).ratio() >= 0.6:
                out.append({"t": line["t"], "text": best})
            else:
                out.append({"t": line["t"], "text": line["text"]})
    return sorted(out, key=lambda n: n["t"])


def find_video(recording: Path) -> Path | None:
    """The lesson's video: the recording itself, or `…_class_video.mp4` beside `…_class_audio.m4a`."""
    if recording.suffix.lower() in VIDEO_SUFFIXES:
        return recording
    stem = recording.stem.replace("_audio", "_video")
    for suffix in VIDEO_SUFFIXES:
        candidate = recording.with_name(stem + suffix)
        if candidate != recording and candidate.exists():
            return candidate
    return None


def run_board(video: Path, date: str, work_dir: Path, out_dir: Path | None, use_gemini: bool = True) -> dict:
    """Read the screen (skipped when already read) and build the board. Delete ocr.jsonl to re-read."""
    if (work_dir / date / "board" / "ocr.jsonl").exists():
        print(f"[board] screen already read for {date} — rebuilding from cache")
    else:
        print(f"[board] reading the shared screen of {video.name} (one frame a second; 10–20 min for an hour)")
        read(video, date, work_dir)
    return build(date, work_dir, use_gemini=use_gemini, out_dir=out_dir)
