"""Spike (jld-2nl): can the Zoom thumbnails say who is speaking?

Recordings from 2026-09-23 on are 2160x1080 with a 240-px strip on the right
holding both faces. This crops the two tiles, runs MediaPipe's face landmarker
on each, and reads how far the jaw is open in every frame. Talking shows up as
the jaw opening and closing, so the score is how much that value varies over
about a second. No API calls.

    uv run --no-project --python 3.12 --with "mediapipe==0.10.14" --with "numpy<2" --with imageio-ffmpeg \
        python scripts/spike_speaker_track.py extract <video.mp4> --date 20260923
    uv run --no-project --python 3.12 --with numpy \
        python scripts/spike_speaker_track.py score --date 20260923

MediaPipe is pinned: newer releases abort at start-up in a shell with no GPU
service ("DrishtiMetalHelper ... Service is unavailable"), whatever delegate is asked for.

`extract` writes work/<date>/video_spike/jaw.npz (slow, once);
`score` compares against Pass A's labels segment by segment (fast, iterate).

The model file is Google's face_landmarker.task (3.7 MB), expected at
~/.cache/jp-lesson-distill/face_landmarker.task:
https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task

A first version used OpenCV's Haar face detector and frame differences in a
mouth box. It scored at chance: the detector lost Steven's face on half the
frames (headset, turned head) and the box jittered more than a mouth moves.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

WORK = Path(__file__).resolve().parent.parent / "work"
MODEL = Path.home() / ".cache/jp-lesson-distill/face_landmarker.task"

SRC_FPS = 25
STEP = 2  # analyse every 2nd frame
FPS = SRC_FPS / STEP
# The thumbnail strip, measured on the 2026-09-23/24/28 recordings.
STRIP_X, TILE_W, TILE_H = 1920, 240, 135
TILES = ("student", "teacher")  # top tile is Steven, bottom is Soso先生
UPSCALE = 4  # faces are ~45 px wide in the source


def _frames(video: Path, start: float, duration: float | None):
    import imageio_ffmpeg

    w, h = TILE_W * UPSCALE, TILE_H * 2 * UPSCALE
    cmd = [imageio_ffmpeg.get_ffmpeg_exe(), "-loglevel", "error", "-ss", str(start), "-i", str(video)]
    if duration:
        cmd += ["-t", str(duration)]
    cmd += [
        "-vf", f"crop={TILE_W}:{TILE_H * 2}:{STRIP_X}:0,fps={FPS},scale={w}:{h}:flags=bicubic",
        "-pix_fmt", "rgb24", "-f", "rawvideo", "-",
    ]
    size = w * h * 3
    with subprocess.Popen(cmd, stdout=subprocess.PIPE) as proc:
        while True:
            buf = proc.stdout.read(size)
            if len(buf) < size:
                return
            yield np.frombuffer(buf, np.uint8).reshape(h, w, 3)


def extract(video: Path, date: str, start: float, duration: float | None) -> None:
    import mediapipe as mp
    from mediapipe.tasks.python import BaseOptions, vision

    def landmarker():
        return vision.FaceLandmarker.create_from_options(vision.FaceLandmarkerOptions(
            # CPU on purpose: the Metal path aborts when there is no GPU service (background shells)
            base_options=BaseOptions(model_asset_path=str(MODEL), delegate=BaseOptions.Delegate.CPU),
            running_mode=vision.RunningMode.VIDEO, num_faces=1, output_face_blendshapes=True,
            min_face_detection_confidence=0.3, min_face_presence_confidence=0.3,
        ))

    marks = [landmarker(), landmarker()]
    jaw, gap, found = [], [], []
    th = TILE_H * UPSCALE
    for n, frame in enumerate(_frames(video, start, duration)):
        row_j, row_g, row_f = [], [], []
        for i in range(2):
            tile = np.ascontiguousarray(frame[i * th:(i + 1) * th])
            res = marks[i].detect_for_video(mp.Image(image_format=mp.ImageFormat.SRGB, data=tile), int(n * 1000 / FPS))
            if res.face_landmarks:
                lm = res.face_landmarks[0]
                shapes = {b.category_name: b.score for b in res.face_blendshapes[0]}
                # inner-lip gap over face height, so it doesn't depend on how close the face is
                row_g.append(abs(lm[14].y - lm[13].y) / (abs(lm[152].y - lm[10].y) + 1e-6))
                row_j.append(shapes.get("jawOpen", 0.0))
                row_f.append(True)
            else:
                row_g.append(np.nan), row_j.append(np.nan), row_f.append(False)
        jaw.append(row_j), gap.append(row_g), found.append(row_f)
        if n % int(FPS * 300) == 0:
            print(f"  {n / FPS / 60:5.1f} min", flush=True)
    out = WORK / date / "video_spike"
    out.mkdir(parents=True, exist_ok=True)
    name = "jaw.npz" if not start and not duration else f"jaw_{int(start)}.npz"
    np.savez_compressed(out / name, jaw=np.array(jaw), gap=np.array(gap), found=np.array(found), start=start)
    f = np.array(found)
    print(f"{len(jaw)} frames; face found on {f[:, 0].mean():.0%} (student) / {f[:, 1].mean():.0%} (teacher)")
    print(f"wrote {out / name}")


def _seconds(stamp: str) -> int:
    parts = [int(p) for p in stamp.split(":")]
    return sum(p * 60 ** k for k, p in enumerate(reversed(parts)))


def activity(x: np.ndarray, win_s: float = 1.0) -> np.ndarray:
    """Per-frame talking score per tile: how much the mouth opening changes over `win_s`.

    Frames with no face count as no movement, so a lost face can't win a segment.
    """
    d = np.abs(np.diff(x, axis=0, prepend=x[:1]))
    d = np.nan_to_num(d, nan=0.0)
    k = max(int(win_s * FPS), 1)
    kernel = np.ones(k) / k
    sm = np.column_stack([np.convolve(d[:, i], kernel, mode="same") for i in range(2)])
    # each person's own scale: divide by their typical movement
    return sm / (np.percentile(sm, 75, axis=0) + 1e-9)


def _content_truth(date: str) -> list[tuple[int, str]]:
    """The hand list in compare_transcribe35.py, read as text (that script imports the API client)."""
    import ast
    import re

    src = (Path(__file__).parent / "compare_transcribe35.py").read_text()
    m = re.search(rf"CONTENT_TRUTH_{date} = (\[.*?\n\])", src, re.S)
    return ast.literal_eval(m.group(1)) if m else []


def score(date: str, max_seg_s: float, feature: str, margin: float) -> None:
    data = np.load(WORK / date / "video_spike" / "jaw.npz")
    act = activity(data[feature])
    segs = json.loads((WORK / date / "transcript.json").read_text())["segments"]
    starts = [_seconds(s["start"]) for s in segs]
    rows = []
    for k, s in enumerate(segs):
        t0 = starts[k]
        t1 = starts[k + 1] if k + 1 < len(segs) else t0 + 5
        t1 = min(max(t1, t0 + 1), t0 + max_seg_s)
        a = act[int(t0 * FPS):int(t1 * FPS)]
        if not len(a):
            continue
        st, te = a.mean(axis=0)
        rows.append((t0, s["speaker"], "student" if st > te else "teacher", np.log((st + 0.05) / (te + 0.05)), len(s["text"])))
    print(f"{len(rows)} segments, feature={feature}; agreement of the video guess with Pass A's label")
    print(f"('confident' = one face moving at least {np.exp(margin):.1f}x the other):")
    for lo, hi in [(0, 21), (21, 44), (44, 70), (0, 70)]:
        sel = [r for r in rows if lo * 60 <= r[0] < hi * 60]
        ok = sum(r[1] == r[2] for r in sel)
        conf = [r for r in sel if abs(r[3]) > margin]
        okc = sum(r[1] == r[2] for r in conf)
        print(f"  {lo:2d}-{hi:2d} min: {ok}/{len(sel)} = {ok / max(len(sel), 1):.0%}"
              f"   confident only: {okc}/{len(conf)} = {okc / max(len(conf), 1):.0%}")
    # A short line is usually a backchannel spoken over the other person, whose mouth is the one moving.
    print("by length of the line (characters), before / after 21:00:")
    for lo, hi in [(0, 8), (8, 50), (50, 10**6)]:
        cells = []
        for a, b in [(0, 21 * 60), (21 * 60, 10**6)]:
            sel = [r for r in rows if a <= r[0] < b and lo <= r[4] < hi]
            cells.append(f"{sum(r[1] == r[2] for r in sel)}/{len(sel)}")
        print(f"  {lo:2d}-{'' if hi > 999 else hi:<3} {cells[0]:>7}  {cells[1]:>8}")
    truth = _content_truth(date)
    if truth:
        # same lookup as compare_transcribe35.passa_at: the segment in force at time t
        picked = [(max((r for r in rows if r[0] <= t), key=lambda r: r[0], default=rows[0]), lab) for t, lab in truth]
        pa_ok = sum(r[1] == lab for r, lab in picked)
        vid_ok = sum(r[2] == lab for r, lab in picked)
        print(f"against {len(truth)} content-derived attributions (Claude's reading, not ears):"
              f" Pass A {pa_ok}/{len(truth)}, video {vid_ok}/{len(truth)}")
    out = WORK / date / "video_spike" / "segments.tsv"
    out.write_text("\n".join(f"{t}\t{pa}\t{vid}\t{d:.2f}" for t, pa, vid, d, _ in rows) + "\n")
    print(f"per-segment table: {out}")


def export(date: str, margin: float) -> None:
    """Write the video's guess as transcripts `distill label --compare` / `distill score --against` can read.

    transcript_video.json relabels every line; transcript_video_confident.json keeps
    Pass A's label unless the video disagrees by at least `margin`, so the kit's
    contested stratum is exactly the confident disagreements.
    """
    out = WORK / date / "video_spike"
    segs = json.loads((WORK / date / "transcript.json").read_text())["segments"]
    guess = {}
    for line in (out / "segments.tsv").read_text().splitlines():
        t, _pa, vid, d = line.split("\t")
        guess[int(t)] = (vid, float(d))
    full, conf, n = [], [], 0
    for s in segs:
        vid, d = guess.get(_seconds(s["start"]), (s["speaker"], 0.0))
        full.append({**s, "speaker": vid if d != 0.0 else s["speaker"]})  # a tie is no evidence
        flip = vid != s["speaker"] and abs(d) >= margin
        n += flip
        conf.append({**s, "speaker": vid if flip else s["speaker"]})
    (out / "transcript_video.json").write_text(json.dumps({"segments": full}, ensure_ascii=False, indent=1))
    (out / "transcript_video_confident.json").write_text(json.dumps({"segments": conf}, ensure_ascii=False, indent=1))
    print(f"{date}: {n} confident disagreements with Pass A (margin {margin}) -> {out}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    e = sub.add_parser("extract")
    e.add_argument("video", type=Path)
    e.add_argument("--date", required=True)
    e.add_argument("--start", type=float, default=0.0, help="seconds in; for a quick look at one stretch")
    e.add_argument("--duration", type=float, default=None)
    s = sub.add_parser("score")
    s.add_argument("--date", required=True)
    s.add_argument("--max-seg-s", type=float, default=15.0)
    s.add_argument("--feature", choices=("jaw", "gap"), default="gap")
    s.add_argument("--margin", type=float, default=0.7)
    x = sub.add_parser("export")
    x.add_argument("--date", required=True)
    x.add_argument("--margin", type=float, default=0.7)
    a = p.parse_args()
    if a.cmd == "extract":
        extract(a.video, a.date, a.start, a.duration)
    elif a.cmd == "export":
        export(a.date, a.margin)
    else:
        score(a.date, a.max_seg_s, a.feature, a.margin)


if __name__ == "__main__":
    sys.exit(main())
