"""Board timeline (ADR-0008): grouping screen reads into slides, offline.

`read` needs a video and macOS Vision, so these start from what it writes: one
record per frame read, each a list of text lines with their place on the screen.
"""
import json
from pathlib import Path

from jp_lesson_distill.board import build, find_video

DATE = "20990101"


def line(text, y, x=0.05, h=0.04, red=False):
    return {"text": text, "conf": 1.0, "x": x, "y": y, "w": 0.4, "h": h, "red": red}


def board_of(tmp_path, records):
    out = tmp_path / DATE / "board"
    out.mkdir(parents=True)
    (out / "ocr.jsonl").write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in records) + "\n")
    return build(DATE, tmp_path)


FREE_TALK = [line("フリートーク", 0.05, h=0.08)]
SLIDE = [line("会話を聞きましょう。", 0.05), line("図書館を利用する前に、何をしますか？", 0.3),
         line("何が必要ですか？", 0.45), line("何を注意されましたか？", 0.7)]


def test_text_present_when_a_slide_appears_is_the_slide_and_later_text_is_timed(tmp_path):
    board = board_of(tmp_path, [
        {"t": 10, "cut": True, "lines": FREE_TALK},
        {"t": 60, "cut": False, "lines": FREE_TALK + [line("腎臓 じんぞう", 0.2)]},
        {"t": 90, "cut": False, "lines": FREE_TALK + [line("腎臓 じんぞう kidney", 0.2), line("できもの bump", 0.3)]},
    ])
    (scene,) = board["scenes"]
    assert [l["text"] for l in scene["slide"]] == ["フリートーク"]
    # the line that grew keeps the time it was started and ends up with its final text
    assert [(l["t"], l["text"]) for l in scene["typed"]] == [(60, "腎臓 じんぞう kidney"), (90, "できもの bump")]


def test_returning_to_a_slide_carries_on_with_it(tmp_path):
    typed = line("利用カードをつくる", 0.36, red=True)
    board = board_of(tmp_path, [
        {"t": 10, "cut": True, "lines": SLIDE},
        {"t": 40, "cut": False, "lines": SLIDE + [typed]},
        {"t": 100, "cut": True, "lines": FREE_TALK + [line("まだ はれです。", 0.2), line("検査 けんさ test", 0.3)]},
        {"t": 130, "cut": False, "lines": FREE_TALK + [line("まだ はれです。", 0.2), line("検査 けんさ test", 0.3)]},
        {"t": 200, "cut": True, "lines": SLIDE + [typed]},
        {"t": 230, "cut": False, "lines": SLIDE + [typed, line("住所を確認できるもの", 0.5)]},
    ])
    first, second = board["scenes"]
    assert first["visits"] == [[10, 40], [200, 230]]
    assert [l["text"] for l in first["typed"]] == ["利用カードをつくる", "住所を確認できるもの"]
    assert first["typed"][0]["red"]
    assert second["visits"] == [[100, 130]]


def test_furigana_and_stray_marks_are_not_notes(tmp_path):
    board = board_of(tmp_path, [
        {"t": 10, "cut": True, "lines": SLIDE},
        {"t": 40, "cut": False, "lines": SLIDE + [line("としょかん", 0.27, h=0.022), line("）））", 0.6), line("煙を吸う", 0.6, x=0.5)]},
        {"t": 70, "cut": False, "lines": SLIDE + [line("としょかん", 0.27, h=0.022), line("）））", 0.6), line("煙を吸う", 0.6, x=0.5)]},
    ])
    assert [l["text"] for l in board["scenes"][0]["typed"]] == ["煙を吸う"]


def test_a_shared_web_page_is_kept_as_a_title_only(tmp_path):
    page = [line("Google マップ", 0.02), line("www.google.co.jp/maps/@33.15", 0.06), line("五島市", 0.5), line("熊本県", 0.7)]
    board = board_of(tmp_path, [{"t": 10, "cut": True, "lines": page}, {"t": 50, "cut": False, "lines": page}])
    (scene,) = board["scenes"]
    assert scene["kind"] == "web" and scene["slide"] == [] and scene["typed"] == []


def test_a_slide_only_flicked_past_is_dropped(tmp_path):
    board = board_of(tmp_path, [
        {"t": 10, "cut": True, "lines": SLIDE},
        {"t": 12, "cut": True, "lines": FREE_TALK + [line("まだ はれです。", 0.2), line("検査 けんさ test", 0.3)]},
        {"t": 60, "cut": False, "lines": FREE_TALK + [line("まだ はれです。", 0.2), line("検査 けんさ test", 0.3)]},
    ])
    assert [s["title"] for s in board["scenes"]] == ["フリートーク"]


def test_the_video_is_found_beside_the_audio(tmp_path):
    audio = tmp_path / "Soso_20260928_class_audio.m4a"
    video = tmp_path / "Soso_20260928_class_video.mp4"
    audio.touch()
    assert find_video(audio) is None  # no video: the lesson simply has no board
    video.touch()
    assert find_video(audio) == video
    assert find_video(video) == video
    # 2026-09-24's audio was saved as …_class_video.m4a; that must not be mistaken for the video
    odd = tmp_path / "Soso_20260924_class_video.m4a"
    odd.touch()
    assert find_video(odd) is None
    (tmp_path / "Soso_20260924_class_video.mp4").touch()
    assert find_video(odd) == Path(tmp_path / "Soso_20260924_class_video.mp4")
