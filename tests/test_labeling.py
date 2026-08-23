"""The diarization ground-truth kit (jld-dli): sampling, blinding, scoring, McNemar."""

from __future__ import annotations

from jp_lesson_distill.labeling import (
    Item,
    build_items,
    marker_conflict,
    match,
    mcnemar,
    parse_script,
    score,
    similarity,
)
from jp_lesson_distill.models import Segment


def seg(start: str, speaker: str, text: str) -> Segment:
    return Segment(start=start, speaker=speaker, text=text)


def lesson(n: int = 60) -> list[Segment]:
    return [seg(f"{i // 2:02d}:{(i % 2) * 30:02d}",
                "teacher" if i % 2 else "student",
                f"これは{i}番目の発話です、はい、そうですね。")
            for i in range(n)]


# --- cues ---

def test_marker_conflict_fires_only_when_the_label_contradicts_the_address_form():
    assert marker_conflict(seg("05:00", "teacher", "先生は韓国に行ったことがありますか？"))
    assert marker_conflict(seg("05:00", "student", "先生は韓国に行ったことがありますか？")) is None
    assert marker_conflict(seg("05:00", "student", "スティーブンさんはどうですか？"))
    assert marker_conflict(seg("05:00", "teacher", "スティーブンさんはどうですか？")) is None


def test_own_wife_is_the_student_and_someone_elses_is_the_teacher():
    assert marker_conflict(seg("12:12", "teacher", "私はもう予定はありませんけど、妻が予定があります。"))
    assert marker_conflict(seg("10:29", "student", "なるほど、奥さんは仕事して、"))


def test_lines_without_a_marker_are_not_contested_by_cue():
    assert marker_conflict(seg("01:00", "teacher", "そうですね、いいと思います。")) is None


def test_match_finds_the_same_utterance_in_another_run():
    a = seg("10:00", "student", "来週末にロンドンに行きます。")
    pool = [seg("09:58", "teacher", "来週末にロンドンに行きます"), seg("30:00", "student", "全然違う話です")]
    found = match(a, pool)
    assert found is not None and found.speaker == "teacher"


def test_match_refuses_a_distant_or_unrelated_line():
    a = seg("10:00", "student", "来週末にロンドンに行きます。")
    assert match(a, [seg("40:00", "teacher", "来週末にロンドンに行きます")]) is None
    assert match(a, [seg("10:02", "teacher", "全く関係のない文章ですね、これは。")]) is None


# --- sampling ---

def test_strata_are_sampled_to_the_requested_sizes():
    items = build_items(lesson(), n_random=5, n_contested=0, n_scripted=0, seed=1)
    assert len(items) == 5
    assert {i.stratum for i in items} == {"random"}


def test_contested_lines_are_drawn_before_random_ones():
    segs = lesson()
    segs[7] = seg(segs[7].start, "teacher", "先生は、あの、韓国に行ったことがありますか？")
    items = build_items(segs, n_random=3, n_contested=5, n_scripted=0, seed=1)
    contested = [i for i in items if i.stratum == "contested"]
    assert len(contested) == 1
    assert "先生" in contested[0].text
    assert contested[0].reason


def test_a_second_run_supplies_the_contested_stratum():
    segs = lesson(20)
    other = [seg(s.start, "teacher" if s.speaker == "student" else "student", s.text) for s in segs]
    items = build_items(segs, compare=other, n_random=0, n_contested=6, n_scripted=0, seed=2)
    assert len(items) == 6
    assert all(i.stratum == "contested" for i in items)
    assert all(i.compare_speaker and i.compare_speaker != i.speaker for i in items)


def test_no_line_is_sampled_twice():
    segs = lesson()
    segs[7] = seg(segs[7].start, "teacher", "先生は韓国に行ったことがありますか？")
    items = build_items(segs, n_random=10, n_contested=5, n_scripted=0, seed=3)
    assert len({i.t_start for i in items}) == len(items)


def test_the_same_seed_rebuilds_the_same_kit():
    a = build_items(lesson(), n_random=8, n_contested=0, n_scripted=0, seed=7)
    b = build_items(lesson(), n_random=8, n_contested=0, n_scripted=0, seed=7)
    assert [i.id for i in a] == [i.id for i in b]
    assert [i.order for i in a] == [i.order for i in b]


def test_presentation_order_is_shuffled_away_from_time_order():
    """A listener working through a lesson in order infers alternation instead of hearing it."""
    items = build_items(lesson(), n_random=20, n_contested=0, n_scripted=0, seed=5)
    assert [i.order for i in items] != sorted(i.order for i in items)
    assert sorted(i.order for i in items) == list(range(len(items)))


def test_truth_starts_empty_so_the_listener_has_nothing_to_agree_with():
    items = build_items(lesson(), n_random=5, n_contested=0, n_scripted=0, seed=1)
    assert all(i.truth is None for i in items)


# --- textbook script ---

SCRIPT_MD = """
# いろどり初級2 第13課

出典についての注意書き。ここは表ではない。

## 会話13-07 スポーツジムの受付

| 話者 | 発話 |
| --- | --- |
| 客 | すみません、初めて利用するんですが。 |
| 係員 | はい。ジムのご利用ですね。 |
"""


def test_script_parsing_takes_utterances_and_skips_headers_and_prose():
    lines = parse_script(SCRIPT_MD)
    assert lines == ["すみません、初めて利用するんですが。", "はい。ジムのご利用ですね。"]


def test_scripted_stratum_picks_lines_that_match_the_dialogue():
    segs = lesson(20)
    segs[4] = seg(segs[4].start, "student", "すみません、初めて利用するんですが。")
    items = build_items(segs, script=parse_script(SCRIPT_MD),
                        n_random=0, n_contested=0, n_scripted=5, seed=1)
    assert [i.stratum for i in items] == ["scripted"]
    assert "初めて利用する" in items[0].text


def test_a_lesson_that_never_reached_the_dialogue_just_has_no_scripted_stratum():
    items = build_items(lesson(20), script=[], n_random=3, n_contested=0, n_scripted=5, seed=1)
    assert {i.stratum for i in items} == {"random"}


# --- scoring ---

def _item(id_, stratum, speaker, truth) -> Item:
    return Item(id=id_, stratum=stratum, t_start=0.0, text="x", speaker=speaker,
                clip="c.m4a", order=0, truth=truth)


def test_accuracy_is_reported_per_stratum_and_overall():
    items = [
        _item("a", "random", "teacher", "teacher"),
        _item("b", "random", "teacher", "student"),
        _item("c", "contested", "student", "student"),
    ]
    scores, confusion = score(items)
    by_name = {s.stratum: s for s in scores}
    assert by_name["random"].accuracy == 0.5
    assert by_name["contested"].accuracy == 1.0
    assert by_name["ALL"].n == 3
    assert confusion == {"student -> teacher": 1}


def test_unsure_and_unlabelled_clips_are_excluded_not_counted_wrong():
    items = [_item("a", "random", "teacher", "teacher"),
             _item("b", "random", "teacher", "unsure"),
             _item("c", "random", "teacher", None)]
    scores, _ = score(items)
    assert {s.stratum: s.n for s in scores}["ALL"] == 1


def test_mcnemar_ignores_agreements_and_counts_only_the_flips():
    items = [_item(str(i), "contested", "teacher", "student") for i in range(10)]
    items += [_item(f"ok{i}", "random", "teacher", "teacher") for i in range(50)]
    after = {str(i): "student" for i in range(8)}      # repair fixes 8 of the 10 wrong ones
    after["ok0"] = "student"                            # and breaks 1 that was right
    fixed, broken, p = mcnemar(items, {}, after)
    assert (fixed, broken) == (8, 1)
    assert p < 0.05


def test_mcnemar_calls_an_even_split_insignificant():
    items = [_item(str(i), "contested", "teacher", "student") for i in range(4)]
    items += [_item(f"ok{i}", "random", "teacher", "teacher") for i in range(4)]
    after = {"0": "student", "1": "student", "ok0": "student", "ok1": "student"}
    fixed, broken, p = mcnemar(items, {}, after)
    assert (fixed, broken) == (2, 2)
    assert p == 1.0


def test_mcnemar_on_a_repair_that_changed_nothing():
    items = [_item("a", "random", "teacher", "teacher")]
    assert mcnemar(items, {}, {}) == (0, 0, 1.0)


def test_similarity_survives_punctuation_and_orthography_differences():
    assert similarity("あ、すみません。図書館内は飲食禁止でお願いします。",
                      "あ、すみません、図書館内は飲食禁止でお願いします") > 0.9
    assert similarity("全然違う文です", "図書館内は飲食禁止です") < 0.5
