#!/bin/zsh
# Synthesizes a ~1 minute fake lesson with macOS `say` for end-to-end smoke tests.
# Planted moments:
#   1. student particle error 「学校で行きました」→ teacher corrects to 「に」 (type: correction)
#   2. student error 「面白いでした」 which the teacher does NOT correct (type: uncorrected-error)
set -euo pipefail
cd "$(dirname "$0")/.."

OUT_DIR="work/test"
mkdir -p "$OUT_DIR"
FFMPEG="$(command -v ffmpeg || uv run python -c 'import imageio_ffmpeg; print(imageio_ffmpeg.get_ffmpeg_exe())')"

TEACHER_VOICE="Kyoko"
# A second Japanese-capable voice for the student so diarization has something to work with.
STUDENT_VOICE="Kyoko"
for v in "Otoya" "Eddy (日本語（日本）)" "Flo (日本語（日本）)" "Reed (日本語（日本）)"; do
  if say -v "$v" -o "$OUT_DIR/_voicetest.aiff" "テスト" 2>/dev/null; then
    STUDENT_VOICE="$v"
    break
  fi
done
rm -f "$OUT_DIR/_voicetest.aiff"
echo "teacher voice: $TEACHER_VOICE / student voice: $STUDENT_VOICE"

# speaker|rate|text  (student reads slower, like a learner)
SEGS=(
  "T|170|こんにちは、スティーブンさん。週末は何をしましたか。"
  "S|130|えーと、日曜日に友達と学校で行きました。"
  "T|170|ああ、学校に行きました、ですね。「で」じゃなくて「に」を使いましょう。"
  "S|130|あ、そうですね。学校に行きました。それから、映画を見ました。"
  "T|170|いいですね。どんな映画を見ましたか。"
  "S|130|アクション映画です。とても面白いでした。"
  "T|170|そうですか。じゃあ、今日はレッスン12の会話を練習しましょう。"
)

CONCAT="$OUT_DIR/_concat.txt"
: > "$CONCAT"
i=0
for line in "${SEGS[@]}"; do
  spk="${line%%|*}"; rest="${line#*|}"; rate="${rest%%|*}"; text="${rest#*|}"
  voice="$TEACHER_VOICE"; [[ "$spk" == "S" ]] && voice="$STUDENT_VOICE"
  i=$((i + 1))
  say -v "$voice" -r "$rate" -o "$OUT_DIR/_seg$i.aiff" "$text"
  "$FFMPEG" -hide_banner -loglevel error -y -i "$OUT_DIR/_seg$i.aiff" -ar 24000 -ac 1 "$OUT_DIR/_seg$i.wav"
  echo "file '_seg$i.wav'" >> "$CONCAT"
  # ~0.8 s pause between turns
  if [[ ! -f "$OUT_DIR/_gap.wav" ]]; then
    "$FFMPEG" -hide_banner -loglevel error -y -f lavfi -i anullsrc=r=24000:cl=mono -t 0.8 "$OUT_DIR/_gap.wav"
  fi
  echo "file '_gap.wav'" >> "$CONCAT"
done

"$FFMPEG" -hide_banner -loglevel error -y -f concat -safe 0 -i "$CONCAT" -c:a aac -b:a 64k "$OUT_DIR/test_lesson.m4a"
rm -f "$OUT_DIR"/_seg*.aiff "$OUT_DIR"/_seg*.wav "$OUT_DIR/_gap.wav" "$CONCAT"
echo "wrote $OUT_DIR/test_lesson.m4a"
