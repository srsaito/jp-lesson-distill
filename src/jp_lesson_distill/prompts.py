"""Prompts for the three Gemini stages. The recurring theme: the student's errors
are the product — every stage is told explicitly not to fix them."""

SPEAKERS = """Speakers:
- "teacher": Soso先生 — native Japanese speaker; speaks mostly Japanese, occasionally English.
- "student": Steven — a native English speaker learning Japanese (intermediate); speaks Japanese \
with learner mistakes, sometimes switching to English."""


PASS_A = f"""You are transcribing an audio recording of a one-on-one Japanese lesson.
{SPEAKERS}

Produce a complete diarized transcript as JSON segments. Rules:
1. VERBATIM. Preserve the student's errors exactly as spoken: wrong particles, wrong conjugations, \
wrong words, false starts, fillers (えーと、あの…), self-corrections. NEVER fix or smooth what was said. \
If the student says something ungrammatical, transcribe the ungrammatical version.
2. If the student mispronounces a word, render what you actually HEARD in kana, not the word he intended.
3. Write Japanese in normal kana/kanji orthography; English as spoken. Mixed-language segments are fine.
4. One segment per speaker turn; split turns longer than about 30 seconds.
5. "start" is the segment's start time measured from the beginning of the recording, formatted "MM:SS" \
(or "H:MM:SS" past one hour).
6. Set "uncertain": true when you could not make the words out clearly (mumbling, overlap, garble). \
Do not guess-correct — write your best hearing and flag it. These regions matter: they are often \
exactly where the student made a mistake.

Transcribe the ENTIRE recording, start to finish."""


DETECT = f"""Below is a diarized, timestamped transcript of a one-on-one Japanese lesson.
{SPEAKERS}

Find LEARNING MOMENTS — exchanges worth turning into flashcards for the student. Types:
- "correction": the teacher corrects or recasts something the student said (echoes it back changed, \
or explicitly corrects it).
- "uncorrected-error": the student's Japanese contains an error the teacher did NOT correct (he let it \
go and moved on). Judge the grammar yourself. These are the most valuable moments — they appear in no \
written notes.
- "hesitation": the student visibly struggles — long pauses, false starts, circumlocution around a form \
he seems to be avoiding.
- "new-item": vocabulary, expressions, or grammar the teacher introduces verbally in passing (likely \
absent from the written notes).

Additionally: any segment marked "uncertain": true that involves the student is a candidate \
(default type "uncorrected-error") — garbled transcription is often a mistake site.

For each candidate report: t_start and t_end ("MM:SS", spanning the whole exchange including the \
teacher's reaction), type, a one-sentence rationale, the transcript excerpt it is based on, and a \
priority from 0 to 1 (how flashcard-worthy).

Skip: pure English meta-discussion, small talk with no language-learning content, and passages the \
student read aloud correctly.

TRANSCRIPT (JSON):
{{transcript}}"""


PASS_B = """This audio is a short clip from a one-on-one Japanese lesson between "teacher" (Soso先生, \
native Japanese speaker) and "student" (Steven, native English speaker learning Japanese). \
The clip was flagged as a possible learning moment:
- suspected type: {type}
- rationale: {rationale}
- rough transcript of the region (may be WRONG — trust the audio over this text): {excerpt}

Listen carefully, then report:
1. keep — false if this is NOT actually a learning moment (transcription artifact, small talk, etc.).
2. type — your corrected classification.
3. student_verbatim — EXACTLY what the student said, errors preserved. Do NOT fix his Japanese; \
the error is the point. Use kana to render mispronunciations as heard.
4. teacher_correction — the corrected form the teacher gave (verbatim if he said one); if he did not \
correct it, the correct form he SHOULD have given. null only if no correction applies (e.g. new-item).
5. explanation — 1–3 sentences in English: what the error or point is and why it matters (particle \
choice, conjugation, register, vocabulary…).
6. confidence — 0 to 1, that this moment is real and correctly captured."""
