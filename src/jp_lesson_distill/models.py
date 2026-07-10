"""Pydantic schemas. Gemini structured output uses the Pass A/B and Detect models
directly as response_schema; MomentsFile is the repo↔vault contract (ADR-0003)."""

from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, Field

MomentType = Literal["correction", "uncorrected-error", "hesitation", "new-item"]


def parse_ts(ts: str) -> float:
    """'MM:SS', 'M:SS.5' or 'H:MM:SS' → seconds from recording start."""
    parts = ts.strip().split(":")
    if not 2 <= len(parts) <= 3:
        raise ValueError(f"bad timestamp: {ts!r}")
    seconds = 0.0
    for p in parts:
        seconds = seconds * 60 + float(p)
    return seconds


def fmt_ts(seconds: float) -> str:
    s = int(seconds)
    if s >= 3600:
        return f"{s // 3600}:{s % 3600 // 60:02d}:{s % 60:02d}"
    return f"{s // 60:02d}:{s % 60:02d}"


# --- Pass A ---

class Segment(BaseModel):
    start: str = Field(description="Segment start time from the beginning of the recording, 'MM:SS' or 'H:MM:SS'")
    speaker: Literal["teacher", "student"]
    text: str
    uncertain: bool = Field(default=False, description="True if the words could not be made out clearly")


class Transcript(BaseModel):
    segments: list[Segment]


# --- Detect ---

class Candidate(BaseModel):
    t_start: str = Field(description="'MM:SS' start of the exchange")
    t_end: str = Field(description="'MM:SS' end of the exchange, including the teacher's reaction")
    type: MomentType
    rationale: str
    excerpt: str = Field(description="The transcript excerpt this is based on")
    priority: float = Field(description="0-1, how flashcard-worthy")


class CandidateList(BaseModel):
    candidates: list[Candidate]


# --- Pass B ---

class Relisten(BaseModel):
    keep: bool = Field(description="False if this turned out not to be a learning moment")
    type: MomentType
    student_verbatim: str = Field(description="Exactly what the student said, errors preserved")
    teacher_correction: Optional[str] = Field(default=None, description="Corrected form (verbatim if the teacher said one)")
    explanation: str
    confidence: float = Field(description="0-1")


# --- Contract output (ADR-0003) ---

class Moment(BaseModel):
    id: str
    t_start: float
    t_end: float
    type: MomentType
    student_verbatim: str
    teacher_correction: Optional[str] = None
    explanation: str
    confidence: float


class MomentsFile(BaseModel):
    lesson_date: str
    source_recording: str
    model: str
    generated_at: str
    moments: list[Moment]
