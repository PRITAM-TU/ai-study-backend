"""
Pydantic schemas for AI feature endpoints.
"""

from pydantic import BaseModel, Field
from typing import Optional


# ── Summary ──
class SummaryRequest(BaseModel):
    doc_id: str
    max_length: Optional[int] = None


class SummaryDefinition(BaseModel):
    term: str
    definition: str


class SummaryPoint(BaseModel):
    topic: str
    points: list[str]


class SummaryResponse(BaseModel):
    overview: str
    key_topics: list[str]
    main_points: list[SummaryPoint]
    definitions: list[SummaryDefinition]
    conclusion: str


# ── Quiz ──
class QuizRequest(BaseModel):
    doc_id: str
    num_questions: int = Field(default=10, ge=3, le=30)
    difficulty: str = Field(default="medium", pattern="^(easy|medium|hard)$")


class QuizQuestion(BaseModel):
    id: int
    type: str
    question: str
    options: list[str] = []
    correct_answer: str
    explanation: str


class QuizResponse(BaseModel):
    quiz_title: str
    difficulty: str
    questions: list[QuizQuestion]


class QuizSubmitRequest(BaseModel):
    quiz: QuizResponse
    answers: dict[str, str]  # {question_id: user_answer}


class QuizResult(BaseModel):
    total_questions: int
    correct: int
    wrong: int
    score_percentage: float
    results: list[dict]


# ── Flashcards ──
class FlashcardRequest(BaseModel):
    doc_id: str
    num_cards: int = Field(default=15, ge=5, le=50)


class Flashcard(BaseModel):
    id: int
    front: str
    back: str
    category: str
    difficulty: str


class FlashcardResponse(BaseModel):
    topic: str
    flashcards: list[Flashcard]


# ── Exam Mode ──
class ExamRequest(BaseModel):
    doc_id: str


class ImportantTopic(BaseModel):
    rank: int
    topic: str
    importance: str
    reason: str


class MCQQuestion(BaseModel):
    id: int
    question: str
    options: list[str]
    correct_answer: str
    explanation: str


class SubjectiveQuestion(BaseModel):
    id: int
    question: str
    key_points: list[str]
    marks: int


class KeyFormula(BaseModel):
    name: str
    formula: str
    usage: str


class ExamPrediction(BaseModel):
    important_topics: list[ImportantTopic]
    predicted_questions: dict
    repeated_topics: list[str]
    key_formulas: list[KeyFormula]
