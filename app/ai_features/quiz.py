"""
Quiz generation and auto-scoring.
"""

from app.ai_features.llm_client import ask_llm_json
from app.ai_features.prompts import QUIZ_SYSTEM, QUIZ_USER


async def generate_quiz(text: str, num_questions: int = 10, difficulty: str = "medium") -> dict:
    """
    Generate a quiz from document text.

    Args:
        text: The document text
        num_questions: Number of questions to generate
        difficulty: easy, medium, or hard

    Returns:
        Dict with quiz_title, difficulty, and questions list
    """
    max_chars = 10000
    if len(text) > max_chars:
        text = text[:max_chars] + "\n\n[Content truncated...]"

    prompt = QUIZ_USER.format(
        text=text,
        num_questions=num_questions,
        difficulty=difficulty,
    )
    result = await ask_llm_json(
        system_prompt=QUIZ_SYSTEM,
        user_prompt=prompt,
    )
    return result


def score_quiz(questions: list[dict], answers: dict[str, str]) -> dict:
    """
    Auto-score a quiz submission.

    Args:
        questions: List of quiz questions with correct_answer
        answers: Dict mapping question ID to user's answer

    Returns:
        Dict with score breakdown and per-question results
    """
    correct = 0
    results = []

    for q in questions:
        q_id = str(q["id"])
        user_answer = answers.get(q_id, "").strip().upper()
        correct_answer = q["correct_answer"].strip().upper()

        is_correct = user_answer == correct_answer
        if is_correct:
            correct += 1

        results.append({
            "question_id": q["id"],
            "question": q["question"],
            "user_answer": user_answer,
            "correct_answer": correct_answer,
            "is_correct": is_correct,
            "explanation": q.get("explanation", ""),
        })

    total = len(questions)
    return {
        "total_questions": total,
        "correct": correct,
        "wrong": total - correct,
        "score_percentage": round((correct / total) * 100, 1) if total > 0 else 0,
        "results": results,
    }
