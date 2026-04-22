"""
Exam mode: predicts important questions, identifies repeated topics,
and generates MCQ + subjective questions.
"""

from app.ai_features.llm_client import ask_llm_json
from app.ai_features.prompts import EXAM_PREDICT_SYSTEM, EXAM_PREDICT_USER, LAZY_MODE_SYSTEM, LAZY_MODE_USER


async def predict_exam(text: str) -> dict:
    """
    Analyze document text to predict exam-worthy content.

    Returns:
        Dict with important_topics, predicted_questions (mcq + subjective),
        repeated_topics, and key_formulas
    """
    max_chars = 12000
    if len(text) > max_chars:
        text = text[:max_chars] + "\n\n[Content truncated...]"

    prompt = EXAM_PREDICT_USER.format(text=text)
    result = await ask_llm_json(
        system_prompt=EXAM_PREDICT_SYSTEM,
        user_prompt=prompt,
    )
    return result


async def generate_lazy_mode_script(text: str) -> dict:
    """
    Convert study material into a conversational audio script for TTS.

    Returns:
        Dict with title, sections (each with title, script, key_takeaways),
        and full_script for TTS
    """
    max_chars = 10000
    if len(text) > max_chars:
        text = text[:max_chars] + "\n\n[Content truncated...]"

    prompt = LAZY_MODE_USER.format(text=text)
    result = await ask_llm_json(
        system_prompt=LAZY_MODE_SYSTEM,
        user_prompt=prompt,
    )
    return result
