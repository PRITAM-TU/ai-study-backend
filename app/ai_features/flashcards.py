"""
Flashcard generation for spaced repetition study.
"""

from app.ai_features.llm_client import ask_llm_json
from app.ai_features.prompts import FLASHCARDS_SYSTEM, FLASHCARDS_USER


async def generate_flashcards(text: str, num_cards: int = 15) -> dict:
    """
    Generate flashcards from document text.

    Args:
        text: The document text
        num_cards: Number of flashcards to create

    Returns:
        Dict with topic and flashcards list (each with front, back, category, difficulty)
    """
    max_chars = 10000
    if len(text) > max_chars:
        text = text[:max_chars] + "\n\n[Content truncated...]"

    prompt = FLASHCARDS_USER.format(
        text=text,
        num_cards=num_cards,
    )
    result = await ask_llm_json(
        system_prompt=FLASHCARDS_SYSTEM,
        user_prompt=prompt,
    )
    return result
