"""
Summary generation from document content.
"""

from app.ai_features.llm_client import ask_llm_json
from app.ai_features.prompts import SUMMARY_SYSTEM, SUMMARY_USER


async def generate_summary(text: str) -> dict:
    """
    Generate a structured summary of the document text.

    Args:
        text: The full document text (or concatenated chunks)

    Returns:
        Dict with overview, key_topics, main_points, definitions, conclusion
    """
    # Truncate if text is too long (LLM context limit)
    max_chars = 12000
    if len(text) > max_chars:
        text = text[:max_chars] + "\n\n[Content truncated for summary...]"

    prompt = SUMMARY_USER.format(text=text)
    result = await ask_llm_json(
        system_prompt=SUMMARY_SYSTEM,
        user_prompt=prompt,
    )
    return result
