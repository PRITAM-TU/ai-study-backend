"""
OpenAI-compatible LLM client.
Works with Ollama (localhost:11434/v1), OpenAI, or any compatible API.
"""

import httpx
import json
import logging
from app.config import get_settings

settings = get_settings()
logger = logging.getLogger(__name__)


async def ask_llm(
    system_prompt: str,
    user_prompt: str,
    model: str | None = None,
    temperature: float | None = None,
    max_tokens: int | None = None,
    json_mode: bool = False,
) -> str:
    """
    Send a chat completion request to the LLM (Ollama via OpenAI-compatible API).

    Args:
        system_prompt: System message to set behavior
        user_prompt: User's message/question
        model: Override the default model
        temperature: Override the default temperature
        max_tokens: Override the default max tokens
        json_mode: If True, request JSON format response

    Returns:
        The LLM's response text
    """
    model = model or settings.LLM_MODEL
    temperature = temperature if temperature is not None else settings.LLM_TEMPERATURE
    max_tokens = max_tokens or settings.LLM_MAX_TOKENS

    url = f"{settings.LLM_BASE_URL}/chat/completions"

    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": temperature,
        "max_tokens": max_tokens,
    }

    if json_mode:
        payload["response_format"] = {"type": "json_object"}

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {settings.LLM_API_KEY}",
    }

    try:
        async with httpx.AsyncClient(timeout=120.0) as client:
            response = await client.post(url, json=payload, headers=headers)
            response.raise_for_status()

            data = response.json()
            content = data["choices"][0]["message"]["content"]
            return content.strip()

    except httpx.ConnectError:
        logger.error("Cannot connect to LLM. Is Ollama running? Start it with: ollama serve")
        raise ConnectionError(
            "Cannot connect to LLM service. Make sure Ollama is running: ollama serve"
        )
    except httpx.HTTPStatusError as e:
        logger.error(f"LLM API error: {e.response.status_code} - {e.response.text}")
        raise RuntimeError(f"LLM API error: {e.response.status_code}")
    except Exception as e:
        logger.error(f"LLM request failed: {e}")
        raise RuntimeError(f"LLM request failed: {str(e)}")


async def ask_llm_json(
    system_prompt: str,
    user_prompt: str,
    **kwargs,
) -> dict:
    """
    Send a request expecting a JSON response.
    Parses the response and returns a dict.
    Falls back to extracting JSON from markdown code blocks.
    """
    response = await ask_llm(
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        json_mode=True,
        temperature=0.3,  # Lower temp for structured output
        **kwargs,
    )

    try:
        return json.loads(response)
    except json.JSONDecodeError:
        # Try extracting JSON from markdown code block
        if "```json" in response:
            json_str = response.split("```json")[1].split("```")[0].strip()
            return json.loads(json_str)
        elif "```" in response:
            json_str = response.split("```")[1].split("```")[0].strip()
            return json.loads(json_str)
        else:
            logger.error(f"Failed to parse LLM JSON response: {response[:200]}")
            raise ValueError("LLM did not return valid JSON")
