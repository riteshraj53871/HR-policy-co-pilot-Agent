import json
import re
from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage
from app.config import get_settings


def _client(model: str, temperature: float) -> ChatGroq:
    return ChatGroq(
        model=model,
        temperature=temperature,
        api_key=get_settings().groq_api_key,
        max_retries=2,
    )


def generate(prompt: str, model: str, system: str | None = None, temperature: float = 0.0) -> str:
    messages = []
    if system:
        messages.append(SystemMessage(content=system))
    messages.append(HumanMessage(content=prompt))
    return _client(model, temperature).invoke(messages).content.strip()


def generate_json(prompt: str, model: str, system: str | None = None) -> dict:
    """For router/grader calls. Returns {} if the model's output isn't valid JSON,
    so callers must treat an empty dict as 'unknown' and fall back safely."""
    raw = generate(prompt, model, system=system, temperature=0.0)
    match = re.search(r"\{.*\}", raw, re.DOTALL)
    if not match:
        return {}
    try:
        return json.loads(match.group(0))
    except json.JSONDecodeError:
        return {}