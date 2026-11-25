from functools import lru_cache
from tavily import TavilyClient
from app.config import get_settings


@lru_cache
def _client() -> TavilyClient:
    return TavilyClient(api_key=get_settings().tavily_api_key)


def search(query: str, max_results: int = 5) -> list[dict]:
    """Returns [{"title", "url", "content", "score"}].
    RULE: `query` must only ever be the user's sanitized question,
    never internal document text."""
    res = _client().search(
        query=query,
        max_results=max_results,
        search_depth="basic",
        include_answer=False,
    )
    return [
        {
            "title": r.get("title", ""),
            "url": r.get("url", ""),
            "content": r.get("content", ""),
            "score": r.get("score", 0.0),
        }
        for r in res.get("results", [])
    ]