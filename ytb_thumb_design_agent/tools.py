"""Provide external tool helpers used by graph nodes.
Right now this module wraps Tavily search and converts raw search results into a compact summary.
"""

from __future__ import annotations

import os
from tavily import TavilyClient


def tavily_search_summary(topic: str, max_results: int = 5) -> str:
    """Search Tavily for topic hooks and visual references, then summarize the results.
    The returned text is fed into the prompt writer to ground thumbnail ideas in web research.
    """

    api_key = os.environ.get("TAVILY_API_KEY")
    if not api_key:
        raise RuntimeError("TAVILY_API_KEY is not set")

    client = TavilyClient(api_key=api_key)
    query = f"Best YouTube thumbnail hooks and visual references for topic: {topic}"
    result = client.search(
        query=query, search_depth="advanced", max_results=max_results
    )

    lines: list[str] = []
    for idx, item in enumerate(result.get("results", []), start=1):
        title = item.get("title", "Untitled")
        content = (item.get("content", "") or "").replace("\n", " ").strip()
        url = item.get("url", "")
        lines.append(f"{idx}. {title} | {url}\n{content[:400]}")

    return "\n\n".join(lines)
