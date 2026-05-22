"""Define the shared LangGraph state used across all nodes in the workflow.
The state also includes a reducer-backed history field so iteration records accumulate cleanly.
"""

from __future__ import annotations

from typing import Annotated, TypedDict
import operator


class ThumbnailState(TypedDict, total=False):
    """Store all mutable data needed by the graph across iterations.
    This includes topic metadata, current artifacts, best result tracking, and append-only history.
    """

    topic: str
    slug: str
    target_rating: int
    max_iterations: int
    iteration: int
    output_dir: str
    search_summary: str
    current_prompt: str
    current_image_path: str
    rating: int
    critique: str
    best_rating: int
    best_image_path: str
    best_prompt: str
    final_image_path: str
    report_path: str
    history: Annotated[list[dict], operator.add]
