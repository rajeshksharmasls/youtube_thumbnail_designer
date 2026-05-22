"""Implement the LangGraph node functions for the thumbnail reflection workflow.
This module contains the five required nodes, the structured critic schema, and the loop routing function.
"""

from __future__ import annotations

import base64
import shutil
from pathlib import Path
from typing import Literal

from openai import BadRequestError, OpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

from ytb_thumb_design_agent.prompts import CRITIC_SYSTEM, PROMPT_WRITER_SYSTEM
from ytb_thumb_design_agent.state import ThumbnailState
from ytb_thumb_design_agent.tools import tavily_search_summary


class CritiqueResult(BaseModel):
    """Represent the vision critic's structured output.
    The model guarantees an integer rating plus actionable written feedback for each image.
    """

    rating: int = Field(ge=1, le=10)
    critique: str = Field(min_length=20)
    strengths: list[str] = Field(default_factory=list)
    improvements: list[str] = Field(default_factory=list)


def web_search(state: ThumbnailState) -> ThumbnailState:
    """Run the one-time Tavily search step for the provided video topic.
    It stores the summarized findings in state and records the event in history.
    """

    summary = tavily_search_summary(state["topic"])
    return {
        "search_summary": summary,
        "history": [
            {"node": "web_search", "topic": state["topic"], "search_summary": summary}
        ],
    }


def prompt_writer(state: ThumbnailState) -> ThumbnailState:
    """Generate or revise the thumbnail image prompt using search context and prior critique.
    On loop iterations, it explicitly tries to fix the weaknesses identified by the critic.
    """

    llm = ChatOpenAI(model="gpt-4.1-mini", temperature=0.7)

    critic_history = [h for h in state.get("history", []) if h.get("node") == "critic"]
    latest_critique = state.get("critique", "")
    previous_feedback = (
        "\n".join(
            f"Iteration {h['iteration']}: score={h['rating']} critique={h['critique']}"
            for h in critic_history
        )
        or "None"
    )

    user_prompt = f"""
Video topic: {state['topic']}
Target rating: {state.get('target_rating', 8)}
Current iteration to create: {state.get('iteration', 0) + 1}

Search summary:
{state.get('search_summary', '')}

Latest critique to address:
{latest_critique or 'None'}

All previous critic feedback:
{previous_feedback}

Write one improved thumbnail image-generation prompt.
The prompt must include:
- the focal subject
- exact scene composition
- text overlay content and placement
- lighting
- mood
- background
- color contrast strategy
- mobile readability considerations
""".strip()

    response = llm.invoke(
        [
            SystemMessage(content=PROMPT_WRITER_SYSTEM),
            HumanMessage(content=user_prompt),
        ]
    )

    prompt_text = (
        response.content if isinstance(response.content, str) else str(response.content)
    )

    return {
        "current_prompt": prompt_text,
        "history": [
            {
                "node": "prompt_writer",
                "iteration": state.get("iteration", 0) + 1,
                "prompt": prompt_text,
            }
        ],
    }


def generator(state: ThumbnailState) -> ThumbnailState:
    """Generate a thumbnail image from the current prompt and save it as iter_N.png.
    This node also increments the iteration counter and updates the current image path.
    """

    client = OpenAI()
    output_dir = Path(state["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    next_iteration = state.get("iteration", 0) + 1
    image_path = output_dir / f"iter_{next_iteration}.png"

    # Use a GPT image model that is supported by the installed OpenAI SDK and
    # is more likely to be available on most accounts than DALL·E models.
    # The size is set explicitly to a standard square resolution.
    image_models = ["gpt-image-1", "gpt-image-1.5", "gpt-image-1-mini"]
    result = None
    for model in image_models:
        try:
            result = client.images.generate(
                model=model,
                prompt=state["current_prompt"],
                size="1024x1024",
            )
            break
        except BadRequestError as error:
            message = str(error).lower()
            if "does not exist" in message or "invalid model" in message:
                continue
            raise

    if result is None:
        raise RuntimeError(
            "Unable to generate an image. None of the configured image models were accepted by OpenAI."
        )

    image_b64 = result.data[0].b64_json
    image_path.write_bytes(base64.b64decode(image_b64))

    return {
        "iteration": next_iteration,
        "current_image_path": str(image_path),
        "history": [
            {
                "node": "generator",
                "iteration": next_iteration,
                "image_path": str(image_path),
            }
        ],
    }


def critic(state: ThumbnailState) -> ThumbnailState:
    """Send the generated image to a vision LLM for strict structured evaluation.
    It returns an integer rating, a critique, and updates the best-so-far result when appropriate.
    """

    image_path = Path(state["current_image_path"])
    img_b64 = base64.b64encode(image_path.read_bytes()).decode()

    llm = ChatOpenAI(model="gpt-4o", temperature=0).with_structured_output(
        CritiqueResult
    )
    result = llm.invoke(
        [
            SystemMessage(content=CRITIC_SYSTEM),
            HumanMessage(
                content=[
                    {
                        "type": "text",
                        "text": (
                            f"Evaluate this YouTube thumbnail for the topic '{state['topic']}'. "
                            f"This is iteration {state['iteration']}. "
                            "Return a strict rating from 1 to 10 and an actionable critique. "
                            "Assume most thumbnails should score 5 to 7 unless truly excellent."
                        ),
                    },
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{img_b64}"},
                    },
                ]
            ),
        ]
    )

    updates: ThumbnailState = {
        "rating": result.rating,
        "critique": result.critique,
        "history": [
            {
                "node": "critic",
                "iteration": state["iteration"],
                "image_path": state["current_image_path"],
                "prompt": state["current_prompt"],
                "rating": result.rating,
                "critique": result.critique,
                "strengths": result.strengths,
                "improvements": result.improvements,
            }
        ],
    }

    if result.rating >= state.get("best_rating", 0):
        updates["best_rating"] = result.rating
        updates["best_image_path"] = state["current_image_path"]
        updates["best_prompt"] = state["current_prompt"]

    return updates


def should_continue(state: ThumbnailState) -> Literal["prompt_writer", "saver"]:
    """Decide whether the graph should loop again or move to the saver node.
    The workflow stops once the target rating is met or the iteration cap is reached.
    """

    if state.get("rating", 0) >= state.get("target_rating", 8):
        return "saver"
    if state.get("iteration", 0) >= state.get("max_iterations", 3):
        return "saver"
    return "prompt_writer"


def saver(state: ThumbnailState) -> ThumbnailState:
    """Copy the best generated image to final.png and write the full markdown report.
    The report captures the search summary plus every iteration's prompt, score, and critique.
    """

    output_dir = Path(state["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    best_image_path = Path(state.get("best_image_path") or state["current_image_path"])
    final_image_path = output_dir / "final.png"
    shutil.copyfile(best_image_path, final_image_path)

    report_path = output_dir / "report.md"
    critic_entries = [h for h in state.get("history", []) if h.get("node") == "critic"]
    prompt_entries = {
        h["iteration"]: h
        for h in state.get("history", [])
        if h.get("node") == "prompt_writer"
    }

    lines = [
        "# Thumbnail Run Report",
        "",
        f"- Topic: {state['topic']}",
        f"- Target rating: {state.get('target_rating', 8)}",
        f"- Max iterations: {state.get('max_iterations', 3)}",
        f"- Completed iterations: {state.get('iteration', 0)}",
        f"- Best rating: {state.get('best_rating', state.get('rating', 0))}",
        f"- Final image: `final.png`",
        "",
        "## Search summary",
        "",
        state.get("search_summary", ""),
        "",
        "## Iteration history",
        "",
    ]

    for entry in critic_entries:
        iteration = entry["iteration"]
        prompt_text = prompt_entries.get(iteration, {}).get(
            "prompt", entry.get("prompt", "")
        )
        lines.extend(
            [
                f"### Iteration {iteration}",
                "",
                f"- Image: `iter_{iteration}.png`",
                f"- Rating: {entry['rating']}/10",
                f"- Critique: {entry['critique']}",
                f"- Strengths: {', '.join(entry.get('strengths', [])) or 'None'}",
                f"- Improvements: {', '.join(entry.get('improvements', [])) or 'None'}",
                "",
                "#### Prompt used",
                "",
                "```text",
                prompt_text,
                "```",
                "",
            ]
        )

    report_path.write_text("\n".join(lines), encoding="utf-8")

    return {
        "final_image_path": str(final_image_path),
        "report_path": str(report_path),
        "history": [
            {
                "node": "saver",
                "final_image_path": str(final_image_path),
                "report_path": str(report_path),
            }
        ],
    }
