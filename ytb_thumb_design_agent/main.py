"""Provide the command-line interface for running the thumbnail designer.
This module parses arguments, builds the initial state, and runs the graph in normal or stream mode.
"""

from __future__ import annotations

import argparse
import re
from datetime import datetime
from pathlib import Path

import ytb_thumb_design_agent
from ytb_thumb_design_agent.graph import build_graph


def slugify(text: str) -> str:
    """Convert a topic string into a safe, readable folder slug.
    This keeps output directory names deterministic and filesystem friendly.
    """

    slug = re.sub(r"[^a-zA-Z0-9]+", "-", text.lower()).strip("-")
    return slug[:80] or "thumbnail-topic"


def parse_args() -> argparse.Namespace:
    """Parse command-line inputs such as the topic, loop settings, and stream flag.
    The resulting namespace controls how the graph run is configured.
    """

    parser = argparse.ArgumentParser(description="LangGraph YouTube Thumbnail Designer")
    parser.add_argument("topic", type=str, help="Video topic for the thumbnail")
    parser.add_argument("--target-rating", type=int, default=8)
    parser.add_argument("--max-iterations", type=int, default=3)
    parser.add_argument(
        "--stream", action="store_true", help="Stream graph node updates live"
    )
    return parser.parse_args()


def build_initial_state(topic: str, target_rating: int, max_iterations: int) -> dict:
    """Create the first graph state and allocate the timestamped output directory.
    It initializes counters, thresholds, and the append-only history container.
    """

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    slug = slugify(topic)
    output_dir = Path("outputs") / f"{ts}_{slug}"
    output_dir.mkdir(parents=True, exist_ok=True)
    return {
        "topic": topic,
        "slug": slug,
        "target_rating": target_rating,
        "max_iterations": max_iterations,
        "iteration": 0,
        "best_rating": 0,
        "output_dir": str(output_dir),
        "history": [],
    }


def main() -> None:
    """Run the compiled graph from the command line and print output locations.
    It supports both normal execution and live streaming of node-level updates.
    """

    args = parse_args()
    graph = build_graph()
    initial_state = build_initial_state(
        args.topic, args.target_rating, args.max_iterations
    )

    if args.stream:
        for chunk in graph.stream(initial_state):
            print(chunk)
    else:
        result = graph.invoke(initial_state)
        print("Run complete")
        print(f"Output dir: {initial_state['output_dir']}")
        print(f"Best rating: {result.get('best_rating')}")
        print(f"Final image: {result.get('final_image_path')}")
        print(f"Report: {result.get('report_path')}")


if __name__ == "__main__":
    main()
