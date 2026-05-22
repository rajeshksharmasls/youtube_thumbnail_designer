"""Generate a PNG diagram of the compiled LangGraph workflow.
This utility is used for assignment submission to visualize the node and loop architecture.
"""

from __future__ import annotations

from pathlib import Path

import ytb_thumb_design_agent
from ytb_thumb_design_agent.graph import build_graph


def main() -> None:
    """Compile the graph and export its Mermaid rendering to graph.png.
    This makes it easy to commit a visual proof of the required LangGraph architecture.
    """

    graph = build_graph()

    # Create output directory if missing
    output_dir = Path("./outputs")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Render graph
    png_bytes = graph.get_graph().draw_mermaid_png()

    # Save PNG
    out = output_dir / "graph.png"
    out.write_bytes(png_bytes)

    print(f"Wrote graph visualization to: {out.resolve()}")


if __name__ == "__main__":
    main()

