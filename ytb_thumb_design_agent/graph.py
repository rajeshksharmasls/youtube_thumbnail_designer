"""Build and compile the LangGraph state machine for the thumbnail workflow.
This module wires the five nodes together and adds the conditional edge that powers reflection.
"""

from __future__ import annotations

from langgraph.graph import END, START, StateGraph

from ytb_thumb_design_agent.nodes import (
    critic,
    generator,
    prompt_writer,
    saver,
    should_continue,
    web_search,
)
from ytb_thumb_design_agent.state import ThumbnailState


def build_graph():
    """Create the graph, register all nodes, wire the loop, and return a compiled agent.
    The compiled workflow starts at web search and exits only after the saver node runs.
    """

    graph = StateGraph(ThumbnailState)

    graph.add_node("web_search", web_search)
    graph.add_node("prompt_writer", prompt_writer)
    graph.add_node("generator", generator)
    graph.add_node("critic", critic)
    graph.add_node("saver", saver)

    graph.add_edge(START, "web_search")
    graph.add_edge("web_search", "prompt_writer")
    graph.add_edge("prompt_writer", "generator")
    graph.add_edge("generator", "critic")
    graph.add_conditional_edges(
        "critic",
        should_continue,
        {
            "prompt_writer": "prompt_writer",
            "saver": "saver",
        },
    )
    graph.add_edge("saver", END)

    return graph.compile()
