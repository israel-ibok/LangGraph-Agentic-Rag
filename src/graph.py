"""
Graph — Assembles and compiles the LangGraph state machine.
Connects all nodes with edges and conditional routing.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(__file__))

from langgraph.graph import StateGraph, START, END
from agent import (
    AgentState,
    route_question,
    retrieve_documents,
    grade_documents,
    rewrite_query,
    generate_answer,
    grade_hallucination,
    route_after_question,
    route_after_grading,
    route_after_hallucination_check,
)


def build_graph():
    """Assemble and compile the LangGraph agentic RAG graph."""
    graph = StateGraph(AgentState)

    # ── Nodes ──────────────────────────────────────────────────────
    graph.add_node("route_question", route_question)
    graph.add_node("retrieve_documents", retrieve_documents)
    graph.add_node("grade_documents", grade_documents)
    graph.add_node("rewrite_query", rewrite_query)
    graph.add_node("generate_answer", generate_answer)
    graph.add_node("grade_hallucination", grade_hallucination)

    # ── Edges ───────────────────────────────────────────────────────
    graph.add_edge(START, "route_question")

    # Route → retrieve or direct answer
    graph.add_conditional_edges(
        "route_question",
        route_after_question,
        {
            "retrieve": "retrieve_documents",
            "direct": "generate_answer",
        },
    )

    # Retrieve → grade relevance
    graph.add_edge("retrieve_documents", "grade_documents")

    # Grade → generate if relevant, rewrite if not
    graph.add_conditional_edges(
        "grade_documents",
        route_after_grading,
        {
            "generate": "generate_answer",
            "rewrite": "rewrite_query",
        },
    )

    # Rewrite → retrieve again (loop)
    graph.add_edge("rewrite_query", "retrieve_documents")

    # Generate → hallucination check
    graph.add_edge("generate_answer", "grade_hallucination")

    # Hallucination check → end or regenerate
    graph.add_conditional_edges(
        "grade_hallucination",
        route_after_hallucination_check,
        {
            "end": END,
            "regenerate": "generate_answer",
        },
    )

    return graph.compile()


if __name__ == "__main__":
    app = build_graph()
    print("Graph compiled successfully.\n")
    try:
        print(app.get_graph().draw_ascii())
    except Exception:
        print("route_question -> retrieve_documents -> grade_documents -> generate_answer -> grade_hallucination -> END")
