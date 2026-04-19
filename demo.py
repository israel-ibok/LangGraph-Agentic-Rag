"""
LangGraph Agentic RAG Demo
Israel Ibok — Xpectra Systems
Built for Plasma Computing Group interview — April 14, 2026

Run from project root: python demo.py

This demo implements the Plan -> Act -> Observe -> Respond pattern using:
  - LangGraph state machine with conditional routing
  - ChromaDB vector store with OpenAI embeddings
  - Hallucination grading before final response
"""

import sys
import os
import json
import time

# Add src/ to path so graph.py can find agent.py and rag_pipeline.py
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))


def print_banner():
    print()
    print("=" * 62)
    print("   AGENTIC RAG DEMO")
    print("   Plan -> Act -> Observe -> Respond")
    print("   LangGraph + ChromaDB + OpenAI gpt-4o-mini")
    print("=" * 62)
    print()


def print_graph_structure():
    print("GRAPH STRUCTURE:")
    print("-" * 62)
    print("  START")
    print("    |")
    print("  [route_question]  ........... PLAN")
    print("    |              \\")
    print("  [retrieve_docs]  [generate_answer] (direct)")
    print("    |")
    print("  [grade_documents] ........... OBSERVE")
    print("    |              \\")
    print("  [generate_answer] [rewrite_query] -> retrieve (loop)")
    print("    |")
    print("  [grade_hallucination] ....... OBSERVE")
    print("    |              \\")
    print("   END         [generate_answer] (retry, max 2x)")
    print("-" * 62)
    print()


def build_store_if_needed():
    """Build ChromaDB vector store on first run."""
    chroma_path = os.path.join(os.path.dirname(__file__), "data", "chroma_db")
    if not os.path.exists(chroma_path) or not os.listdir(chroma_path):
        print("First run detected — building vector store from knowledge base...")
        print("(This takes ~15 seconds, only happens once)\n")
        from rag_pipeline import build_vector_store
        build_vector_store()
        print()
    else:
        print("Vector store found. Loading existing knowledge base.\n")


def run_demo():
    print_banner()
    print_graph_structure()

    # Build vector store if this is first run
    build_store_if_needed()

    # Compile graph
    print("Compiling agent graph...")
    from graph import build_graph
    app = build_graph()
    print("Agent ready.\n")

    # Demo questions optimized for the interview
    demo_questions = [
        "What is Plasma Computing Group's GenAI platform and what can it do?",
        "How does LangGraph implement stateful agentic workflows?",
        "What are the key components of a RAG pipeline and how does it prevent hallucinations?",
        "How would you connect LangGraph to a BPMN workflow engine?",
    ]

    print("SUGGESTED DEMO QUESTIONS:")
    for i, q in enumerate(demo_questions, 1):
        print(f"  {i}. {q}")
    print()

    while True:
        question = input("Enter your question (or press Enter for question 1, 'q' to quit): ").strip()
        if question.lower() in ("q", "quit", "exit"):
            print("Session ended.")
            break
        if not question:
            question = demo_questions[0]

        print(f"\nQuestion: {question}")
        print("=" * 62)

        # Run agent
        start_time = time.time()
        initial_state = {
            "question": question,
            "retrieved_docs": [],
            "generation": "",
            "hallucination_count": 0,
            "rewrite_count": 0,
            "route_decision": "",
            "grade_decision": "",
        }

        result = app.invoke(initial_state)
        elapsed = time.time() - start_time

        # Final output
        print()
        print("=" * 62)
        print("FINAL ANSWER:")
        print("-" * 62)
        print(result["generation"])
        print("-" * 62)
        print(f"Completed in {elapsed:.1f}s")
        print(f"Hallucination checks: {result.get('hallucination_count', 0)}")
        print(f"Final grade: {result.get('grade_decision', 'grounded')}")
        print("=" * 62)
        print()


if __name__ == "__main__":
    run_demo()
