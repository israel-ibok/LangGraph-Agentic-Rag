"""
Agentic RAG Demo — Israel Ibok
Run: python gradio_app.py
A public share link is printed on startup (valid 72 hours).
"""

import sys
import os
import io
import time
import contextlib

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

import gradio as gr

_app = None


def _ensure_ready():
    global _app
    if _app is not None:
        return _app

    chroma_path = os.path.join(os.path.dirname(__file__), "data", "chroma_db")
    if not os.path.exists(chroma_path) or not os.listdir(chroma_path):
        from rag_pipeline import build_vector_store
        build_vector_store()

    from graph import build_graph
    _app = build_graph()
    return _app


def run_query(question: str, history: list):
    if not question.strip():
        return history, ""

    app = _ensure_ready()

    capture = io.StringIO()
    start = time.time()

    with contextlib.redirect_stdout(capture):
        result = app.invoke({
            "question": question,
            "retrieved_docs": [],
            "generation": "",
            "hallucination_count": 0,
            "rewrite_count": 0,
            "route_decision": "",
            "grade_decision": "",
        })

    elapsed = time.time() - start
    answer = result.get("generation", "No answer generated.")
    trace = capture.getvalue().strip()

    full_answer = answer
    if trace:
        full_answer += f"\n\n---\n*{trace}*\n*Completed in {elapsed:.1f}s*"

    history = history + [
        {"role": "user", "content": question},
        {"role": "assistant", "content": full_answer},
    ]
    return history, ""


def load_example(example: str):
    return example


EXAMPLES = [
    "Curious about the guy who built this? Ask me anything about Israel.",
    "What is Plasma Computing Group's GenAI platform?",
    "How does LangGraph implement the Plan → Act → Observe → Respond pattern?",
    "How would you connect LangGraph to a BPMN workflow engine?",
    "What are the key components of a RAG pipeline?",
    "What is Israel's background and working style?",
]

GRAPH_DIAGRAM = """START
  |
[route_question]       ← PLAN
  |            \\
[retrieve_docs]  [generate] (direct)
  |
[grade_documents]      ← OBSERVE
  |            \\
[generate_answer]  [rewrite_query]
  |                      |
  |              [retrieve_docs] (loop)
[grade_hallucination]  ← OBSERVE
  |            \\
 END        [generate] (retry ×2)"""

with gr.Blocks(title="Agentic RAG — Israel Ibok") as demo:

    gr.Markdown("# Agentic RAG Demo &nbsp;·&nbsp; Israel Ibok")

    with gr.Row(equal_height=True):

        # ── Left panel ────────────────────────────────────────────────
        with gr.Column(scale=1, elem_classes=["left-panel"]):

            gr.Markdown("### Agent Graph")
            gr.Code(value=GRAPH_DIAGRAM, language=None, label="", lines=14)

            gr.Markdown("### Suggested Questions")
            suggestion_btns = []
            for q in EXAMPLES:
                btn = gr.Button(q, size="sm", elem_classes=["suggest-btn"])
                suggestion_btns.append((btn, q))

        # ── Right panel — chatbot ──────────────────────────────────────
        with gr.Column(scale=2):

            chatbot = gr.Chatbot(
                label="",
                height=520,
            )

            with gr.Row():
                question_box = gr.Textbox(
                    placeholder="Ask about AI workflows, RAG, LangGraph, Plasma Computing, or Israel...",
                    label="",
                    lines=1,
                    scale=5,
                    autofocus=True,
                )
                send_btn = gr.Button("Send", variant="primary", scale=1)

            clear_btn = gr.Button("Clear chat", size="sm", variant="secondary")

    # ── Wire up interactions ───────────────────────────────────────────
    send_btn.click(
        fn=run_query,
        inputs=[question_box, chatbot],
        outputs=[chatbot, question_box],
    )

    question_box.submit(
        fn=run_query,
        inputs=[question_box, chatbot],
        outputs=[chatbot, question_box],
    )

    clear_btn.click(fn=lambda: ([], ""), outputs=[chatbot, question_box])

    # Clicking a suggested question loads it into the input box
    for btn, q in suggestion_btns:
        btn.click(fn=lambda x=q: x, outputs=question_box)


if __name__ == "__main__":
    print("\nStarting Agentic RAG Demo...")
    print("Building knowledge base and agent graph...\n")
    _ensure_ready()
    print("\nAgent ready. Launching UI...\n")

    demo.launch(
        share=True,
        server_name="localhost",
        show_error=True,
        theme=gr.themes.Soft(),
        css="""
            .left-panel { padding-right: 12px; border-right: 1px solid #e0e0e0; }
            .suggest-btn { margin-bottom: 6px !important; text-align: left !important; }
            footer { display: none !important; }
        """,
    )
