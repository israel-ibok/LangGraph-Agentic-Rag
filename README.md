# LangGraph Agentic RAG — Intelligent Document Q&A System

A production-ready Agentic RAG pipeline built with LangGraph, demonstrating the full Plan → Act → Observe → Respond loop with hallucination detection and automatic query rewriting.

## What This Does

Takes a user question, decides whether retrieval is needed, retrieves and grades relevant documents, generates a grounded answer, and checks for hallucinations before responding — automatically retrying if the output is not grounded in the source material.

## Architecture

```
START
  ↓
route_question (conditional)
  ├── retrieve → retrieve_documents
  └── generate → generate_answer (direct, no retrieval needed)
        ↓
grade_documents (conditional)
  ├── relevant → generate_answer
  └── not relevant → rewrite_query → retrieve_documents (loop)
        ↓
generate_answer
  ↓
grade_hallucination (conditional)
  ├── grounded → END
  └── hallucinated → generate_answer (retry, max 2)
        ↓
END
```

## Key Features

- Stateful agent graph using LangGraph — each node has a defined role, clear handoffs, and explicit failure handling
- Hybrid retrieval with ChromaDB vector store (swap to Qdrant or Pinecone for production)
- Document grading node filters irrelevant chunks before generation
- Hallucination checking node validates answer against retrieved context
- Query rewrite loop — if retrieval fails, the agent rewrites the question and retries
- Structured JSON output with source attribution
- Live node execution trace visible in terminal

## Stack

- LangGraph — stateful agent state machine
- LangChain — RAG pipeline and LLM abstractions
- ChromaDB — local vector store (Qdrant/Pinecone for production)
- OpenAI GPT-4o-mini — LLM for reasoning nodes
- Gradio — web UI with live execution trace
- Python 3.11+

## Setup

```bash
# Clone the repo
git clone https://github.com/israel-ibok/LangGraph-Agentic-Rag.git
cd LangGraph-Agentic-Rag

# Install dependencies
pip install -r requirements.txt

# Set your OpenAI API key
# Create a .env file with:
OPENAI_API_KEY=your_key_here

# Run the terminal demo
python demo.py

# Or run the Gradio web UI
python gradio_app.py
```

On first run, the vector store is built from the documents in `data/`. Subsequent runs load from the existing store.

## Demo

The terminal demo builds the vector store, prints the graph diagram, and accepts live queries. Each node execution step prints in real time so you can follow the agent's reasoning path.

The Gradio app generates a public share link on startup and shows the node execution trace and final answer side by side.

## Project Structure

```
├── src/
│   ├── rag_pipeline.py   — document ingestion, chunking, embedding, ChromaDB store
│   ├── agent.py          — all 6 node definitions and routing functions
│   └── graph.py          — LangGraph state machine compilation
├── demo.py               — terminal screen share script
├── gradio_app.py         — Gradio web UI
└── data/                 — source documents (chroma_db built on first run)
```

## Design Principles

Financial calculations and hard logic stay deterministic — the LLM handles reasoning and language, not arithmetic. This separation is enforced at the architecture level, not just in prompts.

Built by Israel Ibok — AI Automation Architect
