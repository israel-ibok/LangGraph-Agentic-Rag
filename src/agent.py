"""
Agent — State schema and node definitions for the Agentic RAG graph.
Implements: Plan → Act → Observe → Respond
"""

import os
import sys

# Allow sibling imports when running from /src
sys.path.insert(0, os.path.dirname(__file__))

from dotenv import load_dotenv
from typing import TypedDict, List
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()

# ─── LLM ────────────────────────────────────────────────────────────────────
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)


# ─── State Schema ────────────────────────────────────────────────────────────
class AgentState(TypedDict):
    question: str
    retrieved_docs: List[str]
    generation: str
    hallucination_count: int
    rewrite_count: int
    route_decision: str
    grade_decision: str


# ─── Node: Route Question (PLAN) ─────────────────────────────────────────────
def route_question(state: AgentState) -> AgentState:
    """PLAN: Decide whether to retrieve from knowledge base or answer directly."""
    print("\n[PLAN] Routing question...")
    question = state["question"]

    prompt = ChatPromptTemplate.from_template(
        """You are a routing agent for a personal knowledge base. Decide if a question needs retrieval or can be answered from general knowledge.

Always choose 'retrieve' for:
- Questions about a person (Israel, Israel Ibok, the builder, the developer, who built this, about me, his background, his skills, working style)
- Questions about specific companies (Plasma Computing, Xpectra, C2M)
- Questions about specific technical systems or architectures in this project
- Any question that could be about a specific person even if the name is ambiguous

Only choose 'direct' for clearly general knowledge questions (e.g. "what is Python?", "what year is it?")

When in doubt, choose 'retrieve'.

Question: {question}

Respond with ONLY one word: 'retrieve' or 'direct'"""
    )

    response = llm.invoke(prompt.format_messages(question=question))
    decision = response.content.strip().lower()

    if "direct" in decision:
        print("[PLAN] Decision: Answer directly (no retrieval needed)")
        return {**state, "route_decision": "direct"}
    else:
        print("[PLAN] Decision: Retrieve from knowledge base")
        return {**state, "route_decision": "retrieve"}


# ─── Node: Retrieve Documents (ACT) ──────────────────────────────────────────
def retrieve_documents(state: AgentState) -> AgentState:
    """ACT: Retrieve relevant documents from ChromaDB vector store."""
    print("\n[ACT] Retrieving documents from knowledge base...")
    from rag_pipeline import get_retriever

    retriever = get_retriever()
    docs = retriever.invoke(state["question"])
    doc_texts = [doc.page_content for doc in docs]

    print(f"[ACT] Retrieved {len(doc_texts)} chunks")
    for i, doc in enumerate(doc_texts[:2]):
        print(f"  Chunk {i+1}: {doc[:120]}...")

    return {**state, "retrieved_docs": doc_texts}


# ─── Node: Grade Documents (OBSERVE) ─────────────────────────────────────────
def grade_documents(state: AgentState) -> AgentState:
    """OBSERVE: Grade whether retrieved documents are relevant to the question."""
    print("\n[OBSERVE] Grading document relevance...")

    prompt = ChatPromptTemplate.from_template(
        """Are these documents relevant to answering the question?

Question: {question}

Documents:
{docs}

Respond with ONLY one word: 'relevant' or 'not_relevant'"""
    )

    docs_text = "\n\n".join(state["retrieved_docs"][:2])
    response = llm.invoke(prompt.format_messages(
        question=state["question"],
        docs=docs_text,
    ))

    raw = response.content.strip().lower()
    decision = "relevant" if "relevant" in raw and "not" not in raw else "not_relevant"
    print(f"[OBSERVE] Relevance grade: {decision}")
    return {**state, "grade_decision": decision}


# ─── Node: Rewrite Query (PLAN) ───────────────────────────────────────────────
def rewrite_query(state: AgentState) -> AgentState:
    """PLAN: Rewrite question to improve retrieval when first attempt fails."""
    print("\n[PLAN] Documents not relevant — rewriting query...")

    prompt = ChatPromptTemplate.from_template(
        """The retrieved documents were not relevant to the question.
Rewrite the question to be more specific and improve retrieval quality.

Original question: {question}

Rewritten question:"""
    )

    response = llm.invoke(prompt.format_messages(question=state["question"]))
    new_question = response.content.strip()
    count = state.get("rewrite_count", 0) + 1
    print(f"[PLAN] Rewritten: {new_question}")
    return {**state, "question": new_question, "rewrite_count": count}


# ─── Node: Generate Answer (RESPOND) ─────────────────────────────────────────
def generate_answer(state: AgentState) -> AgentState:
    """RESPOND: Generate a grounded answer from retrieved documents."""
    print("\n[RESPOND] Generating answer...")

    if state.get("retrieved_docs"):
        context = "\n\n".join(state["retrieved_docs"])
        prompt = ChatPromptTemplate.from_template(
            """You are a helpful assistant. Answer the question by synthesizing and summarizing the provided context.
You may paraphrase and connect ideas from the context — you do not need to quote it verbatim.
Only say you don't have enough information if the context is completely unrelated to the question.

Context:
{context}

Question: {question}

Answer:"""
        )
        response = llm.invoke(prompt.format_messages(
            context=context,
            question=state["question"],
        ))
    else:
        prompt = ChatPromptTemplate.from_template(
            "Answer this question clearly and concisely: {question}"
        )
        response = llm.invoke(prompt.format_messages(question=state["question"]))

    print("[RESPOND] Answer generated")
    return {**state, "generation": response.content}


# ─── Node: Grade Hallucination (OBSERVE) ─────────────────────────────────────
def grade_hallucination(state: AgentState) -> AgentState:
    """OBSERVE: Check if the answer is grounded in the retrieved documents."""
    print("\n[OBSERVE] Checking answer for hallucinations...")

    if not state.get("retrieved_docs"):
        print("[OBSERVE] No retrieval used — skipping hallucination check")
        return {**state, "grade_decision": "grounded"}

    prompt = ChatPromptTemplate.from_template(
        """Does this answer stay within what the provided documents say?

Rules:
- Answer 'grounded' if the answer is a reasonable synthesis or summary of the documents, even if it paraphrases.
- Answer 'grounded' if the answer says it does not have enough information (that is a safe, honest response).
- Answer 'hallucinated' ONLY if the answer states specific facts that directly contradict the documents.

Documents:
{docs}

Answer to check:
{generation}

Respond with ONLY one word: 'grounded' or 'hallucinated'"""
    )

    docs_text = "\n\n".join(state["retrieved_docs"])
    response = llm.invoke(prompt.format_messages(
        docs=docs_text,
        generation=state["generation"],
    ))

    raw = response.content.strip().lower()
    decision = "grounded" if "grounded" in raw else "hallucinated"
    count = state.get("hallucination_count", 0) + 1
    print(f"[OBSERVE] Hallucination check: {decision} (attempt {count})")
    return {**state, "grade_decision": decision, "hallucination_count": count}


# ─── Routing Functions ────────────────────────────────────────────────────────
def route_after_question(state: AgentState) -> str:
    return state.get("route_decision", "retrieve")


def route_after_grading(state: AgentState) -> str:
    if state.get("grade_decision") == "relevant":
        return "generate"
    if state.get("rewrite_count", 0) >= 1:
        print("[OBSERVE] Rewrite limit reached — proceeding to generate with best available docs.")
        return "generate"
    return "rewrite"


def route_after_hallucination_check(state: AgentState) -> str:
    if state.get("grade_decision") == "grounded":
        return "end"
    if state.get("hallucination_count", 0) >= 2:
        print("[OBSERVE] Max retries reached. Returning best available answer.")
        return "end"
    return "regenerate"
