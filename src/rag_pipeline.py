"""
RAG Pipeline — Document ingestion, chunking, embedding, and retrieval.
Uses LangChain + ChromaDB + OpenAI embeddings.
"""

import os
from dotenv import load_dotenv
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma

load_dotenv()

CHROMA_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "chroma_db")
DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "data")


def build_vector_store():
    """Load documents from /data, chunk them, embed them, store in Chroma."""
    loader = DirectoryLoader(
        DATA_PATH,
        glob="**/*.txt",
        loader_cls=TextLoader,
        loader_kwargs={"encoding": "utf-8"},
        show_progress=True,
    )
    documents = loader.load()
    print(f"Loaded {len(documents)} documents")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=100,
        length_function=len,
    )
    chunks = splitter.split_documents(documents)
    print(f"Created {len(chunks)} chunks from {len(documents)} documents")

    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=CHROMA_PATH,
    )
    print(f"Stored {len(chunks)} chunks in ChromaDB at {CHROMA_PATH}")
    return vector_store


def get_retriever():
    """Load existing Chroma store and return a retriever."""
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    vector_store = Chroma(
        persist_directory=CHROMA_PATH,
        embedding_function=embeddings,
    )
    retriever = vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 6},
    )
    return retriever


if __name__ == "__main__":
    build_vector_store()
    print("RAG pipeline ready.")
