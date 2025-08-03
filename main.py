# /main.py

import os
import logging
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

# This must be the first import to ensure environment variables are loaded
from config import config 
from services.initialise_vectordb import initialize_pinecone, setup_knowledge_base, get_vector_store
from services.initialise_llm import initialize_llm_and_embeddings, create_rag_chain

# --- Global Variables ---
# These are initialized on startup and can be used by your application's API endpoints.
LLM = None
EMBEDDINGS = None
VECTOR_STORE = None
RAG_CHAIN = None

def startup():
    """
    Initializes all services for the backend application.
    Call this function once when the application starts.
    """
    global LLM, EMBEDDINGS, VECTOR_STORE, RAG_CHAIN

    logging.info("Backend application starting up...")
    
    LLM, EMBEDDINGS = initialize_llm_and_embeddings()
    pinecone_index = initialize_pinecone()
    setup_knowledge_base(pinecone_index, EMBEDDINGS)
    VECTOR_STORE = get_vector_store(pinecone_index, EMBEDDINGS)
    RAG_CHAIN = create_rag_chain(LLM, VECTOR_STORE)

    logging.info("Startup complete. Application is ready.")

class EmptyRetriever:
    """A dummy retriever that returns no documents."""
    def get_relevant_documents(self, query):
        return []

def ask_question(question: str, prioritized_docs: list = None):
    """
    Uses the initialized RAG chain to answer a question.

    Args:
        question (str): The user's question.
        prioritized_docs (list, optional): LangChain Documents for prioritized context.
    
    Returns:
        str: The answer from the RAG chain.
    """
    if not RAG_CHAIN:
        logging.error("RAG Chain not initialized. Run startup() first.")
        return "Error: Application is not ready."

    if prioritized_docs:
        logging.info("Creating temporary retriever for prioritized context.")
        prioritized_vector_store = FAISS.from_documents(prioritized_docs, EMBEDDINGS)
        prioritized_retriever = prioritized_vector_store.as_retriever()
    else:
        # Create a "dummy" retriever that returns nothing if no priority docs are given.
        prioritized_retriever = EmptyRetriever()

    chain_input = {
        "question": question,
        "query_doc_retriever": prioritized_retriever
    }
    
    logging.info(f"Invoking RAG chain for question: '{question}'")
    answer = RAG_CHAIN.invoke(chain_input)
    return answer

if __name__ == "__main__":
    # Ensure the documents directory exists
    if not os.path.exists(config.DOCUMENT_DIRECTORY):
        os.makedirs(config.DOCUMENT_DIRECTORY)
        logging.warning(f"Created '{config.DOCUMENT_DIRECTORY}' directory. Please add your PDF files there.")

    # 1. Run the startup sequence
    startup()

    # 2. Example usage
    if RAG_CHAIN:
        print("\n✅ RAG System Initialized. You can now ask questions.")
        
        # Example 1: Query without prioritized context
        print("\n--- Querying General Knowledge Base ---")
        general_question = "What is the policy on accident coverage?" # Change to a relevant question for your docs
        response = ask_question(general_question)
        print(f"Q: {general_question}")
        print(f"A: {response}")

        # Example 2: Query WITH prioritized context
        print("\n--- Querying with Prioritized Context ---")
        priority_question = "What is the new benefit introduced in 2025?"
        priority_docs = [
            Document(page_content="A new cyber insurance benefit was added in 2025. It covers up to $50,000 in digital asset loss.", metadata={"source": "internal_memo.pdf"})
        ]
        response_with_priority = ask_question(priority_question, prioritized_docs=priority_docs)
        print(f"Q: {priority_question}")
        print(f"A: {response_with_priority}")