# /services/llm_service.py

import logging
import google.generativeai as genai
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from config import config

def initialize_llm_and_embeddings():
    """
    Initializes the Google Generative AI models (LLM and Embeddings).
    
    Returns:
        tuple: (llm_instance, embeddings_model_instance)
    """
    try:
        genai.configure(api_key=config.GOOGLE_API_KEY)
        
        # Note: The model 'gemini-2.0-flash-lite' was replaced with the standard 'gemini-1.5-flash'.
        llm = ChatGoogleGenerativeAI(model=config.LLM_MODEL, temperature=0.2, convert_system_message_to_human=True)
        embeddings_model = GoogleGenerativeAIEmbeddings(model=config.EMBEDDING_MODEL)
        
        logging.info("LLM and Embedding models initialized successfully.")
        return llm, embeddings_model
    except Exception as e:
        logging.error(f"Failed to initialize Google AI models: {e}")
        raise

def create_rag_chain(llm, vector_store):
    """
    Creates the complete RAG chain with a retriever.
    """
    logging.info("Creating RAG chain...")

    pinecone_retriever = vector_store.as_retriever(
        search_type=config.RETRIEVER_SEARCH_TYPE,
        search_kwargs=config.RETRIEVER_SEARCH_KWARGS
    )

    template = """You are an expert insurance policy assistant. Answer each question based ONLY on the context below.

Give Answer in sentences with proper grammar and Human.

Try to include numbers and stats whereever possible.

Use PRIORITIZED CONTEXT if it contains the answer.

If not, refer to GENERAL KNOWLEDGE BASE CONTEXT.

If the sources conflict, PRIORITIZED CONTEXT prevails.

If the answer is not found, respond: "The answer could not be found in the provided documents."

Also given output should be concise and precise and should not exceed 30 words.

PRIORITIZED CONTEXT:
{prioritized_context}

GENERAL KNOWLEDGE BASE CONTEXT:
{general_context}

QUESTIONS:
{questions}"""
    prompt = PromptTemplate.from_template(template)

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    rag_chain = (
        {
            "prioritized_context": lambda x: format_docs(x['query_doc_retriever'].get_relevant_documents(x['question'])),
            "general_context": lambda x: format_docs(pinecone_retriever.get_relevant_documents(x['question'])),
            "questions": lambda x: x['question']
        }
        | prompt
        | llm
        | StrOutputParser()
    )

    logging.info("Hybrid Q&A chain is ready.")
    return rag_chain