# Agentic AI RAG Chatbot

A Retrieval-Augmented Generation (RAG) chatbot built around an Agentic AI eBook.

The system retrieves relevant information from the provided Agentic AI knowledge base and generates answers based only on the retrieved context. For questions outside the knowledge base, the system responds that it does not have enough information to answer.

## Features

- PDF document ingestion
- Text chunking using Recursive Character Text Splitter
- Local Hugging Face embeddings
- Pinecone vector database for similarity search
- Query-aware retrieval and relevance reranking
- LangGraph-based retrieval and generation workflow
- Context-grounded responses
- Out-of-context question refusal
- FastAPI REST API
- Retrieved chunks and confidence score included in API response

## Architecture

```text
User Query
    |
    v
FastAPI /chat
    |
    v
LangGraph
    |
    v
Query-Aware Retrieval
    |
    v
Pinecone Vector Database
    |
    v
Relevant Context
    |
    v
Grounded Answer Generation
    |
    v
Response
    |
    +-- Answer
    +-- Retrieved Chunks
    +-- Confidence Score