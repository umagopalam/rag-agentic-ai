from fastapi import FastAPI
from pydantic import BaseModel

from src.graph import graph


app = FastAPI(
    title="Agentic AI RAG API",
    description="RAG chatbot grounded in the Agentic AI eBook",
    version="1.0.0"
)


class ChatRequest(BaseModel):
    query: str


@app.get("/")
def root():
    return {
        "message": "Agentic AI RAG API is running"
    }


@app.post("/chat")
def chat(request: ChatRequest):

    result = graph.invoke({
        "question": request.query,
        "context": [],
        "answer": "",
        "score": 0.0
    })

    retrieved_chunks = [
        doc.page_content
        for doc in result["context"]
    ]

    return {
        "answer": result["answer"],
        "retrieved_chunks": retrieved_chunks,
        "confidence_score": result["score"]
    }