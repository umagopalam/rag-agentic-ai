from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore

from src.config import PINECONE_INDEX_NAME


PDF_PATH = Path("data/Ebook-Agentic-AI.pdf")


def ingest_documents():
    print("Loading PDF...")

    loader = PyPDFLoader(str(PDF_PATH))
    documents = loader.load()

    print(f"Loaded {len(documents)} pages.")

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )

    chunks = text_splitter.split_documents(documents)

    print(f"Created {len(chunks)} chunks.")

    print("Loading local embedding model...")

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-mpnet-base-v2"
    )

    print("Uploading embeddings to Pinecone...")

    PineconeVectorStore.from_documents(
        documents=chunks,
        embedding=embeddings,
        index_name=PINECONE_INDEX_NAME
    )

    print("Ingestion completed successfully!")


if __name__ == "__main__":
    ingest_documents()
