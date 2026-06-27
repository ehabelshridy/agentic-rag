"""
build_vectorstore.py
---------------------
Loads the "MakTek/Customer_support_faqs_dataset" from HuggingFace,
converts it into Documents, and builds a persistent ChromaDB collection
using an open-source embedding model (sentence-transformers/all-MiniLM-L6-v2)
called online through HuggingFace's hosted Inference Providers - the
model weights are never downloaded to this machine.

Requires a free HF_TOKEN, loaded automatically from a local .env file.
Get one at: https://huggingface.co/settings/tokens

Run this once to build the vector store:
    python build_vectorstore.py
"""

import os
from pathlib import Path

from dotenv import load_dotenv
load_dotenv(Path(__file__).parent / ".env")

from datasets import load_dataset
from langchain_huggingface import HuggingFaceEndpointEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document

PERSIST_DIR = "./chroma_db"
COLLECTION_NAME = "customer_support_faqs"
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


def load_faq_documents() -> list[Document]:
    """Loads the dataset and converts each row into a Document.

    Each entry has a 'question' and 'answer' field. We store the question
    as page_content (this is what gets embedded and searched), and store
    the answer in metadata so we can return it in full when answering.
    We also store a combined question+answer string to use as fallback
    context for the LLM.
    """
    print("[1/3] Loading dataset from HuggingFace ...")
    ds = load_dataset("MakTek/Customer_support_faqs_dataset")
    split = ds["train"]  # this dataset usually has a single split

    documents = []
    for i, row in enumerate(split):
        question = row["question"].strip()
        answer = row["answer"].strip()

        combined_text = f"Question: {question}\nAnswer: {answer}"

        doc = Document(
            page_content=combined_text,
            metadata={
                "id": str(i),
                "question": question,
                "answer": answer,
                "source": "MakTek/Customer_support_faqs_dataset",
            },
        )
        documents.append(doc)

    print(f"      Loaded {len(documents)} question/answer pairs.")
    return documents


def build_chroma_vectorstore(documents: list[Document]) -> Chroma:
    """Builds a persistent ChromaDB collection from the documents, using
    HuggingFace's hosted embedding endpoint (no local model download)."""
    print(f"[2/3] Connecting to hosted embedding model: {EMBEDDING_MODEL_NAME} ...")
    embeddings = HuggingFaceEndpointEmbeddings(
        model=EMBEDDING_MODEL_NAME,
        task="feature-extraction",
        huggingfacehub_api_token=os.environ["HF_TOKEN"],
    )

    print(f"[3/3] Building ChromaDB at: {PERSIST_DIR} ...")
    vectorstore = Chroma.from_documents(
        documents=documents,
        embedding=embeddings,
        collection_name=COLLECTION_NAME,
        persist_directory=PERSIST_DIR,
    )
    print(f"      Done! Number of vectors stored: {vectorstore._collection.count()}")
    return vectorstore


def main():
    if os.path.exists(PERSIST_DIR) and os.listdir(PERSIST_DIR):
        print(f"Warning: the folder '{PERSIST_DIR}' already exists and contains data.")
        answer = input("Do you want to delete it and rebuild from scratch? (y/n): ").strip().lower()
        if answer != "y":
            print("Cancelled. Using the existing vector store as-is.")
            return
        import shutil
        shutil.rmtree(PERSIST_DIR)

    documents = load_faq_documents()
    build_chroma_vectorstore(documents)
    print("\n✅ Vector store is ready. You can now run agentic_rag_graph.py.")


if __name__ == "__main__":
    main()