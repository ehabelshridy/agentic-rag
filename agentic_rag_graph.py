"""
agentic_rag_graph.py
---------------------
The full Agentic RAG graph built with LangGraph.

Structure (as approved beforehand):

    START
      -> retrieve
      -> grade_relevance (agentic decision)
           - "not relevant" -> rewrite_query -> retrieve (loop)
           - "relevant"     -> generate_answer
      -> generate_answer
      -> check_groundedness (agentic decision)
           - "hallucinated" -> generate_answer (loop, capped)
           - "grounded"     -> END

All models are open-source AND run online via HuggingFace's hosted
Inference Providers (router.huggingface.co) - nothing is downloaded
to this machine:
  - LLM:        Qwen/Qwen2.5-7B-Instruct      (served by HF Inference Providers)
  - Embeddings: sentence-transformers/all-MiniLM-L6-v2 (HF serverless inference)
  - Vector DB:  ChromaDB (persistent, built by build_vectorstore.py)

A free HF_TOKEN is required for this online mode (the free tier includes
monthly credits for both chat completions and embeddings). Get one at:
https://huggingface.co/settings/tokens
"""

import os
from pathlib import Path
from typing import TypedDict, Literal

# Load HF_TOKEN from a local .env file, if present, before anything else
# reads it from the environment.
try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).parent / ".env")
except ImportError:
    pass

from langchain_huggingface import HuggingFaceEndpointEmbeddings, ChatHuggingFace, HuggingFaceEndpoint
from langchain_community.vectorstores import Chroma
from langgraph.graph import StateGraph, END

# ------------------------------------------------------------------
# Global configuration
# ------------------------------------------------------------------
PERSIST_DIR = "./chroma_db"
COLLECTION_NAME = "customer_support_faqs"
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
LLM_MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct"

TOP_K = 3                    # number of documents returned by the retriever
MAX_REWRITE_ATTEMPTS = 2     # max number of query-rewrite attempts
MAX_GENERATION_ATTEMPTS = 2  # max number of answer-regeneration attempts


def _require_hf_token() -> str:
    token = (
        os.environ.get("HF_TOKEN")
        or os.environ.get("HUGGING_FACE_HUB_TOKEN")
        or os.environ.get("HUGGINGFACE_TOKEN")
    )
    if not token:
        raise RuntimeError(
            "No HF_TOKEN found. This online mode calls HuggingFace's hosted "
            "Inference Providers, which requires a free token.\n"
            "Get one at https://huggingface.co/settings/tokens (the 'Read' "
            "role with 'Make calls to Inference Providers' permission is "
            "enough), then either:\n"
            "  - add it to a .env file in this folder: HF_TOKEN=hf_xxxxxxxxxxxx\n"
            "  - or export it manually: export HF_TOKEN=hf_xxxxxxxxxxxx (Linux/Mac)\n"
            "    $env:HF_TOKEN=\"hf_xxxxxxxxxxxx\" (Windows PowerShell)"
        )
    os.environ["HF_TOKEN"] = token  # normalize the variable name
    return token


# ------------------------------------------------------------------
# 1) Connect to hosted models (no local download - this just sets up
#    lightweight HTTP clients pointed at HuggingFace's router).
# ------------------------------------------------------------------
_require_hf_token()

print(f"Connecting to hosted embedding model: {EMBEDDING_MODEL_NAME} ...")
embeddings = HuggingFaceEndpointEmbeddings(
    model=EMBEDDING_MODEL_NAME,
    task="feature-extraction",
    huggingfacehub_api_token=os.environ["HF_TOKEN"],
)

print(f"Loading vector store from {PERSIST_DIR} ...")
vectorstore = Chroma(
    collection_name=COLLECTION_NAME,
    embedding_function=embeddings,
    persist_directory=PERSIST_DIR,
)
retriever = vectorstore.as_retriever(search_kwargs={"k": TOP_K})

print(f"Connecting to hosted LLM: {LLM_MODEL_NAME} ...")
_llm_endpoint = HuggingFaceEndpoint(
    repo_id=LLM_MODEL_NAME,
    task="text-generation",
    max_new_tokens=300,
    temperature=0.01,  # near-deterministic, helps the grading/checker nodes
    provider="auto",   # let HuggingFace pick the best available Inference Provider
    huggingfacehub_api_token=os.environ["HF_TOKEN"],
)
llm = ChatHuggingFace(llm=_llm_endpoint)
print("✅ Connected. No local model download required.\n")


# ------------------------------------------------------------------
# 2) Graph State definition
# ------------------------------------------------------------------
class GraphState(TypedDict):
    question: str               # the original question (never changes)
    search_query: str           # the query used for retrieval (may change after rewrite)
    documents: list[str]        # retrieved text chunks from ChromaDB
    relevance_decision: str     # "relevant" or "not_relevant"
    rewrite_attempts: int
    answer: str
    groundedness_decision: str  # "grounded" or "hallucinated"
    generation_attempts: int
    final_answer: str


# ------------------------------------------------------------------
# 3) Helper functions: calling the LLM and parsing clean output
# ------------------------------------------------------------------
def call_llm(prompt: str) -> str:
    response = llm.invoke(prompt)
    return response.content.strip()


def extract_decision(text: str, valid_options: list[str]) -> str:
    """Tries to extract a clear single-word decision from the LLM's reply,
    even if the model added extra surrounding text."""
    text_lower = text.lower()
    for option in valid_options:
        if option in text_lower:
            return option
    # Fallback: if the model didn't return anything clear, default to the
    # first (most conservative) option.
    return valid_options[0]


# ------------------------------------------------------------------
# 4) Node definitions
# ------------------------------------------------------------------

def node_retrieve(state: GraphState) -> GraphState:
    """Retriever Node: searches ChromaDB for the closest matching FAQs
    to the current search query."""
    query = state.get("search_query") or state["question"]
    print(f"\n[Retriever] Searching for: {query!r}")

    docs = retriever.invoke(query)
    documents_text = [d.page_content for d in docs]

    print(f"[Retriever] Retrieved {len(documents_text)} result(s).")
    return {**state, "documents": documents_text, "search_query": query}


def node_grade_relevance(state: GraphState) -> GraphState:
    """Agentic Node: evaluates whether the retrieved documents are
    sufficient to answer the question."""
    context = "\n\n".join(state["documents"])
    prompt = f"""You are a precise evaluator. Your only task is to determine whether
the following information is sufficient to answer the user's question.

Question: {state['question']}

Retrieved information:
{context}

Reply with exactly one word: "relevant" if the information is sufficient
and related to the question, or "not_relevant" if it is insufficient
or unrelated.
Answer:"""

    raw_response = call_llm(prompt)
    decision = extract_decision(raw_response, ["not_relevant", "relevant"])

    print(f"[Relevance Grader] Decision: {decision}  (raw: {raw_response[:60]!r})")
    return {**state, "relevance_decision": decision}


def node_rewrite_query(state: GraphState) -> GraphState:
    """Query Rewriter Node: rephrases the question to improve retrieval results."""
    attempts = state.get("rewrite_attempts", 0) + 1
    prompt = f"""The following question did not return sufficient search results in a
customer support FAQ database. Rephrase the question using clearer and
more specific wording that will help find a matching answer, in the same
language as the original question, and with no extra explanation -
only the rephrased question itself.

Original question: {state['question']}

Rephrased question:"""

    new_query = call_llm(prompt)
    # Light cleanup in case the model returns quotes or extra lines
    new_query = new_query.strip().strip('"').split("\n")[0]

    print(f"[Query Rewriter] Attempt #{attempts} - new query: {new_query!r}")
    return {**state, "search_query": new_query, "rewrite_attempts": attempts}


def node_generate_answer(state: GraphState) -> GraphState:
    """Generate Answer Node: produces an answer based solely on the
    retrieved context."""
    context = "\n\n".join(state["documents"])
    prompt = f"""You are a customer support assistant. Answer the user's question relying
only on the information in the context below. Do not add any information
that is not present in the context. If the context does not contain a
clear answer, say so explicitly.

Context:
{context}

Question: {state['question']}

Answer:"""

    answer = call_llm(prompt)
    attempts = state.get("generation_attempts", 0) + 1

    print(f"[Generate Answer] Attempt #{attempts} - answer: {answer[:80]!r}...")
    return {**state, "answer": answer, "generation_attempts": attempts}


def node_check_groundedness(state: GraphState) -> GraphState:
    """Agentic Node: verifies that the generated answer is actually
    grounded in the retrieved context (i.e. no hallucination)."""
    context = "\n\n".join(state["documents"])
    prompt = f"""You are a strict fact-checker. Your task: determine whether the following
answer is fully supported by the given context, or whether it contains
information that is not present in the context (hallucination).

Context:
{context}

Answer to check:
{state['answer']}

Reply with exactly one word: "grounded" if the answer is fully supported
by the context, or "hallucinated" if it contains extra information not
found in the context.
Answer:"""

    raw_response = call_llm(prompt)
    decision = extract_decision(raw_response, ["hallucinated", "grounded"])

    print(f"[Groundedness Checker] Decision: {decision}  (raw: {raw_response[:60]!r})")
    return {**state, "groundedness_decision": decision}


def node_finalize(state: GraphState) -> GraphState:
    """Prepares the final answer for display."""
    return {**state, "final_answer": state["answer"]}


def node_fallback(state: GraphState) -> GraphState:
    """If we exceed the allowed number of attempts, return a clear
    fallback message instead of an unreliable answer or infinite loop."""
    msg = (
        "Sorry, I could not find a reliable and accurate answer to your "
        "question in the current support database. Please try rephrasing "
        "your question or contact our human support team."
    )
    print("[Fallback] Max attempts exceeded - returning fallback message.")
    return {**state, "final_answer": msg}


# ------------------------------------------------------------------
# 5) Routing functions (conditional edges) — this is where the
#    "decision making" logic lives.
# ------------------------------------------------------------------

def route_after_relevance(state: GraphState) -> Literal["generate_answer", "rewrite_query", "fallback"]:
    if state["relevance_decision"] == "relevant":
        return "generate_answer"
    if state.get("rewrite_attempts", 0) >= MAX_REWRITE_ATTEMPTS:
        return "fallback"
    return "rewrite_query"


def route_after_groundedness(state: GraphState) -> Literal["finalize", "generate_answer", "fallback"]:
    if state["groundedness_decision"] == "grounded":
        return "finalize"
    if state.get("generation_attempts", 0) >= MAX_GENERATION_ATTEMPTS:
        return "fallback"
    return "generate_answer"


# ------------------------------------------------------------------
# 6) Build the graph
# ------------------------------------------------------------------
def build_graph():
    graph = StateGraph(GraphState)

    graph.add_node("retrieve", node_retrieve)
    graph.add_node("grade_relevance", node_grade_relevance)
    graph.add_node("rewrite_query", node_rewrite_query)
    graph.add_node("generate_answer", node_generate_answer)
    graph.add_node("check_groundedness", node_check_groundedness)
    graph.add_node("finalize", node_finalize)
    graph.add_node("fallback", node_fallback)

    graph.set_entry_point("retrieve")

    graph.add_edge("retrieve", "grade_relevance")

    graph.add_conditional_edges(
        "grade_relevance",
        route_after_relevance,
        {
            "generate_answer": "generate_answer",
            "rewrite_query": "rewrite_query",
            "fallback": "fallback",
        },
    )

    graph.add_edge("rewrite_query", "retrieve")  # the loop back

    graph.add_edge("generate_answer", "check_groundedness")

    graph.add_conditional_edges(
        "check_groundedness",
        route_after_groundedness,
        {
            "finalize": "finalize",
            "generate_answer": "generate_answer",  # the loop back
            "fallback": "fallback",
        },
    )

    graph.add_edge("finalize", END)
    graph.add_edge("fallback", END)

    return graph.compile()


# ------------------------------------------------------------------
# 7) Interactive demo
# ------------------------------------------------------------------
if __name__ == "__main__":
    app = build_graph()

    print("=" * 60)
    print("Agentic RAG - Customer Support FAQ (online models)")
    print("Type your question (or 'exit' to quit)")
    print("=" * 60)

    while True:
        user_question = input("\nYour question: ").strip()
        if user_question.lower() in {"exit", "quit"}:
            break
        if not user_question:
            continue

        initial_state: GraphState = {
            "question": user_question,
            "search_query": user_question,
            "documents": [],
            "relevance_decision": "",
            "rewrite_attempts": 0,
            "answer": "",
            "groundedness_decision": "",
            "generation_attempts": 0,
            "final_answer": "",
        }

        # recursion_limit higher than the default (25) to make sure the
        # loops (rewrite_query <-> retrieve, generate_answer <-> check_groundedness)
        # have enough room even if both trigger for the same question.
        result = app.invoke(initial_state, config={"recursion_limit": 50})
        print("\n" + "-" * 40)
        print("Final answer:")
        print(result["final_answer"])
        print("-" * 40)
