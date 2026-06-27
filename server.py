"""
server.py
---------------------
FastAPI backend that exposes the Agentic RAG graph (from agentic_rag_graph.py)
over HTTP so the web frontend can talk to it.

Endpoints:
  GET  /                  -> serves the frontend (index.html)
  GET  /api/health        -> simple health check
  POST /api/chat/stream   -> Server-Sent Events stream of graph step updates
                              + the final answer (used by the chat UI)
  POST /api/chat          -> non-streaming version, returns the final answer
                              as plain JSON (useful for testing with curl)

Run with:
    uvicorn server:app --host 0.0.0.0 --port 8000

Then open http://localhost:8000 in your browser.
"""

import asyncio
import json
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

# Reuses everything already defined in agentic_rag_graph.py:
# model loading, GraphState, all nodes, and build_graph().
from agentic_rag_graph import build_graph, GraphState

app = FastAPI(title="Agentic RAG - Customer Support FAQ")

# Allow the frontend to call the API even if served from a different
# origin/port during development.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Build the compiled LangGraph app once at startup (model loading happens
# when agentic_rag_graph is imported above, so this is just graph wiring).
rag_app = build_graph()

FRONTEND_DIR = Path(__file__).parent / "frontend"

# Human-readable labels shown in the UI for each internal node name.
NODE_LABELS = {
    "retrieve": "Searching the FAQ database...",
    "grade_relevance": "Checking if results are relevant...",
    "rewrite_query": "Rephrasing your question...",
    "generate_answer": "Drafting an answer...",
    "check_groundedness": "Double-checking the answer...",
    "finalize": "Finalizing...",
    "fallback": "Could not verify an answer...",
}


class ChatRequest(BaseModel):
    message: str


def make_initial_state(question: str) -> GraphState:
    return {
        "question": question,
        "search_query": question,
        "documents": [],
        "relevance_decision": "",
        "rewrite_attempts": 0,
        "answer": "",
        "groundedness_decision": "",
        "generation_attempts": 0,
        "final_answer": "",
    }


@app.get("/api/health")
def health():
    return {"status": "ok"}


@app.post("/api/chat")
def chat(req: ChatRequest):
    """Non-streaming endpoint: runs the full graph and returns the final answer."""
    initial_state = make_initial_state(req.message)
    result = rag_app.invoke(initial_state, config={"recursion_limit": 50})
    return {"answer": result["final_answer"]}


@app.post("/api/chat/stream")
async def chat_stream(req: ChatRequest):
    """Streaming endpoint used by the chat UI.

    Streams one Server-Sent Event per graph node as it executes (so the
    frontend can show live "thinking" steps, similar to ChatGPT), then a
    final event with the complete answer.
    """
    initial_state = make_initial_state(req.message)

    async def event_generator():
        try:
            # LangGraph's .stream() yields one update per node as it runs.
            # We run the (sync) generator in a thread so it doesn't block
            # the asyncio event loop.
            loop = asyncio.get_event_loop()
            stream_iter = rag_app.stream(
                initial_state, config={"recursion_limit": 50}
            )

            final_state = None
            while True:
                step = await loop.run_in_executor(None, _next_or_none, stream_iter)
                if step is None:
                    break

                for node_name, node_output in step.items():
                    label = NODE_LABELS.get(node_name, node_name)
                    event = {"type": "step", "node": node_name, "label": label}
                    yield f"data: {json.dumps(event)}\n\n"
                    final_state = node_output

            answer = (final_state or {}).get("final_answer", "")
            final_event = {"type": "final", "answer": answer}
            yield f"data: {json.dumps(final_event)}\n\n"

        except Exception as exc:  # surface backend errors to the UI instead of hanging
            error_event = {"type": "error", "message": str(exc)}
            yield f"data: {json.dumps(error_event)}\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")


def _next_or_none(iterator):
    """Helper to pull one item from a sync generator, or None when exhausted."""
    try:
        return next(iterator)
    except StopIteration:
        return None


# Serve the frontend (index.html, style.css, script.js) at the root path.
# This must be mounted last so it doesn't shadow the /api routes above.
app.mount("/", StaticFiles(directory=str(FRONTEND_DIR), html=True), name="frontend")
