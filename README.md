# Agentic RAG — Customer Support FAQ

An agentic RAG system built with LangGraph, based on the
`MakTek/Customer_support_faqs_dataset` (200 question/answer pairs),
using entirely open-source models that run **online** via HuggingFace's
hosted Inference Providers (no local model download, no GPU needed),
with a ChatGPT-style web frontend.

## Components

| Component | Choice | Where it runs |
|---|---|---|
| LLM (grading + generation) | `Qwen/Qwen2.5-7B-Instruct` | Hosted online via HF Inference Providers |
| Embedding model | `sentence-transformers/all-MiniLM-L6-v2` | Hosted online via HF Inference Providers |
| Vector DB | ChromaDB | Local (just stores small vectors, not models) |
| Orchestration | LangGraph | Local |
| Backend API | FastAPI (`server.py`) | Local |
| Frontend | Plain HTML / CSS / JavaScript (`frontend/`) | Local |

Both models are fully open-source (Apache 2.0 / Qwen license) — "online"
here just means the model weights run on HuggingFace's servers instead
of being downloaded to your machine. This avoids large downloads and
works even on a low-spec laptop with no GPU.

A free HuggingFace account and token are required (see step 1b below).
This is **not** a paid API key — it only authenticates your requests to
HF's free tier, which includes monthly credits for both embeddings and
chat completions.

## Project structure

```
agentic_rag_faq/
├── requirements.txt
├── .env.example              # template for the required HF token
├── .gitignore
├── build_vectorstore.py     # builds the ChromaDB collection (run once)
├── agentic_rag_graph.py     # the LangGraph pipeline (can run standalone in terminal)
├── server.py                # FastAPI backend, wraps the graph + serves the frontend
└── frontend/
    ├── index.html
    ├── style.css
    └── script.js
```

## How to run

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

No `torch`, `transformers`, or GPU needed — this project only makes
lightweight HTTP calls to HuggingFace's hosted models.

### 1b. Get a free HF token (required)

Both the embedding model and the LLM are called through HuggingFace's
hosted Inference Providers, which requires a free token to authenticate.

```bash
cp .env.example .env
# then edit .env and paste your token:
# HF_TOKEN=hf_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
```

Get a free token at https://huggingface.co/settings/tokens — when
creating it, make sure the **"Make calls to Inference Providers"**
permission is enabled (it's included by default on a standard "Read"
token, but double-check if requests fail with a permissions error).

Both `build_vectorstore.py` and `agentic_rag_graph.py` automatically
load `HF_TOKEN` from this `.env` file.

### 2. Build the ChromaDB vector store

```bash
python build_vectorstore.py
```

This loads the dataset, sends each FAQ to the hosted embedding API to
get its vector, and stores everything in `./chroma_db`. Run it once
(unless you want to rebuild from scratch). Nothing is downloaded —
this just makes ~200 small API calls.

### 3a. Run in the terminal (optional, for quick testing)

```bash
python agentic_rag_graph.py
```

This opens an interactive prompt where you type your question, with
every graph step printed in the terminal.

### 3b. Run the web app (chat UI)

```bash
python server.py
```

This starts the server directly (no need to call `uvicorn` separately —
`server.py` launches it for you). Then open **http://127.0.0.1:8000**
in your browser. You'll see a ChatGPT-style interface: a sidebar with
chat history, a centered message column, and a streaming "thinking"
trace that shows each graph step (retrieving, grading, rewriting if
triggered, generating, checking groundedness) before the final answer
appears.

> If you prefer running it via the `uvicorn` command instead, that
> works too: `uvicorn server:app --host 127.0.0.1 --port 8000`.

The frontend talks to two backend endpoints:
- `POST /api/chat/stream` — Server-Sent Events stream of graph steps + final answer (used by the UI)
- `POST /api/chat` — plain JSON request/response (useful for testing with `curl`)

```bash
curl -X POST http://127.0.0.1:8000/api/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "How can I track my order?"}'
```

## Graph structure

```
START
  -> retrieve
  -> grade_relevance (decision: relevant / not_relevant)
       not_relevant -> rewrite_query -> retrieve (loop, max 2 attempts)
       relevant     -> generate_answer
  -> generate_answer
  -> check_groundedness (decision: grounded / hallucinated)
       hallucinated -> generate_answer (loop, max 2 attempts)
       grounded     -> END
```

If the maximum number of attempts is exceeded at any stage, the graph
routes to a `fallback` node that returns a clear message instead of
an unreliable answer or an infinite loop.

## Performance notes

- Every question triggers at least 3 LLM API calls (grading +
  generation + groundedness check) plus 1+ embedding calls, so each
  response takes a few seconds depending on the hosted provider's
  current load. The web UI shows each step as it completes so the
  wait doesn't feel like a frozen screen.
- HF's free tier includes monthly credits across providers. If you
  see a quota/rate-limit error, wait a bit or check your usage at
  https://huggingface.co/settings/billing.
- `provider="auto"` in `agentic_rag_graph.py` lets HuggingFace pick
  whichever Inference Provider (Groq, Cerebras, Together, Fireworks,
  etc.) is fastest/available for `Qwen2.5-7B-Instruct` at request
  time — you don't need to choose one manually.
- If you'd rather run everything fully offline with no token at all,
  that's also possible by switching back to local model loading with
  `transformers` + `sentence-transformers` — ask if you'd like that
  version instead.
