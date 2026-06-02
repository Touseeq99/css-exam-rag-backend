# 📚 CSS Exam RAG Backend

> A production-ready hybrid retrieval-augmented generation (RAG) backend for Pakistan Civil Services (CSS) exam preparation — built with FastAPI, Qdrant, and LangChain, featuring dense + sparse search, streaming responses, and startup chain pre-warming.

---

## 🔴 The Problem

CSS exam preparation involves mastering thousands of pages of political science, history, law, and general knowledge material. Students need to:

- Quickly retrieve relevant content from large document collections
- Get contextual, structured answers — not just raw text
- Receive responses tuned to the CSS exam context, not generic answers

Generic chatbots lack domain focus. Keyword search misses semantic meaning. A simple vector search alone ignores exact-match signals that matter for names, dates, and facts in exam material.

---

## ✅ The Solution

A FastAPI backend that ingests PDF study materials into Qdrant and answers student queries using **hybrid retrieval** — combining dense semantic search with sparse BM25 keyword matching — then passes retrieved context to **Llama 3.1 via Groq** for fast, structured, CSS-tuned responses streamed back to the client in real time.

---

## 🏗️ Architecture

```
                        Client
                           │
                    POST /search
                           │
                           ▼
              ┌────────────────────────┐
              │      FastAPI App        │
              │   (main.py)            │
              │                        │
              │  startup → pre-warm    │
              │  chain on boot         │
              └───────────┬────────────┘
                          │
              ┌───────────▼────────────┐
              │   Retrieval Chain       │
              │   (ingestion.py)        │
              │                        │
              │  ┌──────────────────┐  │
              │  │  Qdrant Vector   │  │
              │  │  Store           │  │
              │  │                  │  │
              │  │  Dense:          │  │
              │  │  Google          │  │
              │  │  embedding-001   │  │
              │  │  (768-D cosine)  │  │
              │  │                  │  │
              │  │  Sparse:         │  │
              │  │  FastEmbed BM25  │  │
              │  │                  │  │
              │  │  Mode: HYBRID    │  │
              │  └──────────────────┘  │
              │          │             │
              │  MMR Retriever (k=4)   │
              │  lambda_mult=0.5       │
              │          │             │
              │  ┌──────────────────┐  │
              │  │  ChatGroq LLM    │  │
              │  │  llama-3.1-8b    │  │
              │  │  temp=0.3        │  │
              │  │  max_tokens=2048 │  │
              │  └──────────────────┘  │
              └───────────┬────────────┘
                          │
               StreamingResponse
               (text/event-stream)
                          │
               ┌──────────▼──────────┐
               │  JSON event chunks   │
               │  - metadata          │
               │  - content (3-char)  │
               │  - sources           │
               └─────────────────────┘
```

### Ingestion Pipeline

```
PDF Documents (≤64 pages each)
         │
         ▼
RecursiveCharacterTextSplitter
chunk_size=1024, overlap=256
         │
         ▼
Parallel embedding:
  Dense  → Google embedding-001 (768-D)
  Sparse → FastEmbed BM25
         │
         ▼
Qdrant Collection
  vectors_config:  { "dense":  cosine, 768-D }
  sparse_vectors:  { "sparse": on-disk=False  }
         │
    UUIDs assigned per chunk
```

---

## 🤔 Why I Chose What I Chose

### Hybrid Retrieval (Dense + Sparse)
Pure vector search is great for semantic similarity but can miss exact keywords — critical for CSS where precise names, dates, and legal terms matter. BM25 (sparse) handles exact-match recall. Combining both through Qdrant's hybrid mode gives the best of both worlds. The `alpha=0.5` default balances the two equally; this can be tuned per query type.

### MMR (Maximal Marginal Relevance)
Standard top-k retrieval often returns near-duplicate chunks. MMR diversifies the retrieved set by penalizing similarity to already-selected documents. For long-form exam answers that need breadth of coverage, this significantly improves output quality.

### Groq + Llama 3.1 8B Instant
Groq's inference hardware delivers extremely low latency. For a real-time streaming use case, this matters more than marginal quality improvements from a larger model. Llama 3.1 8B Instant hits the sweet spot for CSS-level reasoning at production speed.

### Chain Pre-warming on Startup
The first call to a LangChain retrieval chain is slow — models load, connections establish, caches warm up. Pre-warming at startup via `@app.on_event("startup")` with a dummy query means the first real user request is fast. The warmed chain is cached globally so subsequent calls reuse it without re-initializing.

### Streaming via `text/event-stream`
Rather than waiting for the full response, the backend simulates streaming by chunking the answer into 3-character slices with a 1ms delay. This gives a typewriter effect on the frontend and makes long answers feel responsive. Source documents are emitted as a final JSON event.

### FastAPI + Uvicorn with 4 Workers
The `Procfile` runs 4 workers for concurrency. Each worker handles requests independently, so long-running retrieval calls don't block others. `gunicorn` manages the workers in production (Render deployment).

### Qdrant Cloud
Managed Qdrant removes infrastructure overhead. The collection is created once on first ingestion; subsequent calls reuse the existing collection, preserving previously indexed documents.

---

## 🚀 Getting Started

### Prerequisites

- Python 3.13+
- [Qdrant Cloud](https://cloud.qdrant.io/) account (or self-hosted)
- [Groq API key](https://console.groq.com/)
- [Google AI Studio API key](https://aistudio.google.com/app/apikey)

### Installation

```bash
git clone https://github.com/Touseeq99/css-rag-backend.git
cd css-rag-backend
pip install -r requirements.txt
```

### Environment Setup

```bash
cp .env.example .env
```

Edit `.env`:

```env
QDRANT_URL=https://your-cluster.qdrant.io
QDRANT_API_KEY=your_qdrant_key
COLLECTION_NAME=CSSDOCS
DIRECTORY_PATH=documents
DENSE_MODEL_NAME=models/embedding-001
GROQ_API_KEY=your_groq_key
GOOGLE_API_KEY=your_google_key
```

### Ingest Documents

Place your PDF files in the `documents/` folder, then run:

```bash
python ingestion.py
```

This loads PDFs (first 64 pages each), chunks them, embeds with dense + sparse models, and uploads to Qdrant. Run once; re-run to add new documents.

### Run Locally

```bash
uvicorn main:app --reload
```

The API will be available at `http://localhost:8000`.

---

## 📡 API Reference

### `GET /`
Health check — confirms the service is running.

### `GET /health`
Returns service status and timestamp.

### `POST /search`

**Request:**
```json
{
  "query": "How does Plato critique Athenian democracy?",
  "k": 4
}
```

**Response:** `text/event-stream` with three event types:

```json
{ "type": "metadata", "processing_time": "1.23 seconds" }
{ "type": "content", "content": "Pla" }
{ "type": "content", "content": "to " }
...
{ "type": "sources", "sources": [
    { "page_number": 12, "page_label": "Page 12", "source": "documents/plato.pdf" }
  ]
}
```

---

## 📁 Project Structure

```
css-rag-backend/
├── main.py             # FastAPI app — routes, streaming, startup pre-warm
├── ingestion.py        # Document loading, chunking, embedding, Qdrant upload
│                       # Also: hybrid retriever + LangChain chain construction
├── retrival.py         # Standalone retrieval test script
├── .env.example        # Environment variable template
├── .dockerignore       # Docker build exclusions
├── Procfile            # Render deployment: uvicorn with 4 workers
├── render.yaml         # Render service configuration
├── runtime.txt         # Python version pin for Render
├── pyproject.toml      # Project metadata and dependencies (uv)
├── requirements.txt    # pip-compatible dependency list
└── README.md
```

---

## ☁️ Deployment

This project is configured for [Render](https://render.com/):

```bash
# Build
pip install -r requirements.txt

# Start
uvicorn main:app --host 0.0.0.0 --port $PORT --workers 4
```

Set the following environment variables in Render's dashboard:
`QDRANT_URL`, `QDRANT_API_KEY`, `GROQ_API_KEY`, `GOOGLE_API_KEY`

---

## 🛠️ Built With

- [FastAPI](https://fastapi.tiangolo.com/) — async web framework
- [LangChain](https://langchain.com/) — retrieval chain orchestration
- [Qdrant](https://qdrant.tech/) — hybrid vector store
- [FastEmbed](https://qdrant.github.io/fastembed/) — BM25 sparse embeddings
- [Google Generative AI](https://ai.google.dev/) — dense embeddings (embedding-001)
- [Groq](https://groq.com/) — ultra-fast LLM inference
- [Llama 3.1 8B Instant](https://ai.meta.com/llama/) — answer generation
- [Unstructured](https://unstructured.io/) — document parsing

---

## 👤 Author

**Touseeq Ahmed**  
AI Engineer | [GitHub](https://github.com/Touseeq99)
