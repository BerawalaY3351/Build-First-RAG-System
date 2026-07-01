# Build-First-RAG-System

A from-scratch implementation of a Retrieval-Augmented Generation (RAG) pipeline using open-source tools — no external API keys required.

## What is RAG?

RAG (Retrieval-Augmented Generation) is a technique that grounds an LLM's response in a specific knowledge base. Instead of relying purely on the model's training data, it:

1. **Retrieves** the most relevant context from your documents
2. **Augments** the user's query with that context
3. **Generates** an answer using only the provided context

## Knowledge Base

The system uses a company policy manual (`my_knowledge.txt`) covering:
- **WFH Policy** — hybrid schedule with mandatory in-office days (Tue/Wed/Thu)
- **PTO Policy** — 20 days/year for full-time employees, accrues monthly
- **Tech Stack** — Python (backend), React (frontend), React Native (mobile)

## Pipeline Steps

### 1. Chunking

Used LangChain's `RecursiveCharacterTextSplitter` to split the raw text into overlapping chunks:

- `chunk_size=150` — max characters per chunk
- `chunk_overlap=20` — overlap between chunks to preserve context at boundaries
- Splits hierarchically on `\n\n` → `\n` → ` ` to keep semantically related text together

The document produced **5 chunks**.

### 2. Embeddings

Used the `sentence-transformers` library with the `all-MiniLM-L6-v2` model to convert each text chunk into a dense vector:

- Runs entirely **locally** (no API key needed)
- Produces vectors of dimension **384**
- Output shape: `(5, 384)` — one 384-dim vector per chunk

### 3. Vector Store (FAISS)

Used Facebook's FAISS library to store and search the embeddings:

- Index type: `IndexFlatL2` — exact L2 (Euclidean) distance search
- All 5 chunk vectors are added to the index
- At query time, FAISS finds the top-k most similar chunks to the query embedding

### 4. Retrieve, Augment, Generate

The full RAG function `answer_question(query)`:

1. **Retrieve** — embeds the user query with the same `all-MiniLM-L6-v2` model, then searches FAISS for the top 2 most relevant chunks
2. **Augment** — injects the retrieved chunks into a prompt template that instructs the model to answer only from the provided context
3. **Generate** — passes the augmented prompt to `google/flan-t5-small` via HuggingFace's `transformers` pipeline

## Tech Stack

| Component | Library / Model |
|-----------|----------------|
| Text splitting | `langchain-text-splitters` — `RecursiveCharacterTextSplitter` |
| Embeddings | `sentence-transformers` — `all-MiniLM-L6-v2` |
| Vector store | `faiss` — `IndexFlatL2` |
| Text generation | `transformers` — `google/flan-t5-small` |

## How to Run

1. Install dependencies:
   ```bash
   pip install langchain-text-splitters sentence-transformers faiss-cpu transformers
   ```

2. Create `my_knowledge.txt` with your knowledge base content.

3. Open and run `ragSystem.ipynb` cell by cell.
