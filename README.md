# Medical RAG System (Python 3.9)

A lightweight retrieval-augmented generation (RAG) project for local medical knowledge QA.

It supports:
- Local knowledge files (`.txt`, `.pdf`)
- Local vector database (`ChromaDB`)
- OpenAI API or local Ollama model
- Streamlit chat UI with source traceability

## Tech Stack

- Python 3.9
- LangChain
- ChromaDB
- Embedding model: `shibing624/text2vec-base-chinese`
- LLM provider: OpenAI / Ollama
- Frontend: Streamlit

## Requirements

```txt
langchain==0.2.16
langchain-community==0.2.16
langchain-openai==0.1.23
langchain-ollama==0.1.3
langchain-chroma==0.1.4
langchain-huggingface==0.0.3
chromadb==0.5.3
sentence-transformers==3.0.1
pypdf==4.3.1
streamlit==1.37.1
python-dotenv==1.0.1
tqdm==4.66.5
requests==2.32.3
beautifulsoup4==4.12.3
lxml==5.3.0
greenlet==3.0.3
posthog<4
```

## Project Structure

```text
rag/
├── app.py
├── requirements.txt
├── .env.example
├── .gitignore
├── README.md
├── config/
│   ├── __init__.py
│   └── settings.py
├── core/
│   ├── __init__.py
│   ├── data_loader.py
│   └── rag_engine.py
├── scripts/
│   └── fetch_medical_docs.py
├── data/
│   └── knowledge/
└── docs/
    └── index.html
```

## Core Modules

- `config/settings.py`: centralized runtime settings.
- `core/data_loader.py`: load docs, split text, generate embeddings, persist vectors.
- `core/rag_engine.py`: retrieve top-k context and generate answer with strict context grounding.
- `app.py`: Streamlit chat app, upload docs, rebuild index, show sources.
- `scripts/fetch_medical_docs.py`: fetch public medical pages and save as local text files.

## Retrieval Principle

Vector retrieval commonly uses cosine similarity:

```text
cosine_similarity(A, B) = (A · B) / (||A|| * ||B||)
```

## Quick Start

### 1) Install dependencies

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
pip install -r requirements.txt
```

### 2) Configure environment

```bash
copy .env.example .env
```

Then set:
- `OPENAI_API_KEY` if using OpenAI
- or install and start Ollama if using local model

### 3) Prepare knowledge documents

Option A (auto fetch):

```bash
python scripts/fetch_medical_docs.py
```

Option B (manual):
- Put local `.pdf` / `.txt` files under `data/knowledge/`

### 4) Build vector store

```bash
python -m core.data_loader
```

### 5) CLI test

```bash
python -m core.rag_engine --question "高血压常见危险因素有哪些？" --provider openai
```

### 6) Run web app

```bash
streamlit run app.py
```

## GitHub

```bash
git init
git add .
git commit -m "feat: build medical rag system"
git branch -M main
git remote add origin https://github.com/liuyaowei-ai/RAG.git
git push -u origin main
```

## Static Page

`docs/index.html` can be used for GitHub Pages (`main` branch + `/docs`).
