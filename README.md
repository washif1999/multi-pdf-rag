# 📚 Multi-PDF RAG Q&A with Conversation Memory

A fully local Retrieval-Augmented Generation (RAG) app that lets you upload
multiple PDF documents and chat with them — with memory of previous questions.
No API keys, no cloud, no cost.

## 🎯 Features

- 📄 Upload multiple PDFs simultaneously
- 🧠 Conversation memory — remembers previous questions in the session
- 🔍 Source tracking — shows which PDF and page each answer came from
- 💾 Persistent embeddings — ChromaDB saves to disk between sessions
- 🏠 Fully local — runs entirely on your machine via Ollama

## 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| LLM | Ollama (llama3.2:3b) — local |
| Embeddings | Ollama (nomic-embed-text) — local |
| RAG Framework | LangChain |
| Vector Store | ChromaDB (persistent) |
| PDF Parsing | PyPDF |
| UI | Streamlit |

## 🏗️ Architecture

```
PDF Upload → PyPDF Loader → Text Chunking (500 tokens)
    → nomic-embed-text Embeddings → ChromaDB Vector Store
    → History-Aware Retriever → llama3.2:3b LLM → Answer
```

## ⚙️ How It Works

1. PDFs are loaded and split into 500-token chunks with 50-token overlap
2. Each chunk is embedded using `nomic-embed-text` via Ollama
3. Embeddings are stored persistently in ChromaDB on disk
4. User question + chat history is passed to a history-aware retriever
5. Top 3 relevant chunks are retrieved via similarity search
6. `llama3.2:3b` generates an answer using retrieved context
7. Chat history is maintained for follow-up questions

## 🚀 Run Locally

### Prerequisites
- Python 3.10+
- [Ollama](https://ollama.com/download) installed

### Setup

```bash
# Clone the repo
git clone https://github.com/washif1999/multi-pdf-rag.git
cd multi-pdf-rag

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Pull Ollama models (one-time)
ollama pull llama3.2:3b
ollama pull nomic-embed-text
```

### Run

```bash
# Terminal 1 — start Ollama
ollama serve

# Terminal 2 — start app
streamlit run app.py
```

Open **http://localhost:8501** in your browser.

## 📖 Usage

1. Upload one or more PDFs from the sidebar
2. Click **"🔄 Process PDFs"** to embed and index them
3. Ask questions in the chat input
4. Expand **"📚 Sources used"** to see which chunks were retrieved
5. Ask follow-up questions — the app remembers context

## 📁 Project Structure

```
multi-pdf-rag/
├── app.py            # Streamlit UI
├── rag_engine.py     # RAG pipeline (LangChain + ChromaDB)
├── requirements.txt
├── README.md
└── chroma_db/        # Persistent vector store (auto-created)
```

## 🔑 Key Concepts Demonstrated

- **RAG Pipeline** — retrieval-augmented generation end-to-end
- **Chunking Strategy** — 500 token chunks with 50 token overlap
- **Vector Similarity Search** — ChromaDB with cosine similarity
- **History-Aware Retrieval** — follow-up questions handled via chat history
- **Persistent Vector Store** — embeddings saved to disk via persist_directory
- **Metadata Filtering** — source file and page number tracked per chunk
- **Local LLM** — zero API cost using Ollama on-device inference

## 🔮 Future Improvements

- [ ] Add metadata filtering (search within specific PDF only)
- [ ] Add FastAPI backend + React frontend
- [ ] Add streaming responses
- [ ] Add document summarization feature
- [ ] Deploy with Docker

## 👤 Author

**Muhd Washif** — [github.com/washif1999](https://github.com/washif1999) | [LinkedIn](https://linkedin.com/in/muhammadwashif-b64b97)