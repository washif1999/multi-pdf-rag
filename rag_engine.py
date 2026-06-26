from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_ollama import OllamaEmbeddings, OllamaLLM
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
from langchain_classic.chains import create_history_aware_retriever
import tempfile, os

EMBED_MODEL = "nomic-embed-text"
LLM_MODEL = "llama3.2:3b"
CHROMA_DIR = "./chroma_db"

def add_pdfs_to_vectorstore(uploaded_files):
    """Load multiple PDFs and store in persistent ChromaDB"""
    embeddings = OllamaEmbeddings(model=EMBED_MODEL)
    all_chunks = []

    for uploaded_file in uploaded_files:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(uploaded_file.read())
            tmp_path = tmp.name

        loader = PyPDFLoader(tmp_path)
        docs = loader.load()

        # Add filename as metadata
        for doc in docs:
            doc.metadata["source_file"] = uploaded_file.name

        splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        chunks = splitter.split_documents(docs)
        all_chunks.extend(chunks)
        os.unlink(tmp_path)

    # Persist to disk so embeddings survive between sessions
    vectorstore = Chroma.from_documents(
        all_chunks,
        embedding=embeddings,
        persist_directory=CHROMA_DIR
    )
    return vectorstore

def build_qa_chain(vectorstore):
    """Build history-aware RAG chain"""
    llm = OllamaLLM(model=LLM_MODEL)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

    # Prompt to rewrite question based on chat history
    contextualize_prompt = ChatPromptTemplate.from_messages([
        ("system", "Given the chat history and latest question, "
                   "rewrite it as a standalone question. "
                   "Do NOT answer it, just rewrite it."),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ])

    # History-aware retriever
    history_aware_retriever = create_history_aware_retriever(
        llm, retriever, contextualize_prompt
    )

    # Answer prompt
    answer_prompt = ChatPromptTemplate.from_messages([
        ("system", "Answer the question using the context below. "
                   "If you don't know, say you don't know. "
                   "Keep answers concise.\n\nContext: {context}"),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ])

    question_answer_chain = create_stuff_documents_chain(llm, answer_prompt)
    qa_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)
    return qa_chain