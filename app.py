import streamlit as st
from langchain_core.messages import HumanMessage, AIMessage
from rag_engine import add_pdfs_to_vectorstore, build_qa_chain

st.set_page_config(page_title="Multi-PDF Q&A", page_icon="📚")
st.title("📚 Multi-PDF Q&A with Memory")
st.caption("Powered by LangChain + ChromaDB + Ollama — fully local")

# Sidebar — PDF upload
with st.sidebar:
    st.header("📄 Upload PDFs")
    uploaded_files = st.file_uploader(
        "Upload one or more PDFs",
        type="pdf",
        accept_multiple_files=True
    )

    if uploaded_files:
        if st.button("🔄 Process PDFs"):
            with st.spinner("Processing PDFs..."):
                vectorstore = add_pdfs_to_vectorstore(uploaded_files)
                st.session_state.qa_chain = build_qa_chain(vectorstore)
                st.session_state.chat_history = []
                st.session_state.file_names = [f.name for f in uploaded_files]
            st.success(f"✅ {len(uploaded_files)} PDF(s) loaded!")

    # Show loaded files
    if "file_names" in st.session_state:
        st.markdown("### 📁 Loaded Files")
        for name in st.session_state.file_names:
            st.markdown(f"- {name}")

    # Clear button
    if st.button("🗑️ Clear & Start Over"):
        st.session_state.clear()
        st.rerun()

# Main chat area
if "qa_chain" not in st.session_state:
    st.info("👈 Upload PDFs from the sidebar and click Process to begin")
else:
    # Display chat history
    for msg in st.session_state.get("chat_history", []):
        if isinstance(msg, HumanMessage):
            with st.chat_message("user"):
                st.write(msg.content)
        else:
            with st.chat_message("assistant"):
                st.write(msg.content)

    # Chat input
    if question := st.chat_input("Ask anything about your PDFs..."):
        with st.chat_message("user"):
            st.write(question)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                result = st.session_state.qa_chain.invoke({
                    "input": question,
                    "chat_history": st.session_state.chat_history
                })
                answer = result["answer"]
                sources = result["context"]

            st.write(answer)

            # Show source chunks
            with st.expander("📚 Sources used"):
                for i, doc in enumerate(sources):
                    fname = doc.metadata.get("source_file", "unknown")
                    page = doc.metadata.get("page", "?")
                    st.markdown(f"**[{fname} — page {page}]**")
                    st.caption(doc.page_content[:300] + "...")

        # Update chat history
        st.session_state.chat_history.extend([
            HumanMessage(content=question),
            AIMessage(content=answer)
        ])