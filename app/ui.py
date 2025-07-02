import streamlit as st
from chatbot import load_chatbot

st.set_page_config(page_title="Medical LLM Chatbot (Offline Embeddings)", layout="wide")
st.title(" Medical Assistant Chatbot (HuggingFace + FAISS)")
qa_chain = load_chatbot()
query = st.text_input("what's your query? :) ")
if query:
    with st.spinner("Thinking..."):
        result = qa_chain(query)
        st.markdown(f"**Answer:** {result['result']}")
        st.markdown("Source Chunks")
        for i, doc in enumerate(result["source_documents"]):
            st.markdown(f"**Chunk {i+1}:** {doc.page_content[:300]}...")
