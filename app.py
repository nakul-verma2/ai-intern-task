#This is the main file to run the streamlit app for RAG chatbot
#run this file after running ingest.py if the chroma db is not setup yet

import streamlit as st
import os
from agent import rag_app

st.set_page_config(page_title="Agentic AI Explorer", layout="wide")

st.title("🤖 Agentic AI RAG Chatbot")
st.markdown("Ask questions based on the *Agentic AI eBook*.")


if not os.path.exists("./chroma_db"):
    st.warning("⚠️ ChromaDB not found! Please run 'python ingest.py' first to process the PDF.")
    st.stop()


if "messages" not in st.session_state:
    st.session_state.messages = []


for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])


if prompt := st.chat_input("What is task decomposition?"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    try:
        with st.spinner("Searching the eBook..."):

            result = rag_app.invoke({"question": prompt})
        
        
        answer = result.get("answer", "No answer found.")
        chunks = result.get("context_chunks", [])
        score = result.get("confidence_score", 0.0)

        with st.chat_message("assistant"):
            st.markdown(answer)
            st.info(f"**Confidence Score:** {score}")
            
            
            with st.expander("🔍 View Retrieved Context Chunks"):
                if chunks:
                    for i, chunk in enumerate(chunks):
                        st.write(f"**Source Chunk {i+1}:**")
                        st.write(chunk)
                        st.divider()
                else:
                    st.write("No relevant segments found in the PDF.")

        st.session_state.messages.append({"role": "assistant", "content": answer})

    except Exception as e:
        st.error(f"Agent Error: {str(e)}")