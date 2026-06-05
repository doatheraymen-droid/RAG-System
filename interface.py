import streamlit as st
import requests
import time

st.set_page_config(page_title="RAG System", layout="wide")

st.title("📚 RAG System")
st.markdown("Upload documents and ask questions")

# API configuration
API_URL = "http://localhost:8000"

# Sidebar for file upload
with st.sidebar:
    st.header("📁 Upload Documents")
    
    uploaded_file = st.file_uploader("Choose a file", type=['txt', 'pdf', 'docx'])
    
    if uploaded_file is not None:
        files = {"file": (uploaded_file.name, uploaded_file.getvalue())}
        response = requests.post(f"{API_URL}/upload", files=files)
        if response.status_code == 200:
            st.success(f"✅ {uploaded_file.name} uploaded and processed!")
        else:
            st.error(f"Error: {response.text}")
    
    st.divider()
    
    st.header("📄 Uploaded Files")
    files_response = requests.get(f"{API_URL}/files")
    if files_response.status_code == 200:
        files = files_response.json().get("files", [])
        for f in files:
            st.text(f"• {f}")
    else:
        st.text("No files yet")

# Main area for Q&A
st.header("💬 Ask Questions")

query = st.text_input("Your question:")

col1, col2 = st.columns([1, 4])
with col1:
    top_k = st.number_input("Top K chunks", min_value=1, max_value=20, value=5)

if st.button("Ask", type="primary"):
    if not query:
        st.warning("Please enter a question")
    else:
        with st.spinner("Thinking..."):
            response = requests.post(
                f"{API_URL}/query",
                json={"question": query, "top_k": top_k}
            )
            
            if response.status_code == 200:
                result = response.json()
                
                # Display answer
                st.markdown("### Answer")
                st.info(result["answer"])
                
                # Display retrieved chunks
                with st.expander("📖 Retrieved Document Chunks", expanded=False):
                    for i, chunk in enumerate(result["retrieved_chunks"]):
                        st.markdown(f"**Chunk {i+1}** (Score: {chunk['score']:.3f}) - Source: {chunk['source']}")
                        st.caption(chunk["text"])
                        st.divider()
            else:
                st.error(f"Error: {response.text}")

# Instructions
st.divider()
with st.expander("ℹ️ How to use", expanded=False):
    st.markdown("""
    1. **Start the backend** first: `python main.py`
    2. **Run this interface**: `streamlit run interface.py`
    3. **Upload documents** (PDF, DOCX, TXT)
    4. **Ask questions** - the system will find relevant info and answer
    
    Supported documents: PDF, Word (.docx), Text files
    """)