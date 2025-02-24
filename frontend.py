import streamlit as st
from backend import comp_process

def frontend():
    st.set_page_config(page_title="Chat with multiple PDFs", layout="centered")
    st.title("Chat with Multiple PDFs")
    question = st.text_input("Ask Question Below: ")
    
    with st.sidebar:
        st.subheader("Enter your OpenAI API: ")
        api_key = st.text_input("Enter API Key", type="password", placeholder="Enter OpenAI Key")
        pdfs = st.file_uploader("Upload PDF files", type="pdf", accept_multiple_files=True)
        st.button("Process")
    
    # if pdfs and api_key is not None:
    if question:
        ans = comp_process(pdfs=pdfs, question=question)
        st.text(ans)
        
if __name__ == "__main__":
    frontend()