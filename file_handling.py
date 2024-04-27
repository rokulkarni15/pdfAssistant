import streamlit as st
import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings.ollama import OllamaEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA

def process_uploaded_file(uploaded_file):
    # Define the path for saving the uploaded PDF
    file_path = f"custom_files/{uploaded_file.name}.pdf"
    # Check if the file already exists to avoid reprocessing
    if not os.path.isfile(file_path):
        with st.status("Analyzing your PDF..."):
            # Save the uploaded file to disk
            save_uploaded_file(uploaded_file, file_path)
            # Load data from the PDF file
            data = load_pdf_data(file_path)
            # Process the loaded PDF data
            process_pdf_data(data)

def save_uploaded_file(uploaded_file, file_path):
    # Read bytes from the uploaded file and write them to a new file
    bytes_data = uploaded_file.read()
    with open(file_path, "wb") as f:
        f.write(bytes_data)

def load_pdf_data(file_path):
    # Load PDF content using a PDF loader
    loader = PyPDFLoader(file_path)
    return loader.load()

def process_pdf_data(data):
    # Split the PDF text into manageable chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=200, length_function=len)
    all_splits = text_splitter.split_documents(data)
    # Create a vector store from the document splits and persist it
    st.session_state.custom_vectorstore = Chroma.from_documents(documents=all_splits, embedding=OllamaEmbeddings(model="mistral"))
    st.session_state.custom_vectorstore.persist()
    # Initialize the QA chain for answering questions
    initialize_retrieval_qa()

def initialize_retrieval_qa():
    # Convert the vector store into a retriever for the QA chain
    st.session_state.custom_retriever = st.session_state.custom_vectorstore.as_retriever()
    # Initialize the QA chain if it has not been set up yet
    if 'custom_qa_chain' not in st.session_state:
        st.session_state.custom_qa_chain = RetrievalQA.from_chain_type(
            llm=st.session_state.custom_llm,
            chain_type='stuff',
            retriever=st.session_state.custom_retriever,
            verbose=True,
            chain_type_kwargs={
                "verbose": True,
                "prompt": st.session_state.custom_prompt,
                "memory": st.session_state.custom_memory,
            }
        )