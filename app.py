import streamlit as st
import os
from conversation import manage_conversation
from file_handling import process_uploaded_file
from setup import initialize_session_state

# Ensure necessary directories exist 
os.makedirs('custom_files', exist_ok=True)
os.makedirs('chroma_db', exist_ok=True)

# Initialize session state variables and configurations
initialize_session_state()

# Set the title of the Streamlit application
st.title("PDF Assistant")

# Interface for uploading PDF files
uploaded_pdf = st.file_uploader("Upload your PDF", type='pdf')
if uploaded_pdf:
    # Process the uploaded PDF file
    process_uploaded_file(uploaded_pdf)

# Handle conversation logic and display
manage_conversation()