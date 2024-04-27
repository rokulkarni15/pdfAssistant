import streamlit as st
from langchain_community.llms import Ollama
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.callbacks.manager import CallbackManager
from langchain_community.embeddings.ollama import OllamaEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory

def initialize_session_state():
    # Set up a custom template for the chat interface
    if 'custom_template' not in st.session_state:
        st.session_state.custom_template = """Hello! I'm here to assist with your PDF-related queries.

        Context: {context}
        History: {history}

        User: {question}
        Assistant:"""

    # Initialize a prompt template for structured input to the language model
    if 'custom_prompt' not in st.session_state:
        st.session_state.custom_prompt = PromptTemplate(
            input_variables=["history", "context", "question"],
            template=st.session_state.custom_template,
        )

    # Set up a memory buffer to store conversation history
    if 'custom_memory' not in st.session_state:
        st.session_state.custom_memory = ConversationBufferMemory(
            memory_key="history",
            return_messages=True,
            input_key="question"
        )

    # Initialize a vector store for document embeddings, with persistence enabled
    if 'custom_vectorstore' not in st.session_state:
        st.session_state.custom_vectorstore = Chroma(persist_directory='chroma_db',
                                                     embedding_function=OllamaEmbeddings(base_url='http://localhost:11434',
                                                                                         model="mistral")
                                                     )

    # Set up the LLM with callback handlers for streaming output
    if 'custom_llm' not in st.session_state:
        st.session_state.custom_llm = Ollama(base_url="http://localhost:11434",
                                             model="mistral",
                                             verbose=True,
                                             callback_manager=CallbackManager(
                                                 [StreamingStdOutCallbackHandler()])
                                             )

    # Initialize an empty list to store chat history for the session
    if 'custom_chat_history' not in st.session_state:
        st.session_state.custom_chat_history = []