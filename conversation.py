import streamlit as st
import time

def manage_conversation():
    chat_placeholder = st.empty()

    for message in st.session_state.custom_chat_history:
        with st.chat_message(message["role"]):
            st.markdown(message["message"])

    if 'custom_qa_chain' in st.session_state:
        if user_text := st.chat_input("You:", key="user_input_custom"):
            user_message = {"role": "user", "message": user_text}
            st.session_state.custom_chat_history.append(user_message)
            chat_placeholder.empty() 
            for message in st.session_state.custom_chat_history:
                with st.chat_message(message["role"]):
                    st.markdown(message["message"])
            
            with st.chat_message("assistant"):
                with st.spinner("Assistant is generating a response..."):
                    response_text = st.session_state.custom_qa_chain(user_text)
                message_placeholder = st.empty()
                full_response_text = ""
                for part in response_text['result'].split():
                    full_response_text += part + " "
                    time.sleep(0.1)
                    message_placeholder.markdown(full_response_text + "▌")
                message_placeholder.markdown(full_response_text)

            chatbot_message = {"role": "assistant", "message": response_text['result']}
            st.session_state.custom_chat_history.append(chatbot_message)
    else:
        st.write("Please upload a PDF file.")