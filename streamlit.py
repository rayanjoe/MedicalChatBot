import streamlit as st
from run_model_support import rag_pipeline
import time

st.title("Medical Chat BOT")
st.write("Query your hurdles which you're facing.")

if "chat_histories" not in st.session_state:
    st.session_state.chat_histories = {}

domain = "Retail Business"

if domain not in st.session_state.chat_histories:
    st.session_state.chat_histories[domain] = []

for chat in st.session_state.chat_histories[domain]:
    with st.chat_message("user"):
        st.markdown(f"**User Query:** {chat['user_query']}")
    with st.chat_message("assistant"):
        st.markdown(f"**Response:** {chat['response']}")
        st.markdown(f"**Execution Time:** {chat['execution_time']:.4f} seconds")

if user_query := st.chat_input("Enter your SQL query or question:"):
    with st.chat_message("user"):
        st.markdown(user_query)
    
    start_time = time.time()
    response = rag_pipeline(user_query) 
    end_time = time.time()
    execution_time = end_time - start_time

    with st.chat_message("assistant"):
        st.markdown(f"**Response:** {response}")
        st.markdown(f"**Execution Time:** {execution_time:.4f} seconds")

    st.session_state.chat_histories[domain].append({
        "domain": domain,
        "user_query": user_query,
        "response": response,  
        "execution_time": execution_time
    })