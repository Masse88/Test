# chat_utils.py
import streamlit as st
def initialize_chat_history():
   if "messages" not in st.session_state:
       st.session_state.messages = [
           {
               "role": "assistant",
               "content": "Hello! Ask me anything about International Football from 1872 to (the almost) present day!",
           }
       ]
def display_chat_history():
   for message in st.session_state.messages:
       with st.chat_message(message["role"]):
           st.markdown(message["content"])

# chat_utils.py
def handle_user_input(openai_api_key, query_graph_func, chain):
   if prompt := st.chat_input("Your question..."):
       st.session_state.messages.append({"role": "user", "content": prompt})
       with st.chat_message("user"):
           st.markdown(prompt)
       if openai_api_key:
           with st.spinner("Thinking..."):
               try:
                   response = query_graph_func(chain=chain, query=prompt)
               except Exception as e:
                   st.error(f"An error occurred: {str(e)}")
                   response = "I'm sorry, I encountered an error while processing your request."
           with st.chat_message("assistant"):
               st.markdown(response)
           st.session_state.messages.append({"role": "assistant", "content": response})
       else:
           st.error(
               "Please enter your OpenAI API key in the sidebar to use the chatbot."
           )
