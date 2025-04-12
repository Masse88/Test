import streamlit as st
from graph_utils import init_resources, query_graph
from chat_utils import initialize_chat_history, display_chat_history, handle_user_input

def validate_api_key(api_key: str) -> bool:
    """Validate the OpenAI API key format."""
    return bool(api_key and api_key.startswith('sk-') and len(api_key) > 40)

def init_app_resources(api_key: str):
    """Initialize application resources with error handling."""
    try:
        with st.spinner("Initializing resources..."):
            graph, chain = init_resources(api_key)
            return graph, chain
    except Exception as e:
        st.error(f"Failed to initialize resources: {str(e)}")
        return None, None

# Initialize session state
if 'initialized' not in st.session_state:
    st.session_state.initialized = False
if 'resources' not in st.session_state:
    st.session_state.resources = None

st.title("Football Memoirs - an AI for Hardcore Football Fans")

# Sidebar for API key input
with st.sidebar:
    openai_api_key = st.text_input("Enter your OpenAI API Key", type="password")
    if not openai_api_key:
        st.warning("Please enter your OpenAI API key to use the chatbot.")
    elif not validate_api_key(openai_api_key):
        st.error("Invalid API key format. Please check your API key.")
    else:
        if not st.session_state.initialized:
            graph, chain = init_app_resources(openai_api_key)
            if graph and chain:
                st.session_state.resources = (graph, chain)
                st.session_state.initialized = True
                st.success("Resources initialized successfully!", icon="🚀")

# Main application logic
if st.session_state.initialized and st.session_state.resources:
    graph, chain = st.session_state.resources
    
    # Initialize and display chat history
    initialize_chat_history()
    display_chat_history()
    
    # Handle user input
    handle_user_input(
        openai_api_key=openai_api_key,
        query_graph_func=query_graph,
        chain=chain
    )
