import os
import sys

# Ensure the project root directory is in the Python module search path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import streamlit as st
from PIL import Image
from src.agent.saree_agent import SareeAgent

# Page Configuration
st.set_page_config(
    page_title="TailorTalk - Saree Visual Search",
    page_icon="🥻",
    layout="wide"
)

st.title("TailorTalk - Saree Visual Search")
st.write("Upload a saree image or provide an image URL to find visually similar items from your catalogue.")

# Initialize Saree Agent in Session State
if "agent" not in st.session_state:
    try:
        st.session_state.agent = SareeAgent()
    except Exception as e:
        st.error(f"Failed to initialize search agent: {e}")

# Sidebar - Image Input Setup
st.sidebar.header("Visual Search Input")
input_option = st.sidebar.radio("Choose Input Method:", ("File Upload", "Image URL"))

query_image = None
image_source_path = None

if input_option == "File Upload":
    uploaded_file = st.sidebar.file_uploader("Upload Saree Image", type=["jpg", "jpeg", "png", "webp"])
    if uploaded_file is not None:
        query_image = Image.open(uploaded_file)
        # Save temp copy for embedder pipeline
        image_source_path = "temp_query.png"
        query_image.save(image_source_path)
        st.sidebar.image(query_image, caption="Query Image", use_container_width=True)

elif input_option == "Image URL":
    url_input = st.sidebar.text_input("Enter Saree Image URL:")
    if url_input:
        image_source_path = url_input.strip()
        try:
            import requests
            import io
            response = requests.get(image_source_path, timeout=5)
            query_image = Image.open(io.BytesIO(response.content))
            st.sidebar.image(query_image, caption="Query Image", use_container_width=True)
        except Exception as e:
            st.sidebar.error(f"Could not load image from URL: {e}")

# Chat Interface
if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

user_prompt = st.chat_input("Ask a question or request recommendations...")

if user_prompt:
    st.session_state.messages.append({"role": "user", "content": user_prompt})
    with st.chat_message("user"):
        st.markdown(user_prompt)

    with st.chat_message("assistant"):
        with st.spinner("Searching catalogue for visual matches..."):
            if "agent" in st.session_state and st.session_state.agent:
                # Include image context if available
                full_input = user_prompt
                if image_source_path:
                    full_input += f"\n[Query Image Source: {image_source_path}]"
                
                try:
                    response_text = st.session_state.agent.run(full_input)
                except Exception as err:
                    response_text = f"Error executing search: {err}"
            else:
                response_text = "Search agent is not initialized. Check your environment variables."

            st.markdown(response_text)
            st.session_state.messages.append({"role": "assistant", "content": response_text})