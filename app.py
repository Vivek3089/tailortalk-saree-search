import os
import streamlit as st
from PIL import Image
from src.agent.saree_agent import SareeAgent
from src.utils.image_helpers import load_image_from_input

# Page configuration
st.set_page_config(
    page_title="Saree Visual Search Assistant",
    layout="wide"
)

st.title("TailorTalk - Saree Visual Search")
st.caption("Upload a saree image or provide an image URL to find visually similar items from your catalogue.")

# Initialize Saree Agent in Streamlit Session State
if "agent" not in st.session_state:
    with st.spinner("Initializing agent and loading models..."):
        st.session_state.agent = SareeAgent()

# Sidebar for Image Input
st.sidebar.header("Visual Search Input")
input_type = st.sidebar.radio("Choose Input Method:", ["File Upload", "Image URL"])

uploaded_image = None
image_ref = None

if input_type == "File Upload":
    file = st.sidebar.file_uploader("Upload Saree Image", type=["jpg", "jpeg", "png", "webp"])
    if file is not None:
        uploaded_image = Image.open(file)
        st.sidebar.image(uploaded_image, caption="Query Image", use_container_width=True)
        # Save temporary image file to pass path to the tool
        temp_path = "temp_query.png"
        uploaded_image.save(temp_path)
        image_ref = temp_path
else:
    url = st.sidebar.text_input("Enter Image URL:")
    if url.strip():
        try:
            uploaded_image = load_image_from_input(url.strip())
            st.sidebar.image(uploaded_image, caption="Query Image", use_container_width=True)
            image_ref = url.strip()
        except Exception as e:
            st.sidebar.error(f"Failed to load image from URL: {e}")

# Chat History Setup
if "messages" not in st.session_state:
    st.session_state.messages = []

# Render Chat History
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# User Chat Input
user_prompt = st.chat_input("Ask a question or request recommendations...")

if user_prompt:
    # Append user prompt to history
    st.session_state.messages.append({"role": "user", "content": user_prompt})
    with st.chat_message("user"):
        st.markdown(user_prompt)

    # Attach image context to prompt if an image is loaded
    full_query = user_prompt
    if image_ref:
        full_query += f"\n[Query Image Source: {image_ref}]"

    # Run Saree Agent
    with st.chat_message("assistant"):
        with st.spinner("Searching catalogue for matching recommendations..."):
            try:
                response = st.session_state.agent.run(full_query)
                st.markdown(response)
                st.session_state.messages.append({"role": "assistant", "content": response})
            except Exception as e:
                error_msg = f"Error executing search: {str(e)}"
                st.error(error_msg)
                st.session_state.messages.append({"role": "assistant", "content": error_msg})