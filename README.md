# TailorTalk - Saree Visual Search Assistant

TailorTalk is an AI-powered visual search application designed to help users find similar Indian sarees from a catalog using image similarity and natural language queries. Built using Streamlit, Qdrant vector database, and Hugging Face Transformers.

## Features

- Visual Similarity Search: Upload a saree image or provide an image URL to find visual matches across pattern, weave, border, and color.
- Hybrid Conversational Agent: Integrated with Gemini LLM via LangChain to answer user style questions alongside product suggestions.
- Vector Search Infrastructure: Uses Fashion-CLIP embeddings stored inside Qdrant vector database for fast nearest-neighbor lookups.

## Architecture & Tech Stack

- Frontend: Streamlit
- Vector Database: Qdrant Cloud
- Embeddings: Hugging Face Transformers (`patrickjohncyh/fashion-clip`)
- Orchestration: LangChain
- LLM Integration: Google Gemini 1.5 Flash

## Repository Structure

```
tailortalk-saree-search/
├── app.py                     # Streamlit frontend entry point
├── requirements.txt           # Python dependencies
├── runtime.txt                # Streamlit Python version lock
├── src/
│   ├── agent/                 # Agent logic and tools
│   │   ├── saree_agent.py
│   │   └── tools.py
│   ├── embeddings/            # Embedding generation code
│   │   └── fashion_embedder.py
│   ├── vector_db/             # Qdrant vector store connection
│   │   └── qdrant_service.py
│   └── config.py              # Environment configuration loader
└── README.md
```

## Local Setup Instructions

1. Clone the repository:
   ```bash
   git clone [https://github.com/Vivek3089/tailortalk-saree-search.git](https://github.com/Vivek3089/tailortalk-saree-search.git)
   cd tailortalk-saree-search
   ```

2. Create a virtual environment and install dependencies:
   ```bash
   python3.11 -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt
   ```

3. Configure environment variables in a `.env` file:
   ```env
   GEMINI_API_KEY=your_gemini_api_key
   QDRANT_URL=your_qdrant_instance_url
   QDRANT_API_KEY=your_qdrant_api_key
   ```

4. Run the Streamlit app locally:
   ```bash
   streamlit run app.py
   ```