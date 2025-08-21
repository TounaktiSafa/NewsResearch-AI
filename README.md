# Finance Information LLM 📈

A powerful Streamlit application that uses Large Language Models (LLM) to analyze and extract insights from financial news articles. Process multiple URLs and ask questions about the content using advanced NLP capabilities.

## Features

- **URL Processing**: Load and process up to 3 financial news URLs simultaneously
- **Document Embedding**: Automatically splits and embeds content using HuggingFace embeddings
- **Vector Storage**: Uses FAISS for efficient similarity search and retrieval
- **LLM Integration**: Powered by Ollama with Llama2 model for intelligent question answering
- **Source Attribution**: Provides sources for each answer with document references
- **User-Friendly Interface**: Clean Streamlit UI with intuitive controls

## Tech Stack

- **Frontend**: Streamlit
- **LLM**: Ollama with Llama2 model
- **Embeddings**: HuggingFace Sentence Transformers (all-MiniLM-L6-v2)
- **Vector Store**: FAISS
- **NLP**: LangChain framework
- **Web Scraping**: BeautifulSoup, UnstructuredURLLoader

