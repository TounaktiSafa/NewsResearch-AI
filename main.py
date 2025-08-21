import os
import time
import streamlit as st
import pickle

from langchain.chains.qa_with_sources.retrieval import RetrievalQAWithSourcesChain
from langchain_community.llms.ollama import Ollama
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import UnstructuredURLLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.docstore.document import Document

file_path = "vector_index.pkl"
main_placeholder = st.empty()

st.title("News Research Tool")
st.sidebar.title("News Article URLs")
urls = []
for i in range(3):
    url = st.sidebar.text_input(f"URL {i + 1}")
    urls.append(url)

process_url_clicked = st.sidebar.button("Process URLs")
if process_url_clicked:
    # Filter out empty URLs
    valid_urls = [url for url in urls if url and url.strip()]

    if not valid_urls:
        st.error("Please enter at least one valid URL")
        st.stop()

    try:
        # Load data
        main_placeholder.text("Data Loading ...")
        loader = UnstructuredURLLoader(urls=valid_urls)
        data = loader.load()

        # Split Data
        text_splitter = RecursiveCharacterTextSplitter(
            separators=['\n\n', '\n', '.', ','],
            chunk_size=1000,
            chunk_overlap=200
        )

        main_placeholder.text("Data Splitting ...")
        docs = text_splitter.split_documents(data)

        # Create embeddings
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        main_placeholder.text("Building Embeddings and Vector Store...")

        # Create vector store with progress indication
        vector_store = FAISS.from_documents(docs, embeddings)

        # Save vector store
        with open(file_path, "wb") as f:
            pickle.dump(vector_store, f)

        main_placeholder.text("✅ Processing complete! Vector store saved.")
        st.success(f"Successfully processed {len(valid_urls)} URL(s)!")

    except Exception as e:
        main_placeholder.text("❌ Error occurred!")
        st.error(f"Error processing URLs: {str(e)}")

# Query section - CORRECTED
query = st.text_input("Question: ")
if query:
    if os.path.exists(file_path):
        # Load the vector store
        with open(file_path, "rb") as f:
            vector_store = pickle.load(f)

        # Initialize LLM
        llm = Ollama(model="llama2")

        # Create chain
        chain = RetrievalQAWithSourcesChain.from_llm(
            llm=llm,
            retriever=vector_store.as_retriever()
        )

        # Get result
        result = chain.invoke({"question": query})  # Changed from "query" to "question"

        # Display results
        st.header("Answer")
        st.write(result['answer'])

        sources = result.get("sources", "")
        if sources:
            st.subheader("Sources:")
            sources_list = sources.split("\n")
            for source in sources_list:
                st.write(source)
    else:
        st.error("Vector store not found. Please process URLs first.")