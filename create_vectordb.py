import shutil
from langchain_openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_community.vectorstores import FAISS
from langchain_openai.embeddings import OpenAIEmbeddings

import os

from dotenv import load_dotenv

load_dotenv()

def main():
    create_vectordb()

def create_vectordb():
    documents = load_documents()
    chunks = split_text(documents)
    save_to_faiss(chunks)

def load_documents():
    loader = PyPDFDirectoryLoader('documentations/')
    documents = loader.load_and_split()
    return documents


def split_text(documents):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    chunks = text_splitter.split_documents(documents)
    return chunks


def save_to_faiss(chunks):
    # Clear out the database first.
    if os.path.exists('vectordb'):
        shutil.rmtree('vectordb')

    # Create a new vector store from the documents.
    vector_store = FAISS.from_documents(
        chunks, embedding=OpenAIEmbeddings()
    )
    vector_store.save_local("vectordb")
    print(f"Saved {len(chunks)} chunks to vectordb.")

    return vector_store


if __name__ == "__main__":
    main()