from langchain_community.document_loaders import DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
import openai
from dotenv import load_dotenv
import os
import shutil
from langchain.document_loaders import PyMuPDFLoader

# Define the folder containing the PDF files
folder_path = 'documentations/'

# Initialize a list to store all documents
all_documents = []

# Load environment variables. Assumes that project contains .env file with API keys
load_dotenv()
#---- Set OpenAI API key 
# Change environment variable name from "OPENAI_API_KEY" to the name given in 
# your .env file.
openai.api_key = os.environ['OPENAI_API_KEY']

CHROMA_PATH = "chroma"
DATA_PATH = "documentations/"

# def main():
#     generate_data_store()

def generate_data_store():
    documents = load_documents()
    chunks = split_text(documents)
    save_to_chroma(chunks)

def load_documents():
    # Loop through all PDF files in the folder
    for file_name in os.listdir(folder_path):
        if file_name.endswith('.pdf'):  # Check if the file is a PDF
            file_path = os.path.join(folder_path, file_name)
            loader = PyMuPDFLoader(file_path)  # Load the PDF
            documents = loader.load()  # Extract documents
            all_documents.extend(documents)  # Append to the list of all documents
    return all_documents
    

def split_text(documents: list[Document]):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=300,
        chunk_overlap=100,
        length_function=len,
        add_start_index=True,
    )
    chunks = text_splitter.split_documents(documents)
    print(f"Split {len(documents)} documents into {len(chunks)} chunks.")

    document = chunks[10]
    print(document.page_content)
    print(document.metadata)

    return chunks

def save_to_chroma(chunks: list[Document]):
    # Clear out the database first.
    if os.path.exists(CHROMA_PATH):
        shutil.rmtree(CHROMA_PATH)

generate_data_store()

#     # Create a new DB from the documents.
#     db = Chroma.from_documents(
#         chunks, OpenAIEmbeddings(), persist_directory=CHROMA_PATH
#     )
#     db.persist()
#     print(f"Saved {len(chunks)} chunks to {CHROMA_PATH}.")


# if __name__ == "__main__":
#     main()