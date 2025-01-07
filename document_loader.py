import os
from langchain.document_loaders import PyMuPDFLoader

# Define the folder containing the PDF files
folder_path = 'documentations/'

# Initialize a list to store all documents
all_documents = []

# Loop through all PDF files in the folder
for file_name in os.listdir(folder_path):
    if file_name.endswith('.pdf'):  # Check if the file is a PDF
        file_path = os.path.join(folder_path, file_name)
        loader = PyMuPDFLoader(file_path)  # Load the PDF
        documents = loader.load()  # Extract documents
        all_documents.extend(documents)  # Append to the list of all documents

# `all_documents` now contains all pages from all PDFs in the folder
print(f"Loaded {len(all_documents)} pages from {len(os.listdir(folder_path))} PDFs.")