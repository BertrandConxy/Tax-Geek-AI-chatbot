# Tax Geek - AI chatbot

This project is a Streamlit application designed to allow users to ask questions related to tax law. It leverages OpenAI embeddings to process and retrieve contextually relevant answers from a collection of tax law documents(RAG system).

## Features
- **Vector embeddings:** Automatically process PDF files into embeddings using Chroma and OpenAI.
- **Question-Answering:** Ask context-specific questions, and get relevant answers based on the document embeddings.
- **Interactive Interface:** A user-friendly interface built with Streamlit.

## Prerequisites
Before running the application, ensure you have the following:
1. Python 3.8 or later
2. Required Python libraries:
   - Streamlit
   - OpenAI
   - Chroma
   - PyPDF2
   - LangChain
  
Install the dependencies using:

```bash
pip install -r requirements.txt
```

## Folder structure
- ./documentations:  Place your PDF documents here.
- app.py: Main application file.

## How to Run

1.
```bash
git clone https://github.com/yourusername/tax-law-qa.git
cd tax-law-qa
```

2. Place the tax-related PDF documents in the documentations folder.

3. ```bash
   streamlit run app.py
   ```
