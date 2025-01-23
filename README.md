# Tax Geek - AI Chatbot

This project is a LangChain-powered chatbot app designed to allow users to ask questions related to tax law. It leverages OpenAI embeddings to process and retrieve contextually relevant answers from a collection of tax law documents using a Retrieval-Augmented Generation (RAG) system. Additionally, LangSmith is used to monitor and debug every user interaction, and the system includes pytest tests for ensuring reliable performance.

## Features
- **RAG System:** Combines retrieval from document embeddings and OpenAI's LLM to provide accurate and context-specific answers.
- **Vector Embeddings:** Automatically processes and stores PDF files into embeddings using FAISS and OpenAI.
- **Interactive UI:** User-friendly interface built with Streamlit for seamless interaction.
- **LangSmith Monitoring:** Tracks and analyzes all interactions to improve the chatbot's performance and reliability.
- **Test Coverage:** Pytest tests ensure the RAG system functions as expected and retrieves accurate results.

## Prerequisites
Before running the application, ensure you have the following:

1. Python 3.8 or later
2. Required Python libraries:
   - `streamlit`
   - `faiss-cpu`
   - `pypdf`
   - `langchain`
   - `langchain-openai`
   - `pytest`
   - `langsmith`



## Folder Structure
- **`./documentations`**: Place your tax-related PDF documents here.
- **`app.py`**: Main application file.
- **`create_vectordb.py`**: Creates vectordb with document embeddings and save the vectordb locally.
- **`test_rag.py`**: Contains pytest tests for the RAG system.

## How to Run

1. Clone the repository:

   ```bash
   git clone https://github.com/BertrandConxy/Tax-Geek-AI-chatbot.git
   cd Tax-Geek-AI-chatbot
   ```
2. Create virtual env
   ```bash
   python -m venv venv
   ```

3. Install the dependencies using:

   ```bash
   pip install -r requirements.txt
   ```
4. Ensure tax-related PDF documents are in the `documentations` folder.
5. Create `.env` file for credentials
   ```
   OPENAI_API_KEY
   LANGSMITH_TRACING=True
   LANGSMITH_ENDPOINT
   LANGSMITH_API_KEY
   LANGSMITH_PROJECT
   ```
6. Run `create_vectordb.py` to create vector embeddings for the documents and store the db locally.
   ```
   python create_vectordb.py
   ```
7. Run the Streamlit app:

   ```bash
   streamlit run app.py
   ```
## Demo
![Demo](demo_1.gif)
![Demo](demo_2.png)
## Monitoring with LangSmith
LangSmith is integrated into this project to monitor and analyze chatbot interactions. This ensures the app remains robust and user-friendly. To configure LangSmith:

1. Set up your LangSmith account and API key.
2. Ensure the `LANGSMITH_API_KEY` is added to your environment variables.

## Testing the RAG System
Pytest tests are included to validate the functionality of the RAG system. To run the tests:

```bash
pytest tests_rag.py
```