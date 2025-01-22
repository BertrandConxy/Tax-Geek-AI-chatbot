from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_community.vectorstores import FAISS
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain.chains import RetrievalQA

import streamlit as st

from dotenv import load_dotenv

load_dotenv()


model = ChatOpenAI(model="gpt-4o-mini")

def setup_qa_system():
    loader = PyPDFDirectoryLoader('documentations/')
    docs = loader.load_and_split()
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_documents(docs)

    embeddings = OpenAIEmbeddings()
    vector_store = FAISS.from_documents(chunks, embeddings)
    retriever = vector_store.as_retriever()

    llm = ChatOpenAI(model="gpt-4o-mini")
    qa_chain = RetrievalQA.from_chain_type(retriever=retriever, llm=llm)

    return qa_chain


if __name__ == "__main__":
    st.title("Tax Geek Chatbot")

    qa_chain = setup_qa_system()
    question = st.text_input("Ask anything related to tax law....")
    if question:
        response = qa_chain.invoke(question)
        st.write(response['result'])

    # with st.expander("References"):
    #     for i, doc in enumerate(response['context']):
    #         st.write(doc.page_content)
    #         st.write("------------------------------------")