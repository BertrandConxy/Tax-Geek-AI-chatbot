from langchain import hub
from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain.chains import RetrievalQA

from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain

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

    retrieval_qa_chat_prompt = hub.pull("langchain-ai/retrieval-qa-chat")

    combine_docs_chain = create_stuff_documents_chain(llm, retrieval_qa_chat_prompt)
    rag_chain = create_retrieval_chain(retriever, combine_docs_chain)

    return rag_chain


if __name__ == "__main__":
    st.title("Tax Geek Chatbot")

    rag_chain = setup_qa_system()
    question = st.text_input("Ask anything related to tax law....")
    if question:
        response = rag_chain.invoke({"input": question})
        st.write(response['answer'])

    with st.expander("References"):
        for i, doc in enumerate(response['context']):
            st.write(doc.page_content)
            st.write("------------------------------------")