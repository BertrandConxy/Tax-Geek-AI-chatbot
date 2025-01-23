from langchain import hub
from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_openai.embeddings import OpenAIEmbeddings

from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain

import streamlit as st

from dotenv import load_dotenv

load_dotenv()

llm = ChatOpenAI(model="gpt-4o-mini")

retrieval_qa_chat_prompt = hub.pull("langchain-ai/retrieval-qa-chat")

def prompt_rag(question):
    # Load vectorstore from disk
    loaded_vectorstore = FAISS.load_local("vectordb", embeddings=OpenAIEmbeddings(), allow_dangerous_deserialization=True)
    retriever = loaded_vectorstore.as_retriever()
    combine_docs_chain = create_stuff_documents_chain(llm, retrieval_qa_chat_prompt)
    rag_chain = create_retrieval_chain(retriever, combine_docs_chain)
    response = rag_chain.invoke({"input": question})
    return response

def main():
    st.title("Tax Geek Chatbot")

    question = st.text_input("Ask me anything related to tax law...")

    if question:
        response = prompt_rag(question)
        st.write(response['answer'])

        with st.expander("References"):
            for i, doc in enumerate(response['context']):
                st.write(doc.page_content)
                st.write("------------------------------------")

if __name__ == "__main__":
    main()