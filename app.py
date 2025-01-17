from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFDirectoryLoader
import streamlit as st

from dotenv import load_dotenv

load_dotenv()

st.title("Tax Geek Chatbot")

model = ChatOpenAI(model="gpt-4o-mini")

prompt = ChatPromptTemplate.from_template(
    """
    Answer the questions based on the provided context first. Provide
    most accurate responses. If you don't know the answer, you can skip
    the question. If you need more information, you can ask for it.
    <context>
    {context}
    <context>
    Question: {question}
    """
)

def vector_embedding():
    if "vector_store" not in st.session_state:
        st.session_state.embeddings = OpenAIEmbeddings()
        st.session_state.loader = PyPDFDirectoryLoader("./documentations")
        st.session_state.docs = st.session_state.loader.load()
        st.session_state.splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        st.session_state.split_docs = st.session_state.splitter.split_documents(st.session_state.docs)
        st.session_state.chroma_inst = Chroma(embedding_function=st.session_state.embeddings).add_documents(st.session_state.split_docs)

questionPrompt = st.text_input("Ask anything related to tax law....")

if st.button("Document embeddings"):
    vector_embedding()
    st.write("Document embeddings created")




if questionPrompt:
    document_chain = create_stuff_documents_chain(model, prompt)
    retriever = st.session_state.chroma_inst.as_retriever()
    retrieval_chain = create_retrieval_chain(document_chain, retriever)
    response = retrieval_chain.invoke({"question": questionPrompt})
    st.write(response["answer"])

    with st.expander("References"):
        for i, doc in enumerate(response['context']):
            st.write(doc.page_content)
            st.write("------------------------------------")