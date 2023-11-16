import os.path
import tempfile
import streamlit as st
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA


def config_retriever(uploaded_file1, openai_api_key1):
    # print("config retriever")
    # Load document if file is uploaded
    if uploaded_file1 is not None:
        # Load PDF Document
        tmp_dir = tempfile.TemporaryDirectory()
        tmp_filepath = os.path.join(tmp_dir.name, uploaded_file1.name)
        with open(tmp_filepath, "wb") as f:
            f.write(uploaded_file1.getvalue())
        loader = PyPDFLoader(tmp_filepath)
        # Split Documents
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=5000, chunk_overlap=0)
        splits = text_splitter.split_documents(loader.load())
        # Embed, Store splits and return retriever
        vectorstore = Chroma.from_documents(documents=splits,
                                            embedding=OpenAIEmbeddings(openai_api_key=openai_api_key1))
        retriever1 = vectorstore.as_retriever()
        return retriever1


def generate_response(retriever1, openai_api_key1, query_text1):
    # print("generate response")
    # LLM
    llm_name = "gpt-3.5-turbo"
    llm = ChatOpenAI(openai_api_key=openai_api_key1, model_name=llm_name, temperature=0)
    # Retrieval QA Chain
    qa_chain = RetrievalQA.from_chain_type(llm, retriever=retriever1)
    result = qa_chain({"query": query_text1})
    st.info(result["result"])


# Page title
st.set_page_config(page_title='GT RAG Q&A Demo')
st.title('GT RAG Q&A Demo')

# OpenAI API Key
openai_api_key = st.text_input('OpenAI API Key', type='password')

# File upload
uploaded_file = st.file_uploader('Choose your .pdf file', type='pdf')

# Query text
query_text = st.text_input('Enter your question:',
                           placeholder='Please provide a short summary.',
                           disabled=not uploaded_file)

# Form input and query
with st.form('Q&AForm'):
    # print("form")
    submitted = st.form_submit_button('Submit', disabled=not(uploaded_file and query_text))
    if submitted and openai_api_key.startswith('sk-'):
        retriever = config_retriever(uploaded_file, openai_api_key)
        generate_response(retriever, openai_api_key, query_text)
