from pprint import pprint
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain.embeddings import GPT4AllEmbeddings
from langchain.llms import GPT4All

# Load Documents
loader = PyPDFLoader("docs/lec1.pdf")

# Split Documents
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
splits = text_splitter.split_documents(loader.load())

# Embed and store splits. Download and use GPT4All embeddings locally.
vectorstore = Chroma.from_documents(documents=splits, embedding=GPT4AllEmbeddings())

# LLM
# llm = GPT4All(model="/Users/awarke/models/GPT4All/gpt4all-13b-snoozy-q4_0.gguf", max_tokens=2048)
llm = GPT4All(model="/Users/awarke/models/GPT4All/gpt4all-13b-snoozy-q4_0.gguf")

# DEBUG
'''
question = "what are different flavors of project?"
docs = vectorstore.similarity_search(question, k=10)
pprint(docs)
'''

# Retrieval QA Chain
# question = "What is the course objective?"
# question = "different flavors of project?"
question = "what are different flavors of project?"
qa_chain = RetrievalQA.from_chain_type(llm, retriever=vectorstore.as_retriever(search_kwargs={'k': 10}))
# qa_chain = RetrievalQA.from_chain_type(llm, retriever=vectorstore.as_retriever())

result = qa_chain({"query": question})
print(result["result"])
