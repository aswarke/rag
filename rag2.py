from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA

# Load Documents
loader = PyPDFLoader("docs/lec1.pdf")

# Split Documents
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=0)
splits = text_splitter.split_documents(loader.load())

# Embed and store splits
vectorstore = Chroma.from_documents(documents=splits, embedding=OpenAIEmbeddings())
retriever = vectorstore.as_retriever()

# LLM
llm_name = "gpt-3.5-turbo"
llm = ChatOpenAI(model_name=llm_name, temperature=0)

# Retrieval QA Chain
# question = "What is the course objective?"
question = "who is the instructor of this course?"
qa_chain = RetrievalQA.from_chain_type(llm, retriever=vectorstore.as_retriever())

result = qa_chain({"query": question})
print(result["result"])
