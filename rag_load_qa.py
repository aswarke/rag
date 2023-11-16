from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.chains.question_answering import load_qa_chain

# Load Documents
loader = PyPDFLoader("docs/lec1.pdf")

# Split Documents
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=0)
splits = text_splitter.split_documents(loader.load())

# Embed and store splits
vectorstore = Chroma.from_documents(documents=splits, embedding=OpenAIEmbeddings())

# Similarity Search
# question = "who is the instructor of this course?"
# question = "what is the course objective?"
question = "who are the TAs of this course?"
docs = vectorstore.similarity_search(question)

# LLM
llm_name = "gpt-3.5-turbo"
llm = ChatOpenAI(model_name=llm_name, temperature=0)

# QA Chain
# chain = load_qa_chain(llm, chain_type="stuff")
chain = load_qa_chain(llm)
result = chain({"input_documents": docs, "question": question}, return_only_outputs=True)
print(result["output_text"])
