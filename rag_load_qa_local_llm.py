from langchain.chains.question_answering import load_qa_chain
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.embeddings import GPT4AllEmbeddings
from langchain.llms import GPT4All

# Load Documents
loader = PyPDFLoader("docs/lec1.pdf")

# Split Documents
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
splits = text_splitter.split_documents(loader.load())

# Embed and store splits. Download and use GPT4All embeddings locally.
vectorstore = Chroma.from_documents(documents=splits, embedding=GPT4AllEmbeddings())

# Similarity Search
question = "who is the instructor of this course?"
# question = "what is the course objective?"
# question = "who are the TAs of this course?"
# question = "what are different flavors of project?"
docs = vectorstore.similarity_search(question, k=10)

# LLM
llm = GPT4All(model="/Users/awarke/models/GPT4All/gpt4all-13b-snoozy-q4_0.gguf")

# QA Chain
chain = load_qa_chain(llm, chain_type="stuff")
result = chain({"input_documents": docs, "question": question}, return_only_outputs=True)
print(result["output_text"])
