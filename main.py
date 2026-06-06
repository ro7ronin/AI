import time
import os
from langchain_community.document_loaders import PyPDFLoader
# from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_ollama import ChatOllama

start_time = time.perf_counter()
loader = PyPDFLoader("Books/1984.pdf")
docs = loader.load()

# from langchain.text_splitter import RecursiveCharacterTextSplitter
splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)

chunks = splitter.split_documents(docs)


# from langchain_ollama import OllamaEmbeddings

embeddings = OllamaEmbeddings(
    model="nomic-embed-text"
    ,base_url="http://localhost:11434"
)

# from langchain.vectorstores import Chroma
print(f"Loaded {len(docs)} pages")
print(f"Generated {len(chunks)} chunks")


# --------------------------------------------------EMBEDDING
embedding_start_time = time.perf_counter()
# db = Chroma.from_documents(
#     chunks,
#     embeddings,
#     persist_directory="./chroma_db"
# )

if not os.path.exists("./chroma_db"):
    db = Chroma.from_documents(
        chunks,
        embeddings,
        persist_directory="./chroma_db"
    )

db = Chroma(
    persist_directory="./chroma_db",
    embedding_function=embeddings
)


total_embedding_time = time.perf_counter()-embedding_start_time
print(f"Total embedding time: {total_embedding_time:.4f} seconds")
# --------------------------------------------------EMBEDDING ENDS


retriever = db.as_retriever(search_kwargs={"k": 3})



query = "What is the main topic of the document?"
# query = "Who wrote this book?"
# query = "How many characters are there in novel 1984?"

docs = retriever.invoke(query)




# from langchain_ollama import ChatOllama

llm = ChatOllama(
    model="qwen2.5:3b"
)

context = "\n".join([doc.page_content for doc in docs])

prompt = f"""
Answer only from the context.

Context:
{context}

Question:
{query}
"""

response = llm.invoke(prompt)

print(response.content)

end_time = time.perf_counter()
total_time = end_time - start_time

print(f"Total run time: {total_time:.4f} seconds")