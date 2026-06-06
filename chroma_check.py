from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import Chroma

embeddings = OllamaEmbeddings(model="nomic-embed-text")

texts = [
    "Java is a programming language.",
    "Python is widely used in AI."
]

db = Chroma.from_texts(
    texts,
    embeddings,
    persist_directory="./test_db"
)

print("Success")