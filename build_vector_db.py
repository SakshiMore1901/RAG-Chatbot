from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_community.vectorstores import Chroma
import os

documents = []

data_path = "data"

for file in os.listdir(data_path):
    loader = TextLoader(os.path.join(data_path, file))
    documents.extend(loader.load())

text_splitter = CharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50
)

texts = text_splitter.split_documents(documents)

embeddings = SentenceTransformerEmbeddings(
    model_name="all-MiniLM-L6-v2"
)

vector_db = Chroma.from_documents(
    texts,
    embeddings,
    persist_directory="chroma_db"
)

vector_db.persist()

print("Vector database created successfully!")