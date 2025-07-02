from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

def ingest_pdf(file_path="data/textbook.pdf", persist_path="faiss_index"):
    loader= PyPDFLoader(file_path)
    pages= loader.load()
    splitter= RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    docs= splitter.split_documents(pages)
    embeddings=HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectorstore=FAISS.from_documents(docs, embeddings)
    vectorstore.save_local(persist_path)
    print("Ingestion complete")

if __name__ == "__main__":
    ingest_pdf()
