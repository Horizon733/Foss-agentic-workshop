from typing import List
from pathlib import Path
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores.chroma import Chroma
from langchain_community.document_loaders import DirectoryLoader, UnstructuredFileLoader, PyPDFLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.schema import Document
from langchain.schema.embeddings import Embeddings

def extract_documents(docs_folder: str) -> List[Document]:
    print(f"Extracting files from: {Path(docs_folder).absolute()}")
    if not Path(docs_folder).exists():
        raise SystemExit(f"Directory '{docs_folder}' does not exist.")

    pdf_loader = DirectoryLoader(
        docs_folder,
        glob="**/*.pdf",
        loader_cls=PyPDFLoader,
        show_progress=True,
    )

    other_loader = DirectoryLoader(
        docs_folder,
        glob="**/*.txt",
        loader_cls=UnstructuredFileLoader,
        show_progress=True,
    )

    documents = pdf_loader.load() + other_loader.load()
    return documents

def create_chunks(documents: List[Document], chunk_size: int, chunk_overlap: int) -> List[Document]:
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
    )
    return text_splitter.split_documents(documents)

def embeddings_factory() -> HuggingFaceEmbeddings:
    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

def create_chroma_collection(embeddings: Embeddings, docs: List[Document]) -> Chroma:
    return Chroma.from_documents(
        documents=docs,
        embedding=embeddings,
        persist_directory="./chroma_db",
    )

def main():
    docs_folder = "docs"
    chunk_size = 1000
    chunk_overlap = 80

    docs = extract_documents(docs_folder)
    print(f"{len(docs)} documents extracted.")

    chunks = create_chunks(docs, chunk_size, chunk_overlap)
    print(f"{len(chunks)} chunks created.")
    for i, chunk in enumerate(chunks[:3]):
        print(f"chunk {i}")
        print(chunk)

    embeddings = embeddings_factory()

    create_chroma_collection(
        embeddings=embeddings,
        docs=chunks,
    )
    print("Chroma collection created successfully")

if __name__ == "__main__":
    main()