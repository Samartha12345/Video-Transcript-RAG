"""Split one transcript, embed the passages and retrieve matching evidence."""

from uuid import uuid4

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter


def index_transcript(text: str, source: str):
    document = Document(page_content=text, metadata={"source": source})
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_documents([document])
    if not chunks:
        raise ValueError("The transcript is empty.")
    for number, chunk in enumerate(chunks, start=1):
        chunk.metadata["passage"] = number
    encoder = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"},
    )
    # An independent collection keeps repeated notebook runs and videos separate.
    store = Chroma.from_documents(
        chunks, encoder, collection_name=f"transcript_{uuid4().hex}"
    )
    return store.as_retriever(search_kwargs={"k": min(4, len(chunks))})
