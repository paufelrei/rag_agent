# here I define my openai based function
import os.path
import tempfile

from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

def process_documents(uploads):
    # here pdf documents get process to a vector db

    documents = []

    temp_dir = tempfile.TemporaryDirectory()

    for file in uploads:
        temp_path = os.path.join(temp_dir.name, file.name)
        with open(temp_path, "wb") as f:
            f.write(file.getvalue())

        loader = PyMuPDFLoader(temp_path)

        documents.extend(loader.load())

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=200)

    documents_chunks = text_splitter(documents)

