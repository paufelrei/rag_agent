# here I define my langchain/openai based function
import os.path
import tempfile
from typing import Sequence, Optional, Any
from uuid import UUID

import pandas as pd
import streamlit as st

from langchain_community.document_loaders import PyMuPDFLoader
from langchain_community.vectorstores.chroma import Chroma
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_core.callbacks.base import BaseCallbackHandler


@st.cache_resource(ttl='1h')
def process_documents(uploads, key):
    # here pdf documents get process to a vector db

    documents = []

    temp_dir = tempfile.TemporaryDirectory()

    for file in uploads:
        temp_path = os.path.join(temp_dir.name, file.name)
        with open(temp_path, "wb") as f:
            f.write(file.getvalue())

        loader = PyMuPDFLoader(temp_path)

        documents.extend(loader.load())

    # split text and put in db
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=200)
    documents_chunks = text_splitter.split_documents(documents)
    embeddings_model = OpenAIEmbeddings(api_key=key)

    db = Chroma.from_documents(documents_chunks, embeddings_model)

    retriever = db.as_retriever()

    return retriever


class StreamHandler(BaseCallbackHandler):
    def __init__(self, container, initial_text=''):
        self.container = container
        self.text = initial_text

    def on_llm_new_token(self, token: str, **kwargs):
        self.text += token
        self.container.markdown(self.text)


class PostMessageHandler(BaseCallbackHandler):
    def __init__(self, msg: st.write):
        BaseCallbackHandler.__init__(self)
        self.msg = msg
        self.sources =  []

    def on_retriever_end(self, documents, *, run_id, parent_run_id, **kwargs):
        source_ids = []

        for doc in documents:
            metadata = {"source": doc.metadata["source"],
            "page": doc.metadata["page"],
            "content": doc.page_content[:100]}

        idx = (metadata["source"], metadata["page"])
        if idx not in source_ids:
            source_ids.append(idx)
            self.sources.append(metadata)

    def on_llm_end(self, documents, *, run_id, parent_run_id, **kwargs):
        if len(self.sources):
            st.markdown("__Sources:__" +  "\n")
            st.dataframe(data=pd.DataFrame(self.sources[:3]), width=1000)




