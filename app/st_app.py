import os

import streamlit as st

from openai_func.functions import process_documents, StreamHandler, PostMessageHandler
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.chat_message_histories import StreamlitChatMessageHistory
from operator import itemgetter

from st_helper import format_docs

#get open_ai key
key = st.secrets["API_KEY"]
# start page
st.set_page_config(page_title='RAG-Agent', page_icon="RAG")
st.title('Welcome to the RAG-Chat!')


# create sidebar for upload
uploaded_file = st.sidebar.file_uploader(label='upload files here:', type=["pdf"], accept_multiple_files=True)

if not uploaded_file:
    st.info("Please upload file to continue!")
    st.stop()

retriever = process_documents(uploaded_file, key=key)

# get openai conn
chatgpt = ChatOpenAI(model_name='gpt-3.5-turbo', temperature=0.1, streaming=True, api_key=key)


# basic prompt
bp_template = '''You are  a Data Scientist who is writing an application. Use the following pieces of context to complete the task at the end. 
                  
                  {context}
                  
                  Question: {question}'''

prompt = ChatPromptTemplate.from_template(bp_template)

# qu chain

rag_chain = ({"context": itemgetter("question")
              |
              retriever
              |
              format_docs,
              "question": itemgetter("question")}
            | prompt
            | chatgpt)

# keep message history
st_message_hist = StreamlitChatMessageHistory(key="langchain_messages")

# start message
if len(st_message_hist.messages) == 0:
     st_message_hist.add_ai_message("Go on!")

# show message in app
for msg in st_message_hist.messages:
    st.chat_message(msg.type).write(msg.content)

# new prompt reaction
if user_prompt := st.chat_input():
    st.chat_message("human").write(user_prompt)
    with st.chat_message("ai"):
        stream_handler = StreamHandler(st.empty())
        sources_container = st.write("")
        pm_handler = PostMessageHandler(sources_container)
        config = {"callbacks": [stream_handler, pm_handler]}
        # get response
        response = rag_chain.invoke({"question": user_prompt}, config)





