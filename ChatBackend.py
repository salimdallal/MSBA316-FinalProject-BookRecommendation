
# Imports

import os

import pandas as pd
#from langchain.text_splitter import RecursiveCharacterTextSplitter

# from langchain.schema import Document # Import Document from langchain.schema

from langchain_openai import OpenAIEmbeddings

import sqlite3

__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')


from langchain_chroma import Chroma

#from langchain.chains import RetrievalQA

from langchain_community.llms import OpenAI
from langchain.chat_models import ChatOpenAI

#from langchain.prompts import PromptTemplate

from langchain.retrievers.self_query.base import SelfQueryRetriever
from langchain.chains.query_constructor.base import AttributeInfo

from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor

#import lark

import streamlit as st

openai_api_key = st.secrets['OPENAI_API_KEY']
print(f"key = {openai_api_key}")

# Initialize OpenAI embeddings
embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)

persist_directory = './chroma_persist'

# load the vector store
vector_store = Chroma(persist_directory=persist_directory, embedding_function=embeddings)



def create_chat_completion_like_iterable(docs):
  for doc in docs:
    message = AIMessage(
        content=doc.page_content,
        additional_kwargs={
            "title": doc.metadata['title'],
            "author": doc.metadata.get('author'),
            "category": doc.metadata.get('category'),
            "rating": doc.metadata.get('average_rating'),
            "page_count": doc.metadata.get('num_pages'),
            "rating_count": doc.metadata.get('ratings_count'),
        }
    )
    yield message





###############  *******  Contextual Retriever  *********  ################


#Create the compressor retreiver that will be used for the chatbot
llm = OpenAI( temperature=0, openai_api_key=openai_api_key) #model="gpt-4o-mini",
compressor = LLMChainExtractor.from_llm(llm)

compression_retriever = ContextualCompressionRetriever(
    base_compressor=compressor,
    base_retriever=vector_store.as_retriever()
)



# Functions

# Function to print the output in pretty format
def pretty_print_docs(docs):
    print(f"\n{'-' * 100}\n".join([f"Title: {d.metadata['title']}\nRating: {d.metadata['average_rating']}\nRating count: {d.metadata['ratings_count']}\n" 
    + d.page_content for i, d in enumerate(docs)]))


from langchain.schema import AIMessage, HumanMessage, SystemMessage







# Function to get the answer from the ContextualCompressionRetriever 
def getContextualReponse(question):
#question = "give me a the highest rating count for stephen king novels"
    compressed_docs = compression_retriever.get_relevant_documents(question)
    if not compressed_docs:
        print("No results found for your query.")
        return
    else:
        chat_completion_like_iterable = create_chat_completion_like_iterable(compressed_docs)
        return chat_completion_like_iterable
#    pretty_print_docs(removeDuplicates(compressed_docs))







###############  *******  SelfQuery Retriever  *********  ################




metadata_field_info = [
    AttributeInfo(
        name="title",
        description="The title of the novel from the title metadata",
        type="string",
    ),
        AttributeInfo(
        name="category",
        description="The category or subject the novel belongs to",
        type="string",
    ),
        AttributeInfo(
        name="author",
        description="The author of the novel",
        type="string",
    ),
        AttributeInfo(
        name="published_year",
        description="The year the novel was published",
        type="integer",
    ),
        AttributeInfo(
        name="average_rating",
        description="The average rating of the novel by the readers",
        type="decimal",
    ),
        AttributeInfo(
        name="num_pages",
        description="The number of pages the novel has",
        type="integer",
    ),
        AttributeInfo(
        name="ratings_count",
        description="The number of ratings given by readers for this novel",
        type="integer",
    )
]

document_content_description = "Novel in a library"

Self_llm = ChatOpenAI(model='gpt-4o-mini',temperature=0, openai_api_key=openai_api_key)

retriever = SelfQueryRetriever.from_llm(
    Self_llm,
    vector_store,
    document_content_description,
    metadata_field_info,
    verbose=True,
)



# Function to get the answer from the ContextualCompressionRetriever 
def getSelfQueryReponse(question):
#question = "give me a the highest rating count for stephen king novels"
    docs = retriever.get_relevant_documents(question)
    if not docs:
        print("No results found for your query.")
        return
    else:
        chat_completion_like_iterable = create_chat_completion_like_iterable(docs)
        return chat_completion_like_iterable
#    pretty_print_docs(removeDuplicates(compressed_docs))




















































def removeDuplicates(compressed_docs):
    unique_docs = set()
    unique_doc_list = []

    for doc in compressed_docs:
        # Create a unique identifier for each document using a combination of metadata attributes
        unique_id = (doc.metadata['title'], doc.metadata['author'], doc.page_content)

        if unique_id not in unique_docs:
            unique_docs.add(unique_id)
            unique_doc_list.append(doc)
    
    return unique_doc_list