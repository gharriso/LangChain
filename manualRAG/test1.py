# See; https://www.mongodb.com/developer/products/atlas/rag-atlas-vector-search-langchain-openai/
from pymongo import MongoClient
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import MongoDBAtlasVectorSearch
from langchain_community.document_loaders import DirectoryLoader
from langchain_openai import OpenAI
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
import gradio as gr
from gradio.themes.base import Base
import key_param

import os

MONGO_URI=os.environ["vectorUser"]
OPENAI_API_KEY=os.environ["OPENAI_API_KEY"]

client = MongoClient(MONGO_URI)
dbName = "vectorSearch"
collectionName = "tbj"
collection = client[dbName][collectionName]
ATLAS_VECTOR_SEARCH_INDEX_NAME = "vector_index"
EMBEDDING_FIELD_NAME = "embedding"


# NB: Data already loaded into MongoDB Atlas using the ingest.py script

embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)

#vectorStore = MongoDBAtlasVectorSearch( collection, embeddings )
vectorStore=MongoDBAtlasVectorSearch.from_connection_string(
    MONGO_URI,
    dbName + "." + collectionName,
    OpenAIEmbeddings(disallowed_special=()),
    index_name=ATLAS_VECTOR_SEARCH_INDEX_NAME,
)

question="Who is George Water"

print("similarity")

 
results = vectorStore.similarity_search_with_score(
    query=question,
    k=5,
)

# Display results
for result in results:
    print(result )

qa_retriever = vectorStore.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 25},
)


prompt_template = """Use the following pieces of context to answer the question at the end. If you don't know the answer, just say that you don't know, don't try to make up an answer.

{context}

Question: {question}
"""

PROMPT = PromptTemplate(
    template=prompt_template, input_variables=["context", "question"]
)


qa = RetrievalQA.from_chain_type(
    llm=OpenAI(),
    chain_type="stuff",
    retriever=qa_retriever,
    return_source_documents=True,
    chain_type_kwargs={"prompt": PROMPT},
)

docs = qa({"query": question})

print(docs["result"])
#print(docs["source_documents"])


def query_data(query):
    # Convert question to vector using OpenAI embeddings
    # Perform Atlas Vector Search using Langchain's vectorStore
    # similarity_search returns MongoDB documents most similar to the query    

    docs = vectorStore.similarity_search(query, K=1)
    
    print(docs)
    # Get the number of documents in the docs variable
    # If there are no documents, return "No results found"
    # Else, return the page_content of the first document
    if (len(docs) == 0):
        print( "No results found")
    else:
        print(docs[0].page_content)

# Leveraging Atlas Vector Search paired with Langchain's QARetriever

# Define the LLM that we want to use -- note that this is the Language Generation Model and NOT an Embedding Model
# If it's not specified (for example like in the code below),
# then the default OpenAI model used in LangChain is OpenAI GPT-3.5-turbo, as of August 30, 2023

    llm = OpenAI(openai_api_key=OPENAI_API_KEY, temperature=0)


# Get VectorStoreRetriever: Specifically, Retriever for MongoDB VectorStore.
# Implements _get_relevant_documents which retrieves documents relevant to a query.
    retriever = vectorStore.as_retriever( )
    docs = retriever.get_relevant_documents(query)
    print("docs")
    print(docs)


    return()

# Load "stuff" documents chain. Stuff documents chain takes a list of documents,
# inserts them all into a prompt and passes that prompt to an LLM.

    qa = RetrievalQA.from_chain_type(llm, chain_type="stuff", retriever=retriever)

# Execute the chain

    retriever_output = qa.invoke(query)
    print("retriever_output")
    print(retriever_output)


# Return Atlas Vector Search output, and output generated using RAG Architecture
    return retriever_output

#output=query_data("George Water")

#print(output)