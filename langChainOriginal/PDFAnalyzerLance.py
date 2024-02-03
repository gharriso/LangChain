# Answer questions about a PDF file using the RAG model
 
# TODO: Maintain the context of the conversation

import lancedb

from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import LanceDB
from langchain_community.document_loaders import PyPDFLoader
from langchain_openai import OpenAI
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

 
import random
import time
import os
import sys

debug=False

# Get the name of a PDF file from the command line
pdf_name = sys.argv[1]
if not os.path.exists(pdf_name):
    print("The PDF file does not exist. Exiting program.")
    os._exit(1)

# Load the document, split it into chunks, embed each chunk and load it into the vector store.
 
# Vector DB connection
vectorStore = lancedb.connect("/tmp/lancedb")
embeddings=OpenAIEmbeddings()
table=random.seed(time.time())
table = vectorStore.create_table(
    table,
    data=[
        {
            "vector": embeddings.embed_query("Hello World"),
            "text": "Hello World",
            "id": "1",
        }
    ],
    mode="overwrite",
)

# Load the document, split it into chunks, embed each chunk and load it into the vector store.
print(pdf_name)
loader = PyPDFLoader(pdf_name)
data = loader.load()

    # Split docs
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=0)
docs = text_splitter.split_documents(data)
vectorStore = LanceDB.from_documents(docs, OpenAIEmbeddings(), connection=table)

def answerQuestion(debug, vectorStore):
    question = input("Please enter a question about the PDF: ")
    # If question is blank, exit program
    if not question:
        print("No question entered. Exiting program.")
        os._exit(1)


# Display results if debug is true

    if debug: 
        results = vectorStore.similarity_search_with_score(
    query=question,
    k=5,)   
        for result in results:
        # print just the page_content field
            print(result[0].page_content )

    qa_retriever = vectorStore.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 25},
)


    prompt_template = """Use the following pieces of context to answer the question at the end. If you don't know the answer, just say that you don't know, don't try to make up an answer.
{context}
Question: {question}
"""

    PROMPT = PromptTemplate(
    template=prompt_template, input_variables=["context", "question"],history_variables=["chat_history"]
)


    qa = RetrievalQA.from_chain_type(
    llm=OpenAI(),
    chain_type="stuff",
    retriever=qa_retriever,
    return_source_documents=True,
    chain_type_kwargs={"prompt": PROMPT},
)

    docs = qa.invoke({"query": question})

    print(docs["result"])
    print()

while True:
    answerQuestion(debug, vectorStore)
