# See; https://www.mongodb.com/developer/products/atlas/rag-atlas-vector-search-langchain-openai/
 

from langchain_openai import OpenAIEmbeddings

from langchain_openai import OpenAI,ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_community.vectorstores import LanceDB
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.llms import Ollama 
from langchain_community.vectorstores import Neo4jVector
from langchain.vectorstores import FAISS
import os
import sys
import datetime
import pytz

debug=False 

if "OPENAI_API_KEY" not in os.environ:
    print("OPENAI_API_KEY not set")
    os._exit(1)

pdf_name = sys.argv[1]
if not os.path.exists(pdf_name):
    print("The PDF file does not exist. Exiting program.")
    os._exit(1)
 
OPENAI_API_KEY=os.environ["OPENAI_API_KEY"]

gpt4=ChatOpenAI(openai_api_key=OPENAI_API_KEY,model_name="gpt-4",max_tokens=1000)
gpt3=ChatOpenAI(openai_api_key=OPENAI_API_KEY,model_name='gpt-3.5-turbo-16k',max_tokens=1000)
llama2=Ollama(model="llama2")
llm=gpt3

pdfShortName=os.path.basename(pdf_name)

print("{} Questions".format(pdfShortName))  # Text input field for the user to enter a topic  
print("Loading PDF into vector store")

# Load the document, split it into chunks, embed each chunk and load it into the vector store.
 # Check to see if docs is defined

 
pdfShorterName=os.path.splitext(pdfShortName)[0]+'.faiss'
print(pdfShorterName)
embeddings = OpenAIEmbeddings()
if os.path.exists(pdfShorterName):
    print("Loading the vector store from the file")
    vectorStore = FAISS.load_local(pdfShorterName, embeddings)
else:
    print("Creating the vector store")
    loader = PyPDFLoader(pdf_name)
    data = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    docs = text_splitter.split_documents(data)
    vectorStore = FAISS.from_documents(docs, OpenAIEmbeddings())
    print("Saving the vector store to the file")
    vectorStore.save_local(pdfShorterName)
    vectorStore = FAISS.load_local(pdfShorterName, embeddings)

qa_retriever = vectorStore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 50},
    )
prompt_template = """Use the following pieces of context to answer the question at the end. 
If you don't know the answer, just say that you don't know, don't try to make up an answer.
{context}
Question: {question}
"""
PROMPT = PromptTemplate(
template=prompt_template, input_variables=["context", "question"],history_variables=["chat_history"]
)

qa = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=qa_retriever,
    return_source_documents=True,
    chain_type_kwargs={"prompt": PROMPT},
)

def answerQuestion(debug, vectorStore):
    question = input("Please enter a question about the PDF: ")
    # If question is blank, exit program
    if not question:
        print("No question entered. Exiting program.")
        os._exit(1)

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

 
