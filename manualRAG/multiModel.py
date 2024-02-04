# See; https://www.mongodb.com/developer/products/atlas/rag-atlas-vector-search-langchain-openai/
 
 

from langchain_openai import OpenAI,ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_community.llms import Ollama 
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_openai import OpenAIEmbeddings
import lancedb
from langchain_community.vectorstores import LanceDB

# List all the methods in LanceDB
for module in dir(LanceDB):
    print(module)

import os
import sys

debug=False

if "vectorUser" not in os.environ:
    print("vectorUser not set")
    os._exit(1)

if "OPENAI_API_KEY" not in os.environ:
    print("OPENAI_API_KEY not set")
    os._exit(1)

OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]
gpt4=ChatOpenAI(openai_api_key=OPENAI_API_KEY,model_name="gpt-4",max_tokens=1000)
gpt3=OpenAI(openai_api_key=OPENAI_API_KEY,max_tokens=1000)

llm=gpt3 #Default to GPT-3

if len(sys.argv) < 1:
    print("Usage: python3 multiModel.py <PDF file> [gpt4|phi|llama2]")
    os._exit(1)

if  len(sys.argv) > 2:  
    if sys.argv[2] == "gpt4":
        llm = gpt4
    elif sys.argv[2] == "phi":
        llm = Ollama(model="phi")
    elif sys.argv[2] == "llama2":
        llm = Ollama(model="llama2")

pdf_name = sys.argv[1]
if not os.path.exists(pdf_name):
    print("The PDF file does not exist. Exiting program.")
    os._exit(1)
 
# Vector DB connection
def loadFile(pdf_name):
    vectorStore = lancedb.connect("/tmp/lancedb")
    tables=vectorStore.table_names()
    
    embeddings=OpenAIEmbeddings()
    table= os.path.basename(pdf_name)
    # if the table is already in the tables list, delete it
    if not table in tables:
        print('Creating new lanceDB table')
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
        
        loader = PyPDFLoader(pdf_name)
        data = loader.load()

        # Split docs
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=0)
        docs = text_splitter.split_documents(data)
        vectorStore = LanceDB.from_documents(docs, OpenAIEmbeddings(), connection=table)
    else:
        print('Using existing lanceDB table')
        table=vectorStore.open_table(table)
        dir(table)
  
        vectorStore = vectorStore.from_existing_index(connection=table)

    return vectorStore



def answerQuestion(debug, vectorStore):
    print()
    question = input("Please enter a question about The Brave Japanese: ")
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
    llm=llm,
    chain_type="stuff",
    retriever=qa_retriever,
    return_source_documents=True,
    chain_type_kwargs={"prompt": PROMPT},
)

    docs = qa.invoke({"query": question})


    print()
    if debug:
        for doc in docs["source_documents"]:
            print(doc.page_content)
            print()
    print(docs["result"])
 
vectorStore = loadFile(pdf_name)

while True:
    answerQuestion(debug, vectorStore)
