# See; https://www.mongodb.com/developer/products/atlas/rag-atlas-vector-search-langchain-openai/
 

from langchain_openai import OpenAIEmbeddings

from langchain_openai import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_community.vectorstores import LanceDB
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.llms import Ollama 
from langchain.memory import ConversationBufferWindowMemory
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain
import os
import sys
import logging

logging.basicConfig(format='%(asctime)s - %(message)s', level=logging.INFO)

debug=False
verbose=False

if os.environ.get("DEBUG", "").lower() == "true":
    debug=True
    verbose=True
    logging.getLogger().setLevel(logging.DEBUG)
    
chat_history = []
memory = ConversationBufferWindowMemory( k=5)  #TODO: Use this instead of chat_history

if "OPENAI_API_KEY" not in os.environ:
    print("OPENAI_API_KEY not set")
    os._exit(1)

if "GOOGLE_AI_KEY" not in os.environ:
    print("GOOGLE_API_KEY not set")
    os._exit(1)
    
 

# If there are no arguments, exit the program
if len(sys.argv) < 2:
    print("Usage:\ python pdfAnalyzerTextBased.py <pdf_name> [gpt3|gpt4|llama2]")
    os._exit(1)
    
pdf_name = sys.argv[1]
if not os.path.exists(pdf_name):
    print("The PDF file does not exist. Exiting program.")
    os._exit(1)
 
OPENAI_API_KEY=os.environ["OPENAI_API_KEY"]
gpt4=ChatOpenAI(openai_api_key=OPENAI_API_KEY,model_name="gpt-4",max_tokens=1000)
gpt3=ChatOpenAI(openai_api_key=OPENAI_API_KEY,model_name='gpt-3.5-turbo-16k',max_tokens=1000)
gemini= ChatGoogleGenerativeAI(model="gemini-pro", google_api_key=os.environ["GOOGLE_AI_KEY"])
 


if len(sys.argv) > 2:
    if sys.argv[2] == "gpt4":
        llm=gpt4
    elif sys.argv[2] == "gpt3":
        llm=gpt3
    elif sys.argv[2] == "gemini":
        llm=gemini
    else:
        llm=Ollama(model=sys.argv[2])
else:
    llm=gpt3
    
logging.info("model: {}".format(llm))
pdfShortName=os.path.basename(pdf_name)

logging.info("{} Questions".format(pdfShortName))  # Text input field for the user to enter a topic  
logging.info("Loading PDF into vector store")

# Load the document, split it into chunks, embed each chunk and load it into the vector store.
 # Check to see if docs is defined

 
pdfShorterName=os.path.splitext(pdfShortName)[0]+'.faiss'
logging.info(pdfShorterName)
embeddings = OpenAIEmbeddings()
if os.path.exists(pdfShorterName):
    logging.info("Loading the vector store from the file")
    vectorStore = FAISS.load_local(pdfShorterName, embeddings)
else:
    logging.info("Creating the vector store")
    loader = PyPDFLoader(pdf_name)
    data = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    print(data)
    docs = text_splitter.split_documents(data)
    vectorStore = FAISS.from_documents(docs, OpenAIEmbeddings())
    logging.info("Saving the vector store to the file")
    vectorStore.save_local(pdfShorterName)
    vectorStore = FAISS.load_local(pdfShorterName, embeddings)
    
prompt_template = """Answer the question in detail as truthfully as possible from the context given to you.
    If you do not know the answer to the question, simply respond with "I don't know. Can you ask another question".
    If questions are asked where there is no relevant context available, simply respond 
    with "I don't know. Please ask a question relevant to the documents" 
    Combine the chat history and the context into a stand alone question. 
            Context: {context}
            chat history: {chat_history}
            Question: {question}
"""

PROMPT = PromptTemplate(
    template=prompt_template, input_variables=["context", "chat_history","question"] 
    )
qa_retriever = vectorStore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 50},
    )

chain = ConversationalRetrievalChain.from_llm(
        llm=llm, retriever=qa_retriever, return_source_documents=True,verbose=verbose,
        combine_docs_chain_kwargs={'prompt': PROMPT})

def answerQuestion():
    print()
    question = input("Please enter a question about the PDF: ")
    # If question is blank, exit program
    if not question:
        print("No question entered. Exiting program.")
        os._exit(1)

    docs = chain.invoke({"question": question, "chat_history": chat_history})
 
    print(docs["answer"])
    if debug:
        print(docs)
        print(docs["answer"])
    chat_history.append((question,docs["answer"]) ) 

    print()


while True:
    answerQuestion()

 
