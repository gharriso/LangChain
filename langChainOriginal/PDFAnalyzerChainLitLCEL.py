# See; https://www.mongodb.com/developer/products/atlas/rag-atlas-vector-search-langchain-openai/
 

from langchain_openai import OpenAIEmbeddings
import chainlit as cl
from langchain_openai import ChatOpenAI
 
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.llms import Ollama 
from langchain.prompts import ChatPromptTemplate
from langchain.schema import StrOutputParser
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.schema.runnable import Runnable, RunnablePassthrough, RunnableConfig
from langchain_community.embeddings import OllamaEmbeddings
from langchain_google_genai import GoogleGenerativeAIEmbeddings


import streamlit as st  
import os
import logging
logging.basicConfig(format='%(asctime)s - %(message)s', level=logging.INFO)


if "OPENAI_API_KEY" not in os.environ:
    print("OPENAI_API_KEY not set")
    os._exit(1)

OPENAI_API_KEY=os.environ["OPENAI_API_KEY"]

GOOGLE_AI_KEY = os.getenv("GOOGLE_AI_KEY")

gpt4=ChatOpenAI(openai_api_key=OPENAI_API_KEY,model_name="gpt-4",max_tokens=5000)
gpt3=ChatOpenAI(openai_api_key=OPENAI_API_KEY,model_name='gpt-3.5-turbo-16k',max_tokens=1000)
gemini= ChatGoogleGenerativeAI(model="gemini-pro", google_api_key=GOOGLE_AI_KEY)
llama2=Ollama(model="llama2")

model=gemini

if "LLM" in os.environ:
    if os.environ["LLM"] == "gpt4":
        model=gpt4
        embeddings=OpenAIEmbeddings()
    elif os.environ["LLM"] == "gpt3":
        model=gpt3
        embeddings=OpenAIEmbeddings()
    elif os.environ["LLM"] == "llama2":
        model=llama2
        embeddings = OllamaEmbeddings()
    elif os.environ["LLM"] == "gemini":
        model=gemini
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=GOOGLE_AI_KEY)
    else:
        model=Ollama(model=os.environ["LLM"])
        embeddings = OllamaEmbeddings()
 
logging.info("model: {}".format(model))

@cl.on_chat_start
async def on_chat_start():
    # Request the user to upload a PDF file
    files = None
    while not files:
        files = await cl.AskFileMessage(
            content="Please upload a PDF file",
            accept=["application/pdf"],
            max_size_mb=125,
            timeout=180,
        ).send()

    file = files[0]

    # Notifying the user about processing
    processing_msg = cl.Message(
        content=f"Processing `{file.name}`..."
    )
    await processing_msg.send()


    vectorStoreName=os.path.splitext(file.name)[0]+'.faiss'
    print(vectorStoreName)
    if os.path.exists(vectorStoreName):
        processing_msg.content = f"Loading existingvector store"
        await processing_msg.update()
        persisted_vectorstore = FAISS.load_local(vectorStoreName, embeddings)
    else:
        processing_msg.content = f"Creating the vector store"
        await processing_msg.update()
 
        pdf_loader = PyPDFLoader(file_path=file.path)
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        docs = pdf_loader.load_and_split(text_splitter=text_splitter)
        vectorStore = FAISS.from_documents(docs, embeddings)
        vectorStore.save_local(vectorStoreName)
        persisted_vectorstore = FAISS.load_local(vectorStoreName, embeddings)

    # Inform the user about readiness
    processing_msg.content = f"I am ready to answer questions about `{file.name}`."
    await processing_msg.update()
    # Setting up the chat prompt template
    prompt_template = ChatPromptTemplate.from_template(
        """Answer questions based on the following: {context}

        Question: {question}"""
    )

    # Function to format document content
    def format_docs(docs):
        return "\n\n".join([doc.page_content for doc in docs])

    # Setting up the retriever and runnable
    retriever = persisted_vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 25},
    )
    runnable = (
            {"context": retriever | format_docs, "question": RunnablePassthrough()}
            | prompt_template
            | model
            | StrOutputParser()
    )

    # Storing the runnable in the user session
    cl.user_session.set("runnable", runnable)


# Function triggered upon receiving a message
@cl.on_message
async def on_message(message: cl.Message):
    # Retrieve the runnable from the user session
    runnable = cl.user_session.get("runnable")  # type: Runnable

    # Create a message object for response
    response_msg = cl.Message(content="")
    await response_msg.send()

    # Stream the response from the runnable
    async for chunk in runnable.astream(
            message.content,
            config=RunnableConfig(callbacks=[cl.LangchainCallbackHandler()]),
    ):
        await response_msg.stream_token(chunk)

    # Update the message with the final response
    await response_msg.update()




 
