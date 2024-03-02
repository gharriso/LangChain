# See; https://www.mongodb.com/developer/products/atlas/rag-atlas-vector-search-langchain-openai/
 
import lancedb
from langchain_openai import OpenAIEmbeddings

from langchain_openai import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
import os
import sys



pdf_name = sys.argv[1]
OPENAI_API_KEY=os.environ["OPENAI_API_KEY"]

llm=ChatOpenAI(openai_api_key=OPENAI_API_KEY,model_name="gpt-4",max_tokens=1000)

# load the PDF into a vector store 
loader = PyPDFLoader(pdf_name)
data = loader.load()
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=0)
docs = text_splitter.split_documents(data)
vectorStore = FAISS.from_documents(docs, OpenAIEmbeddings())

# Define the vector store as a retriever
qa_retriever = vectorStore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 20},
    )
prompt_template = """Use the following pieces of context to answer the question at the end. If you don't know the answer, just say that you don't know, don't try to make up an answer.
{context}
Question: {question}
"""
PROMPT = PromptTemplate(template=prompt_template, input_variables=["context", "question"])

# Define the QA engine
qa = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=qa_retriever,
    return_source_documents=True,
    chain_type_kwargs={"prompt": PROMPT},
)

question = input("Please enter a question about the PDF: ")
docs = qa.invoke({"query": question})
print(docs["result"])





 
