# See; https://www.mongodb.com/developer/products/atlas/rag-atlas-vector-search-langchain-openai/
 
import lancedb
from langchain_openai import OpenAIEmbeddings

from langchain_openai import OpenAI,ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_community.vectorstores import LanceDB
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import streamlit as st  
import os
import sys


if "OPENAI_API_KEY" not in os.environ:
    print("OPENAI_API_KEY not set")
    os._exit(1)

pdf_name = sys.argv[1]
if not os.path.exists(pdf_name):
    print("The PDF file does not exist. Exiting program.")
    os._exit(1)
 
OPENAI_API_KEY=os.environ["OPENAI_API_KEY"]

vectorStore = lancedb.connect("/tmp/lancedb")
embeddings=OpenAIEmbeddings()
pdfShortName=os.path.basename(pdf_name)
table = vectorStore.create_table(
    pdfShortName,
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


st.title("""{table} Questions""")  # Text input field for the user to enter a topic  
question = st.text_input("""please enter a question about {pdfShortName}""")
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
llm=ChatOpenAI(openai_api_key=OPENAI_API_KEY,model_name="gpt-4"),
chain_type="stuff",
retriever=qa_retriever,
return_source_documents=True,
chain_type_kwargs={"prompt": PROMPT},
)

 
if topic:   
    with st.spinner("Creating story title"):
        chain = prompt | model | StrOutputParser()   
        response = chain.invoke({"topic": topic})          # Displaying the generated title          
        st.write(response)

docs = qa.invoke({"query": question})
st.write(docs["result"])


 
