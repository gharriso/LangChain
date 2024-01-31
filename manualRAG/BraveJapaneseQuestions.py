# See; https://www.mongodb.com/developer/products/atlas/rag-atlas-vector-search-langchain-openai/
from pymongo import MongoClient
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import MongoDBAtlasVectorSearch
from langchain_community.document_loaders import DirectoryLoader
from langchain_openai import OpenAI
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
#from langchain.memory import BufferMemory
import os

debug=0

# Check to see if the environment variables are set
# If not, set them
if "vectorUser" not in os.environ:
    print("vectorUser not set")
    os._exit(1)
if "OPENAI_API_KEY" not in os.environ:
    print("OPENAI_API_KEY not set")
    os._exit(1)

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

# Prompt the user to enter a question

def answerQuestion(debug, vectorStore):
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
    llm=OpenAI(openai_api_key=OPENAI_API_KEY,model_name="gpt-3.5-turbo-instruct"),
    chain_type="stuff",
    retriever=qa_retriever,
    return_source_documents=True,
    chain_type_kwargs={"prompt": PROMPT},
)

    docs = qa.invoke({"query": question})

    print(docs["result"])
    print()
    if debug:
        for doc in docs["source_documents"]:
            print(doc)
            print()
 

while True:
    answerQuestion(debug, vectorStore)
