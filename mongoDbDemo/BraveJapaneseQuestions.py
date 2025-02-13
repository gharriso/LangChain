# See; https://www.mongodb.com/developer/products/atlas/rag-atlas-vector-search-langchain-openai/
from pymongo import MongoClient
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import MongoDBAtlasVectorSearch

from langchain_openai import ChatOpenAI
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents.stuff import create_stuff_documents_chain
from langchain.prompts import PromptTemplate
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
import os

debug=True

# Check to see if the environment variables are set
# If not, set them
if "LOCAL_ATLAS" not in os.environ:
    print("vectorUser not set")
    os._exit(1)
if "OPENAI_API_KEY" not in os.environ:
    print("OPENAI_API_KEY not set")
    os._exit(1)

MONGO_URI=os.environ["LOCAL_ATLAS"]
OPENAI_API_KEY=os.environ["OPENAI_API_KEY"]

client = MongoClient(MONGO_URI)
dbName = "vectorDemo"
collectionName = "tbj"
collection = client[dbName][collectionName]
ATLAS_VECTOR_SEARCH_INDEX_NAME = "vector1"
EMBEDDING_FIELD_NAME = "embedding"

# NB: Data already loaded into MongoDB Atlas using the ingest.py script

embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)

vectorStore = MongoDBAtlasVectorSearch.from_connection_string(
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
            k=5,
        )   
        for result in results:
            # print just the page_content field
            print(result[0].page_content)

    retriever = vectorStore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 25},
    )
    docs = retriever.invoke(question)
    llm = ChatOpenAI(openai_api_key=OPENAI_API_KEY, model_name="gpt-4o")
    messageTemplate = """
        Answer the user's questions based on the below context. 
        If the context doesn't contain any relevant information to the question, 
        don't make something up and just say "I don't know":

        <context>
        {context}
        </context>
"""
    question_answering_prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                messageTemplate,
            ),
            MessagesPlaceholder(variable_name="messages"),
        ]
    )

    document_chain = create_stuff_documents_chain(llm, question_answering_prompt)

    result = document_chain.invoke(
        {
            "context": docs,
            "messages": [
                HumanMessage(content=question)
            ],
        }
    )
    print(result)
    # Print the structure of the result object
 
   

while True:
    answerQuestion(debug, vectorStore)
