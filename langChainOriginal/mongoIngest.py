import os
import sys

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import MongoDBAtlasVectorSearch
from pymongo import MongoClient

MONGO_URI = os.environ["LOCAL_ATLAS"]

# Note that if you change this, you also need to change it in `rag_mongo/chain.py`
mongoDbName = "vectorSearch"
atlasVectorIndex = "vectorEmbedding"
embeddingFieldName = "embedding"
client = MongoClient(MONGO_URI)
db = client[mongoDbName]



# Get the name of the PDF document from as the first command line argument
if len(sys.argv) < 2:
    print("No PDF file specified. Exiting program.")
    os._exit(1)
    
pdf_name = sys.argv[1]
if not os.path.exists(pdf_name):
    print("The PDF file does not exist. Exiting program.")
    os._exit(1)
pdfShortName=os.path.basename(pdf_name)
pdfShorterName=os.path.splitext(pdfShortName)[0]+'.faiss'
mongoDbCollection = db[pdfShorterName]
# create index on mongoDbCollection
mongoDbCollection.create_index([(embeddingFieldName, "2dsphere")])

if __name__ == "__main__":
    # Load docs
    loader = PyPDFLoader(pdf_name)
    data = loader.load()

    # Split docs
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    docs = text_splitter.split_documents(data)

    # Insert the documents in MongoDB Atlas Vector Search
    _ = MongoDBAtlasVectorSearch.from_documents(
        documents=docs,
        embedding=OpenAIEmbeddings(disallowed_special=()),
        collection=mongoDbCollection,
        index_name=atlasVectorIndex,
    )
    # create a mongodb aggregation pipeline to find the nearest neighbors
    # of a given query
    pipeline = [
        {
            "$geoNear": {
                "near": {"type": "Point", "coordinates": [0, 0]},
                "distanceField": "distance",
                "spherical": True,
                "key": embeddingFieldName,
            }
        }
    ]
 
     
