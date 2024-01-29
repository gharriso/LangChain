# See https://github.com/langchain-ai/langchain/tree/master/templates/rag-mongo#mongodb-setup
# and https://www.mongodb.com/developer/products/atlas/boosting-ai-build-chatbot-data-mongodb-atlas-vector-search-langchain-templates-using-rag-pattern/
from fastapi import FastAPI
from fastapi.responses import RedirectResponse
from langserve import add_routes
from rag_mongo import chain as rag_mongo_chain




app = FastAPI()

add_routes(app, rag_mongo_chain, path="/rag-mongo")
 
@app.get("/")
async def redirect_root_to_docs():
    return RedirectResponse("/docs")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
