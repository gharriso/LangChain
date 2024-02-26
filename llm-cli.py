
 

from langchain_openai import ChatOpenAI

from langchain_community.llms import Ollama 
from langchain.memory import ConversationBufferWindowMemory
from langchain_openai import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_community.vectorstores import LanceDB
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.llms import Ollama 
from langchain.memory import ConversationBufferWindowMemory
from langchain_community.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain
from langchain.schema import StrOutputParser
from langchain_google_genai import ChatGoogleGenerativeAI

import os
import sys
import logging

logging.basicConfig(format='%(asctime)s - %(message)s', level=logging.ERROR)

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
    

 
OPENAI_API_KEY=os.environ["OPENAI_API_KEY"]
gpt4=ChatOpenAI(openai_api_key=OPENAI_API_KEY,model_name="gpt-4",max_tokens=1000)
gpt3=ChatOpenAI(openai_api_key=OPENAI_API_KEY,model_name='gpt-3.5-turbo-16k',max_tokens=1000)
gemini= ChatGoogleGenerativeAI(model="gemini-pro", google_api_key=os.environ["GOOGLE_AI_KEY"])
llm = gpt3
prompt = 'gpt3> '
    
 



# Set the initial model to GPT-3


# Initialize the chat history
chat_history = []

while True:
    # Get the user's input
    user_input = input(prompt)
    if not user_input:
        sys.exit(0)

    # Check if the user wants to switch models
    if user_input.startswith('use '):
        model_name = user_input.split(' ')[1]
        if model_name == 'gpt3':
            llm = gpt3
            prompt = 'gpt3> '
        elif model_name == 'gpt4':
            llm = gpt4
            prompt = 'gpt4> '
        elif model_name == 'gemini':
            llm = gemini
            prompt = 'gemini> '
        else:
            print('Unknown model:', model_name)
    else:
        # Add the user's input to the chat history
        chat_history.append(('user', user_input))


        chain = llm | StrOutputParser()
        # Invoking the chain, storing the result
        response = chain.invoke(user_input)
        # Displaying the generated title

        # Add the model's response to the chat history
        chat_history.append((prompt[:-2], response))

        # Print the model's response
        print(prompt, response)