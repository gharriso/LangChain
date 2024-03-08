
 

from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic 
from langchain_community.llms import Ollama
from langchain.memory import ConversationBufferWindowMemory
from langchain_openai import ChatOpenAI
from langchain_community.llms import Ollama 
from langchain.memory import ConversationBufferWindowMemory
from langchain.schema import StrOutputParser
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import ConversationChain
from langchain_openai import ChatOpenAI
from langchain.memory import ConversationBufferMemory

import os
import logging

logging.basicConfig(format='%(asctime)s - %(message)s', level=logging.ERROR)

debug=False
verbose=False
useAll=False

def selectModel(model_name):
    #If we find the model name in the array, we select it
    if model_name in modelsArray:
        llm = modelsArray[model_name]
        prompt = model_name+'> '
    else:
        print('Unknown model:', model_name)
    return prompt,llm


if os.environ.get("DEBUG", "").lower() == "true":
    debug=True
    verbose=True
    logging.getLogger().setLevel(logging.DEBUG)
    
 

if "OPENAI_API_KEY" not in os.environ:
    print("OPENAI_API_KEY not set")
    os._exit(1)

if "GOOGLE_AI_KEY" not in os.environ:
    print("GOOGLE_API_KEY not set")
    os._exit(1)
   
if "ANTHROPIC_API_KEY" not in os.environ:
    print("ANTHROPIC_API_KEY not set")
    os._exit(1)
 
modelsArray= {
    "gpt3": ChatOpenAI(openai_api_key=os.environ["OPENAI_API_KEY"],model_name='gpt-3.5-turbo-16k',max_tokens=1000),
    "gpt4": ChatOpenAI(openai_api_key=os.environ["OPENAI_API_KEY"],model_name="gpt-4",max_tokens=1000),
    "gemini": ChatGoogleGenerativeAI(model="gemini-pro", google_api_key=os.environ["GOOGLE_AI_KEY"]),
    "claude": ChatAnthropic(model="claude-2.1",anthropic_api_key=os.environ["ANTHROPIC_API_KEY"] ),
    "llama2": Ollama(model="llama2")
}
prompt,llm=selectModel("gpt3")
    
  
conversation = ConversationChain(
    llm=llm,
    verbose=verbose,
    memory=ConversationBufferMemory()
)

while True:
    # Get the user's input
    user_input = input(prompt)
    if not user_input:
        print('Goodbye!')
        os._exit(0)

    # Check if the user wants to switch models
    if user_input.startswith('use '):
        model_name = user_input.split(' ')[1]
        prompt,llm=selectModel(model_name)
        conversation = ConversationChain(
            llm=llm,
            verbose=verbose,
            memory=ConversationBufferMemory()
        )
    else:
        response = conversation.predict(input=user_input)
        print(response)