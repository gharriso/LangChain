
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic 
from langchain_community.llms import Ollama
from langchain_groq import ChatGroq
from langchain_google_genai import ChatGoogleGenerativeAI

from langchain_openai import ChatOpenAI
from langchain_community.llms import Ollama 
import os

 
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
    #"gemini1.5": ChatGoogleGenerativeAI(model="gemini-1.5", google_api_key=os.environ["GOOGLE_AI_KEY"]),
    "claude3-sonnet": ChatAnthropic(model="claude-3-sonnet-20240229",anthropic_api_key=os.environ["ANTHROPIC_API_KEY"] ),
    "claude3-haiku": ChatAnthropic(model="claude-3-haiku-20240307",anthropic_api_key=os.environ["ANTHROPIC_API_KEY"] ),
    "claude3-opus": ChatAnthropic(model="claude-3-opus-20240229",anthropic_api_key=os.environ["ANTHROPIC_API_KEY"] ), 
    #"llama2": Ollama(model="llama2"),
    #"llama2:13b": Ollama(model="llama2:13b"),
    #"groq-gemma":ChatGroq( groq_api_key=os.environ["GROQ_API_KEY"], model_name="gemma-7b-it"),
    "groq-mixtral":ChatGroq( groq_api_key=os.environ["GROQ_API_KEY"], model_name="mixtral-8x7b-32768"),
    "groq-llama":ChatGroq( groq_api_key=os.environ["GROQ_API_KEY"], model_name="llama2-70b-4096"),
    #"phi": Ollama(model="phi"),
    #"mistral": Ollama(model="mistral"), # TODO: Check that models exist
    #"gemma": Ollama(model="gemma")
}

if "OPENAI_API_KEY" not in os.environ:
    print("OPENAI_API_KEY not set")
    os._exit(1)

if "GOOGLE_AI_KEY" not in os.environ:
    print("GOOGLE_API_KEY not set")
    os._exit(1)
   
if "ANTHROPIC_API_KEY" not in os.environ:
    print("ANTHROPIC_API_KEY not set")
    os._exit(1)
    

def selectModel(model_name):
    #If we find the model name in the array, we select it
    if model_name in modelsArray:
        llm = modelsArray[model_name]
    else:
        print('Unknown model:', model_name)
        llm=modelsArray["gpt3"]
    prompt = model_name+'> '    
    return prompt,llm