
 
from models import modelsArray

from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory
import streamlit as st  

import os
import logging
import time 
import sys

logging.basicConfig(format='%(asctime)s - %(message)s', level=logging.ERROR)

debug=False
verbose=False
useAll=False
st.title("Ask the LLMs")  # Text input field for the user to enter a topic  


def selectModel(model_name):
    #If we find the model name in the array, we select it
    prompt="unknown"
    llm="unknown"
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
 



# If the user specified a model on the command line, use that model
model_name='gpt3'
if len(sys.argv) > 1:
    model_name = sys.argv[1]
    if model_name == 'all':
        useAll=True
        prompt='all> '
        logging.info('Now using all models')
    else:
        prompt,llm=selectModel(model_name)
        conversation = ConversationChain(
            llm=llm,
            verbose=verbose,
            memory=ConversationBufferMemory()
        )
        logging.info('Now using model:', model_name)
else:
    prompt,llm=selectModel("gpt3")
    logging.info('Now using model gpt3')
    
  
conversation = ConversationChain(
    llm=llm,
    verbose=verbose,
    memory=ConversationBufferMemory()
)


user_input = st.text_input("Question for  {}".format( prompt))
if user_input:
    with st.spinner("Asking {}".format(llm)):
    # Check if the user wants to switch models
        if user_input.startswith('use '):
            model_name = user_input.split(' ')[1]
            if model_name == 'all':
                useAll=True
                prompt='all> '
            else:
                useAll=False
                _,llm=selectModel(model_name)
                conversation = ConversationChain(
                    llm=llm,
                    verbose=verbose,
                    memory=ConversationBufferMemory()
                )
            st.write('Now using model:', model_name)
            logging.info('Now using model:', model_name)
        else:
            if useAll:
                logging.info('Now asking all models')
                for model_name in modelsArray:
                    logging.info('Now asking model:', model_name)
                    try:
                        startTime = time.time()
                        
                        print(f'\n=================== '+ model_name + ' ===================')
                        _, llm = selectModel(model_name)
                        conversation = ConversationChain(
                            llm=llm,
                            verbose=verbose
                        )
                        response = conversation.predict(input=user_input)
                        st.write(response)
                        st.write(f'\nElapsed time:', round((time.time()-startTime)*1000), 'ms')

                    except Exception as e:
                        st.write(f'An error occurred: {str(e)}')
            else:
                logging.info('Now asking model:', model_name)
                startTime=time.time()  
                response = conversation.predict(input=user_input)
                st.write(response)
                st.write(f'\nElapsed time:',round((time.time()-startTime)*1000),'ms')