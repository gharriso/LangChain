
from models import modelsArray,selectModel
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
if os.environ.get("DEBUG", "").lower() == "true":
    debug=True
    verbose=True
    logging.getLogger().setLevel(logging.DEBUG)
    
# If the user specified a model on the command line, use that model
if len(sys.argv) > 1:
    model_name = sys.argv[1]
    if model_name == 'all':
        useAll=True
        prompt='all> '
    else:
        prompt,llm=selectModel(model_name)
    prompt,llm=selectModel("gpt3")
    
# return an array of keys of the models
modelNames = list(modelsArray.keys())
realModels=modelNames.copy()
# Add "all" to the array
modelNames.append('all')

def run_model(llmOption, realModels, user_input):
    prompt, llm = selectModel(llmOption)
    startTime = time.time()  
    conversation = ConversationChain(
        llm=llm,
        verbose=verbose,
        memory=ConversationBufferMemory()
    )
    response = conversation.predict(input=user_input)
    st.write(response)
    st.write(f'\nElapsed time:', round((time.time()-startTime)*1000), 'ms')

st.title('AI Tool')        
col1, col2 = st.columns(2)
llmOption=col1.radio('Select Model', modelNames)
mode=col2.radio('Select Mode', ['question', 'rewrite'])
user_input=st.text_input('Enter your question or text')
goButton = st.button('go')

if goButton:
    if llmOption == 'all':
        for model in realModels:
            st.write(f'\nModel: {model}')
            prompt, llm = selectModel(model)
            run_model(model, realModels, user_input)
    else:
        run_model(llmOption, realModels, user_input)

