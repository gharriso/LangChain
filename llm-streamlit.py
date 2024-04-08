
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

# Load sample text into a variable from all the files in the sampleText directory
sampleText = ''
for filename in os.listdir('sampleText'):
    with open(f'sampleText/{filename}', 'r') as file:
        sampleText += file.read() + '\n\n'
        

def run_model(llmOption, realModels, user_input):
    print(f"\n\nModel: {llmOption}\n")

    prompt, llm = selectModel(llmOption)
    startTime = time.time()  
    conversation = ConversationChain(
        llm=llm,
        verbose=verbose,
        memory=ConversationBufferMemory()
    )
    print(f'\n\n{prompt}\n{user_input}\n\n')
    response = conversation.predict(input=user_input)
    st.write(response)

    st.write(f'\nElapsed time:', round((time.time()-startTime)*1000), 'ms')
    # Write the prompt and the response to the llm.log file
    with open('llm.log', 'a') as file:
        file.write(f"\n\nModel: {llmOption}\n")
        file.write(f'\n\n{prompt}\n{user_input}\n\n{response}')
    print(f"\n\nModel: {llmOption}\n")
    print(f'\n\n{response}')


st.title('AI Tool')   
# If there is an environment variable LLM_PASSWORD then create a streamlit entry box for the password and don't proceed unless the user types a matching password
if os.environ.get("LLM_PASSWORD"):
    password = st.text_input('Enter Password', type='password')
    if password != os.environ.get("LLM_PASSWORD"):
        st.write('Incorrect password')
        st.stop()
     
col1, col2 = st.columns(2)
llmOption=col1.radio('Select Model', modelNames)
mode=col2.radio('Select Mode', ['question', 'rewrite','critique','jagawag','Harrison Article','fix transcription'])
target=col2.radio('Select Audience', ['technical', 'general'])
user_input = st.text_area('Enter your question or text')
goButton = st.button('go')



if goButton:
    if target == 'technical':
        audience='The text is intended for a reasonably tech-literal general audience and is part of a technical blog or article.'
    elif target == 'general':
        audience='The text is intended for a general audience without a strong technical background.'
        
    if mode == 'question':
        aiPrompt = user_input
    elif mode == 'rewrite':
        aiPrompt="""Please do a rewrite of the following text. """+audience+""" 
         Correct any grammatical errors, and change 
        the phrasing to match the language typical of popular technology articles in mainstream journals 
        such as the new york times.  Feel free to change the wording but please preserve the overall sentence structure. 
        Here's the text: """+user_input
    elif mode == 'Harrison Article':
        aiPrompt="""Write a 300 word article on the following topic, creating output that matches the style of Guy Harrison who writes for database trends and applications.
         Here's the request: """+user_input
    elif mode == 'fix transcription':
        aiPrompt="""I want you to convert the following transcription into a well-written article, suitable for popular technology articles in mainstream journals 
        such as the new york times.  Use the text between the words "START SAMPLE" and "END SAMPLE" as a guide to the writing style. 
        START SAMPLE
        """ + sampleText + """
        END SAMPLE
        Here is the transcription that I want you to convert:
        """+user_input
    elif mode == 'jagawag':
        aiPrompt="""Please do a rewrite of the following text as a catchy, humorous and fun communication from our
        border collie breeding company "Jagawag Kennels".   
        Here's the text: """+user_input
    elif mode == 'critique':
        aiPrompt="""I would like you to critique the following text.  """+audience+""" 
        Comment on technical accuracy, clarity, and style.  Let me know if you think the 
        text is suitable for the target audience.    Finally, provide a minimal rewrite that addresses your key concerns and corrects any grammatical or spelling errors.  
        Here's the text: """+user_input
    if llmOption == 'all':
        for model in realModels:
            st.subheader(f'\nModel: {model}')
            prompt, llm = selectModel(model)
            run_model(model, realModels, aiPrompt)
    else:
        run_model(llmOption, realModels, aiPrompt)

