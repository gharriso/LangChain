
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
        

def run_model(llmOption, realModels, user_input, temperature=0.5,max_tokens=1000):
    print(f"\n\nModel: {llmOption}\n")

    prompt, llm = selectModel(llmOption)
    startTime = time.time()  
    #if the llmOption text includes gemini or claude3  we can't pass args 
    if 'gemini' in llmOption or 'claude3' in llmOption:
        llm_kwargs = {}
    else:
        llm_kwargs = {"temperature": temperature,"max_tokens": max_tokens}
 
 
    conversation = ConversationChain(
        llm=llm,
        llm_kwargs=llm_kwargs,
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
temperature=col1.slider('Select Temperature', 0.0, 2.0, 0.7)

mode=col2.radio('Select Mode', ['question', 'fix transcription','book section','sidebar','CopyEdit','glossary',
                                'rewrite','critique',
                                'Harrison Article','transcribe article', ])
target=col2.radio('Select Audience', [ 'general','technical',])
max_tokens=col1.slider('Select Max Tokens', 0, 16000, 1000)

user_input = st.text_area('Enter your question or text')
goButton = st.button('go')



if goButton:
    if target == 'technical':
        audience='The text is intended for a reasonably tech-literal general audience and is part of a technical blog or article.'
    elif target == 'general':
        audience='The text is intended for a general audience without a strong technical background.'
        
    if mode == 'question':
        aiPrompt = user_input
    elif mode == 'book section':
        aiPrompt="""Please write a section for my book 'Quantum Computing, AI and Blockchain: What you need to know about the technologies changing our world'
        """+audience+""" 
         The section should be between 500-1000 words long and should provide a high-level overview of the topic. 
          Here's the topic and some notes to work from: """+user_input
    elif mode == 'sidebar':
        aiPrompt="""Please write a sidebar for my book 'Quantum Computing, AI and Blockchain: What you need to know about the technologies changing our world'
        """+audience+""" 
         The section should be 2 or three paragraphs long and should provide a high-level overview of the topic. 
         Here's the topic and some notes to work from: """+user_input
    elif mode == 'glossary':
        aiPrompt="""Please write a glossary for my book 'Quantum Computing, AI and Blockchain: What you need to know about the technologies changing our world'
        """+audience+""" 
         The description should be one or two sentences long.
        Here's the glossary entry: """+user_input
    elif mode == 'rewrite':
        aiPrompt="""Please do a rewrite of the following text. """+audience+""" 
         Correct any grammatical errors, and if neccessary do minor rewrites to improve clarity.   
         You can change the wording slightly but please preserve the overall sentence structure. 
        Here's the text: """+user_input
    elif mode == 'CopyEdit':
        aiPrompt="""You copyediting my book 'AI, Quantum Computing and Blockchain'. """+audience+""" 
         Correct any grammatical or spelling errors, and if neccessary do some rewrites to improve clarity 
         or to eliminate redudancy or contradictions.   
         You can change the wording but please preserve the overall sentence structure and maintain 
         the writing style and meaning. 
         
         End the rewrite with "============================\n".
         After the rewrite, warn me if you think there are any factual errors in the text.
         Also list the changes you have made in the format of the UNIX diff command.
        Here's the text: """+user_input
    elif mode == 'Harrison Article':
        aiPrompt="""Write a 300 word article on the following topic, creating output that matches the style of Guy Harrison who writes for database trends and applications.
         Here's the request: """+user_input
    elif mode == 'transcribe article':
        aiPrompt="""I want you to convert the following transcription into a well-written article, suitable for popular technology articles in mainstream journals 
        such as the new york times.  Use the text between the words "START SAMPLE" and "END SAMPLE" as a guide to the writing style. 
        START SAMPLE
        """ + sampleText + """
        END SAMPLE
        Here is the transcription that I want you to convert:
        """+user_input
    elif mode == 'fix transcription':
        aiPrompt="""The following transcription was created from a voice recording.  Correct any obvious errors in the transcription, fix any grammatical errors, and
        make the text more readable.  The text should be suitable for popular technology articles in mainstream journals.  Don't add any new information to the text.
        
        After the corrected transcription, warn me if you think there are any factual errors in the text. 
        
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
 
            try:
                run_model(model, realModels, aiPrompt, temperature)
            except Exception as e:
                print(f"An error occurred: {e}")

    else:
        run_model(llmOption, realModels, aiPrompt, temperature,max_tokens)

