
from models import modelsArray,selectModel
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory

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
        conversation = ConversationChain(
            llm=llm,
            verbose=verbose,
            memory=ConversationBufferMemory()
        )
else:
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
        if model_name == 'all':
            useAll=True
            prompt='all> '
        else:
            useAll=False
            prompt,llm=selectModel(model_name)
            conversation = ConversationChain(
                llm=llm,
                verbose=verbose,
                memory=ConversationBufferMemory()
            )
    else:
        if useAll:
            for model_name in modelsArray:
                try:
                    startTime = time.time()
                    
                    print(f'\n=================== '+ model_name + ' ===================')
                    _, llm = selectModel(model_name)
                    conversation = ConversationChain(
                        llm=llm,
                        verbose=verbose
                    )
                    response = conversation.predict(input=user_input)
                    print(response)
                    print(f'\nElapsed time:', round((time.time()-startTime)*1000), 'ms')

                except Exception as e:
                    print(f'An error occurred: {str(e)}')
        else:
            startTime=time.time()  
            response = conversation.predict(input=user_input)
            print(response)
            print(f'\nElapsed time:',round((time.time()-startTime)*1000),'ms')