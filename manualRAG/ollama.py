from langchain.chains import ConversationChain

from langchain_community.llms import Ollama 

 

llm = Ollama(model="phi")

 
 

#llm = Ollama(server="http://localhost:11434", model="llama2")
 
chatbot = ConversationChain(llm=llm , verbose=False)
while True:
    human=input('Human: ')
    print()
    if not human:
        break
    aiOutput=chatbot.predict(input=human)
    print(aiOutput) 
 