from langchain.chains import ConversationChain
from langchain_openai import OpenAI
chatbot_llm = OpenAI(model_name='gpt-3.5-turbo-instruct')
chatbot = ConversationChain(llm=chatbot_llm , verbose=False)
while True:
    human=input('ask a question: ')
    if not human:
        break
    aiOutput=chatbot.predict(input=human)
    print(aiOutput) 
 