from langchain.chains import ConversationChain
from langchain_openai import ChatOpenAI
import os

OPENAI_API_KEY=os.environ["OPENAI_API_KEY"]
chatbot_llm=ChatOpenAI(openai_api_key=OPENAI_API_KEY,model_name="gpt-4",max_tokens=4000)

chatbot = ConversationChain(llm=chatbot_llm , verbose=False)
with open('newEmail.txt', 'r') as f:
    new_email = f.read()
question='Does the following email text appear to be spam: '+new_email+''
aiOutput=chatbot.predict(input=question)
print(aiOutput) 
 