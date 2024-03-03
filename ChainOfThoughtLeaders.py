
from langchain.chains import ConversationChain
from langchain_openai import ChatOpenAI
from langchain.memory import ConversationBufferMemory
import os

if "OPENAI_API_KEY" not in os.environ:
    print("OPENAI_API_KEY not set")
    os._exit(1)
OPENAI_API_KEY=os.environ["OPENAI_API_KEY"]
gpt3=ChatOpenAI(openai_api_key=OPENAI_API_KEY,model_name='gpt-3.5-turbo-16k',max_tokens=4000)
gpt4=ChatOpenAI(openai_api_key=OPENAI_API_KEY,model_name="gpt-4",max_tokens=4000)
llm = gpt4

conversation = ConversationChain(
    llm=llm,
    verbose=True,
    memory=ConversationBufferMemory()
)

topic = input("Enter the topic: ")

inputs = [
    f"""Identify the top books by thought leaders in the topic of {topic}""",
    f"""Provide a comprehensive summary list of the key ideas from each of these books relevant to the topic of{topic}. Aim for 5 bullet points for each book.""",
    f"""Synthesize the ideas above into actionable insights related to the topic of {topic}, focusing on their implications and applications.""",
    f"""Form a cohesive straightforward narrative that integrates these insights into a comprehensive overview of the topic of {topic}, addressing its key aspects and implications.""",
    f"""Refine the narrative to ensure it is clear and comprehensive. Provide a summary that encapsulates the essential insights and perspectives on the topic of {topic}."""
]

for input in inputs:
    output = conversation.predict(input=input)
    print(output)
