import os  
import streamlit as st  
from langchain_openai import OpenAI

from langchain.prompts import PromptTemplate  
from langchain.schema import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
 
from langchain_community.llms import Ollama 

if "OPENAI_API_KEY" not in os.environ:
    print("OPENAI_API_KEY not set")
    os._exit(1)

OPENAI_API_KEY=os.environ["OPENAI_API_KEY"]

 
model=OpenAI(openai_api_key=OPENAI_API_KEY,max_tokens=1000)
#model = Ollama(model="llama2")

st.title("Story Creator")  # Text input field for the user to enter a topic  
topic = st.text_input("Choose a topic to create a story about")
 
title_prompt = PromptTemplate.from_template("Provide a single great title for a story about {topic}"  )
story_prompt = PromptTemplate.from_template("""You are a talented writer. Given the title of a story, 
                                            it is your job to write a story for that title.      
                                            Title: {title}""")
title_chain = title_prompt | model | StrOutputParser()  
story_chain = story_prompt | model | StrOutputParser()  
chain = (      {"title": title_chain}      | RunnablePassthrough.assign(story=story_chain)  )


if topic:   
    with st.spinner("Creating story"):
        result = chain.invoke({"topic": topic})          # Displaying the generated title   
        st.write(result)
        st.header(result['title'])       
        st.write(result['story'])
