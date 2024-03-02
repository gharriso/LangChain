import os  
import streamlit as st  
from langchain_openai import OpenAI

from langchain.prompts import PromptTemplate  
from langchain.schema import StrOutputParser

if "OPENAI_API_KEY" not in os.environ:
    print("OPENAI_API_KEY not set")
    os._exit(1)

OPENAI_API_KEY=os.environ["OPENAI_API_KEY"]

 
model=OpenAI(openai_api_key=OPENAI_API_KEY,max_tokens=1000)

st.title("Story Creator")  # Text input field for the user to enter a topic  
topic = st.text_input("Choose a topic to create a story about")
 
prompt = PromptTemplate.from_template("Write a great title for a story about {topic}"  )

if topic:   
    with st.spinner("Creating story title"):
        chain = prompt | model | StrOutputParser()   
        response = chain.invoke({"topic": topic})          # Displaying the generated title          
        st.write(response)
