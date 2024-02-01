from langchain_openai import ChatOpenAI
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
template = """Question: {question}
Let's think step by step.
Answer: """
prompt = PromptTemplate(template=template, input_variables=["question"])
llm = ChatOpenAI(model_name="gpt-3.5-turbo")
llm_chain = LLMChain(prompt=prompt, llm=llm)
question = """ What is the population of the capital of the country where the
Olympic Games were held in 2016? """
output=llm_chain.invoke(question)
print(output )