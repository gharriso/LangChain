from langchain_openai import ChatOpenAI
from langchain.agents import load_tools, create_react_agent, AgentType
llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)
tools = load_tools(["wikipedia", "llm-math"], llm=llm)
agent = create_react_agent(
    tools, llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True
)
question = """What is the square root of the population of the capital of the
Country where the Olympic Games were held in 2016?"""
agent.invoke(question)