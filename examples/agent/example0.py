from langchain.agents import load_tools
from langchain.agents import initialize_agent
from langchain.agents import AgentType
from langchain_openai import OpenAI

llm = OpenAI(temperature=0)
# serpapi 负责搜索，llm-math 负责计算20%
tools = load_tools(["serpapi", "llm-math"], llm=llm)
# ZERO_SHOT_REACT_DESCRIPTION 的意思是使用react思维框架、不使用样本
agent = initialize_agent(tools, llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True)
agent.invoke("目前台灣市面上大蒜的平均價格是多少？如果我在此基礎上加價20%賣出，應該如何定價？")
