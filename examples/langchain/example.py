from langchain_openai import ChatOpenAI

llm = ChatOpenAI()
print(llm.invoke("how can langsmith help with testing?"))
