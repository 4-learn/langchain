import os
from langchain_openai import OpenAI

# os.environ["OPENAI_API_KEY"]

llm = OpenAI(temperature = 0.9)
text = "What would be a good company name for a company that makes colorful socks?"
print(llm.invoke(text))
