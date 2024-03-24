import os
from langchain_openai import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
os.environ["OPENAI_API_KEY"]
llm = OpenAI(temperature=0.9)

#text = "What would be a good company name for a company that makes colorful socks?"
#print(llm(text))

prompt = PromptTemplate(
    input_variables=["對象"],
    template="請幫我替 {對象} 取一個好名子。",
)
chain = LLMChain(llm=llm, prompt=prompt)
print(chain.invoke("小男孩"))
