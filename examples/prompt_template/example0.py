import os
from langchain_openai import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

os.environ["OPENAI_API_KEY"]
llm = OpenAI(temperature=0.9)
prompt = PromptTemplate(
    input_variables=["對象"],
    template="請幫我替 {對象} 取一個好名子。",
)
print(prompt.format(對象="小男孩"))
