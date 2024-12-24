from langchain_openai import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

prompt = PromptTemplate(
    input_variables=["對象"],
    template="請幫我替 {對象} 取一個好名子。",
)
print(prompt.format(對象="小男孩"))

llm = OpenAI(temperature=0.7)
print(llm.invoke(prompt.format(對象="小男孩")))
