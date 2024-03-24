import os
from langchain_openai import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

# 設定環境變量以存取 OpenAI API
os.environ["OPENAI_API_KEY"]

# 創建一個 OpenAI LLM 實例
llm = OpenAI(temperature=0.9)

# 定義一個 PromptTemplate，用於生成問候語
prompt = PromptTemplate(
    input_variables=["title"],
    template="尊貴的 {title} 歡迎來到智能科技有限公司！我是您的虛擬助手，有什麼可以幫您的嗎？",
)

# 創建一個 LLMChain
chain = LLMChain(llm=llm, prompt=prompt)

# 定義一個函數來生成問候語
def generate_greeting(title):
    return chain.invoke(title)

# 使用範例
greeting_f = generate_greeting("阿土")

print(greeting_f)
