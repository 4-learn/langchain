from langchain_openai import OpenAI
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.chains import SequentialChain

llm = OpenAI()

# LLM Chain 
temperature_template = "人類舒適的氣溫為攝氏 {temperature} 度，用戶所在地區，氣溫為，請您以用者給予的氣溫，回報其舒適程度:"
temperature_prompt = PromptTemplate(input_variables = ["temperature"],
        template = temperature_template)
temperature_chain = LLMChain(llm = llm,
        prompt = temperature_prompt)

# 呼叫
print(temperature_chain.invoke("我在台北，目前氣溫 35 度。"))
