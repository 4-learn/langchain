from langchain_openai import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.prompts.few_shot import FewShotPromptTemplate

llm = OpenAI()

## PromptTemplate 的用法
prompt = PromptTemplate(
    input_variables=["equation"],
    template="請計算以下數學問題的答案: {equation}",
)
print(llm.invoke(prompt.format(equation="8 * 12")))

## 改為 FewShotPromptTemplate
examples = [
  {
    "question": "5+3",
    "answer": "8"
  },
  {
    "question": "10-2",
    "answer": "8"
  },
  {
    "question": "4*2",
    "answer": "8"
  },

]

example_prompt = PromptTemplate.from_template(
    template = "Question: {question}\n{answer}",
)
prompt = FewShotPromptTemplate(
    examples = examples,
    example_prompt = example_prompt,
    suffix = "Question: {input}",
    input_variables=["input"],
)
print(llm.invoke(prompt.format(input="8 * 12")))
