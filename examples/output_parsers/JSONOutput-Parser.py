from langchain.output_parsers.json import SimpleJsonOutputParser
from langchain.prompts import PromptTemplate
from langchain_openai import OpenAI

model = OpenAI(temperature=0.0)

json_prompt = PromptTemplate.from_template(
    "Return a JSON object with an `answer` key that answers the following question: {question}"
)

json_parser = SimpleJsonOutputParser()
json_chain = json_prompt | model | json_parser
print(json_chain.invoke({"question": "Who invented the microscope?"}))
