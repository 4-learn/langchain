from langchain_community.llms import HuggingFaceEndpoint

hub_llm = HuggingFaceEndpoint(
    repo_id="google/gemma-2b-it", temperature=0.1
)

response = hub_llm.invoke("Hello, what's ur name?")
print(response)
