from langchain.llms import HuggingFaceEndpoint
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate

# 初始化 HuggingFaceEndpoint 實例
# 注意，這裏直接使用 repo_id 和可能的其他參數，但不包括 max_new_tokens
# 因為 max_new_tokens 需要在發送請求時設置，而不是在這裏
hub_llm = HuggingFaceEndpoint(
    repo_id="google/gemma-2b-it", max_length=1024, temperature=0.1
)

prompt = PromptTemplate(
    input_variables=["profession"],
    template="你有一個職業，您是 {profession} 而您講話不諷刺"
)

# 這裏假設 LLMChain 或 HuggingFaceEndpoint 支持在 invoke 或 generate 時設置 max_new_tokens
# 如果不支持，您可能需要修改 HuggingFaceEndpoint 來接受此參數
hub_chain = LLMChain(prompt=prompt, llm=hub_llm, verbose=True)

# 在調用模型時明確指定 max_new_tokens 參數
# 注意：這裏的做法依賴於 HuggingFaceEndpoint 和 LLMChain 的實現支持這樣的參數傳遞
# 如果原始實現不支持，您可能需要自行擴展或修改相關類的功能
response = hub_chain.invoke("customer service agent", max_new_tokens=200)

print(response)
