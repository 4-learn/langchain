from langchain_openai import OpenAI
from langchain.chains import LLMChain, MultiPromptChain, ConversationChain
from langchain.prompts import PromptTemplate
from langchain.chains.router.llm_router import LLMRouterChain, RouterOutputParser
from langchain.chains.router.multi_prompt_prompt import MULTI_PROMPT_ROUTER_TEMPLATE

llm_chat = OpenAI()

# 定義控制空調的模板
ac_control_template = """您是智能家居的管家。當用戶提供當前溫度時，您需要根據這個溫度判斷是否需要開啟空調。
當前溫度：{input} 度。請問是否需要開啟空調？
"""

# 定義控制除濕機的模板
dehumidifier_control_template = """您是智能家居的管家。當用戶提供當前濕度時，您需要根據這個濕度判斷是否需要開啟除濕機。
當前濕度：{input}%。請問是否需要開啟除濕機？
"""

# 建立執行鏈實例
ac_chain = LLMChain(llm=llm_chat, prompt=PromptTemplate(template=ac_control_template, input_variables=["input"]))
dehumidifier_chain = LLMChain(llm=llm_chat, prompt=PromptTemplate(template=dehumidifier_control_template, input_variables=["input"]))

# 建立預設執行鏈
default_chain = ConversationChain(llm=llm_chat, output_key="text")

# 使用 MULTI_PROMPT_ROUTER_TEMPLATE 建立路由提示訊息
destinations_str = "ac_control: 根據溫度控制空調\n" \
                   "dehumidifier_control: 根據濕度控制除濕機"

router_template = MULTI_PROMPT_ROUTER_TEMPLATE.format(destinations=destinations_str)
router_prompt = PromptTemplate(template=router_template, input_variables=["input"], output_parser=RouterOutputParser())

# 建立路由器鏈
router_chain = LLMRouterChain.from_llm(llm_chat, router_prompt)

# 組合所有鏈到 MultiPromptChain
chain = MultiPromptChain(
    router_chain=router_chain,
    destination_chains={
        "ac_control": ac_chain,
        "dehumidifier_control": dehumidifier_chain
    },
    default_chain=default_chain,
    verbose=True
)

# 測試用例
print(chain.invoke("現在的溫度是 35"))
print(chain.invoke("現在的濕度是 60"))

