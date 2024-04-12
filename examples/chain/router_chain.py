from langchain.chains.router import MultiPromptChain
from langchain_openai import OpenAI
from langchain.chains import ConversationChain
from langchain.chains.llm import LLMChain
from langchain.prompts import PromptTemplate
from langchain.chains.router.llm_router import LLMRouterChain, RouterOutputParser
from langchain.chains.router.multi_prompt_prompt import MULTI_PROMPT_ROUTER_TEMPLATE

lex_teaching_template = """您是詞彙教學的專家。\
您精通各種語言的詞彙，並能夠提供詳細的解釋和背景知識。\
當學生想知道某個單詞的意思或用法時，您總是能夠給予清晰且有趣的答案。

這裡有一個問題：
{input}"""


sentence_teaching_template = """您擅長例句教學。\
您知道如何使用例句來解釋語言中的規則和結構。\
當學生對於如何在實際情境中使用某個語句或表達有疑問時，您能夠提供生動且實用的例子。

這裡有一個問題：
{input}"""

prompt_infos = [
    {
        "name": "lex_teaching", # 執行鏈名稱
        "description": "適合回答與詞彙學習相關的問題", # 執行鏈簡單描述
        "prompt_template": lex_teaching_template, # 實際任務的提示樣板
    },
    {
        "name": "sentence_teaching", # 執行鏈名稱
        "description": "適合回答與例句使用和學習相關的問題", # 執行鏈簡單描述
        "prompt_template": sentence_teaching_template, # 實際任務的提示樣板
    },
]

llm_chat = OpenAI()

# 建立任務的執行鏈實例，以及「名稱-實例對照表」
destination_chains = {}
for p_info in prompt_infos:
    name = p_info["name"]
    prompt_template = p_info["prompt_template"]
    prompt = PromptTemplate(template=prompt_template, input_variables=["input"])
    chain = LLMChain(llm=llm_chat, prompt=prompt)
    destination_chains[name] = chain

# 建立預設執行鏈
default_chain = ConversationChain(llm=llm_chat, output_key="text")

# 使用任務基本資訊以及 MULTI_PROMPT_ROUTER_TEMPLATE 建立 LLMRouterChain 實例
destinations = [f"{p['name']}: {p['description']}" for p in prompt_infos]
destinations_str = "\n".join(destinations)

# 這裏使用 MULTI_PROMPT_ROUTER_TEMPLATE 格式化我們的提示訊息
router_template = MULTI_PROMPT_ROUTER_TEMPLATE.format(destinations=destinations_str)

# 建立實例
router_prompt = PromptTemplate(
    template=router_template,
    input_variables=["input"],
    output_parser=RouterOutputParser(),
)
router_chain = LLMRouterChain.from_llm(llm_chat, router_prompt)
chain = MultiPromptChain(
    router_chain=router_chain,
    destination_chains=destination_chains,
    default_chain=default_chain,
    verbose=True,
)
print(chain.invoke("coding 這字什麼意思？"))
print(chain.invoke("幫我解釋「I like to code」這句話"))
