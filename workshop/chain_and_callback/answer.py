from langchain_openai import OpenAI
from langchain.chains import LLMChain, MultiPromptChain, ConversationChain
from langchain.prompts import PromptTemplate
from langchain.chains.router.llm_router import LLMRouterChain, RouterOutputParser
from langchain.chains.router.multi_prompt_prompt import MULTI_PROMPT_ROUTER_TEMPLATE
from weather import get_weather_from_opendata

# 初始化 Langchain LLM
llm_chat = OpenAI()

""" def get_weather_from_opendata(location):
    return {
        "temperature": 26,
        "humidity": 60
    } """

# 定義冷氣和除濕機的控制函數
def air_conditioner_control(input_temperature):
    if input_temperature > 26:
        print("callback: 開啟冷氣機")
        return "開啟冷氣機"
    else:
        print("callback: 關閉冷氣機")
        return "關閉冷氣機"

def dehumidifier_control(input_humidity):
    if input_humidity > 60:
        print("callback: 開啟除濕機")
        return "開啟除濕機"
    else:
        print("callback: 關閉除濕機")
        return "關閉除濕機"

def callbacks(msg, value):
    # 調用 Langchain 來判斷應該使用哪個回調函數
    # llm_chat2 = OpenAI()
    response_text = llm_chat.invoke(f"""請依照問題，選擇回答: "冷氣" 或 "除濕機" 兩個答案選擇一個。如果都沒有符合，請回答 None, 問題: {msg}""")

    # remove newline
    response_text = response_text.replace("\n", "")

    if response_text == "冷氣":
        return air_conditioner_control(value)
    elif response_text == "除濕機":
        return dehumidifier_control(value)
    else:
        return "No suitable action was found based on the response."

llm_chat = OpenAI()

# 定義控制冷氣的模板
ac_control_template = """您是智能家居的管家。當用戶提供當前溫度時，您需要根據這個溫度判斷是否需要開啟冷氣。
當前溫度：{input} 度。請問是否需要開啟冷氣？
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
destinations_str = "ac_control: 根據溫度控制冷氣\n" \
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

# 測試函數
location = "C0I110"  # 站點 ID
temperature, humidity = get_weather_from_opendata(location)
msg = chain.invoke(f"現在的溫度是 {temperature}°C")
callbacks(msg, temperature)
msg = chain.invoke(f"現在的濕度是 {humidity}%")
callbacks(msg, humidity)