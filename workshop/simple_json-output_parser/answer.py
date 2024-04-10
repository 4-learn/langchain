from langchain.output_parsers.json import SimpleJsonOutputParser
from langchain.prompts import PromptTemplate
from langchain_openai import OpenAI

# 初始化 OpenAI 語言模型，並設定溫度參數為 0 以獲得更穩定的回答
model = OpenAI(temperature=0.0)

# 定義 prompt template，要求模型根據自然語言指令返回 JSON 格式的裝置控制信息
json_prompt = PromptTemplate.from_template(
    "根據以下的自然語言指令：`{command}`，"
    "返回一個包含 `device` 和 `action` 鍵的 JSON 對象，用以控制 IoT 裝置。"
)

# 創建一個 SimpleJsonOutputParser 用於將語言模型的文本輸出轉換為 JSON 對象
json_parser = SimpleJsonOutputParser()

# 組合 PromptTemplate 和 OpenAI 模型，並將輸出通過 SimpleJsonOutputParser 解析
json_chain = json_prompt | model | json_parser

# 定義用戶的自然語言控制指令
command = "請把客廳的燈開啟。"

# 使用格式化的 prompt 調用 langchain 鏈，傳入控制指令
response = json_chain.invoke({"command": command})

# 輸出解析後的 JSON 對象，該對象包含了設備名稱和控制動作
print(response)
