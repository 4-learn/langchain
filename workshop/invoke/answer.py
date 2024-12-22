from langchain.chat_models import ChatOpenAI
from langchain.schema import HumanMessage

# 1. 初始化 ChatOpenAI 模型
llm = ChatOpenAI(model="gpt-4", temperature=0.7)

# 2. 問答功能
def ask_question():
    print("請輸入問題，或輸入 'exit' 退出：")
    while True:
        question = input("問題：")
        if question.lower() == "exit":
            print("退出應用程式。")
            break
        # 使用 invoke 方法調用模型
        answer = llm.invoke([HumanMessage(content=question)])
        print(f"回答：{answer.content}")

# 3. 啟動問答應用
ask_question()

