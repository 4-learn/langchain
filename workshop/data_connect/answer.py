from langchain.document_loaders import WikipediaLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma

# 1. 加載 Wikipedia 資料
query_wiki = "Machine Learning"
docs = WikipediaLoader(query=query_wiki, lang="en", load_max_docs=1).load()

# 2. 分割文本
text_splitter = RecursiveCharacterTextSplitter(chunk_size=100, chunk_overlap=20)
split_texts = text_splitter.split_text(docs[0].page_content)

# 3. 嵌入向量
embeddings_model = OpenAIEmbeddings()
db = Chroma.from_texts(split_texts, embeddings_model)

# 4. 檢索功能
def retrieve_answer(question):
    results = db.similarity_search(question, k=2)  # 返回最相關的兩個段落
    print("相關段落：")
    for result in results:
        print(result.page_content)
        print("-" * 50)

# 測試檢索
query = "What is the history of Machine Learning?"
retrieve_answer(query)
