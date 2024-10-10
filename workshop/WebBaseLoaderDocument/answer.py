from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.retrieval import create_retrieval_chain
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings  # 使用 Hugging Face 嵌入
from langchain_core.prompts import ChatPromptTemplate

WEB_SITE = "https://oghome.com.tw/%E7%9D%A1%E7%9C%A0%E6%BA%AB%E6%BF%95%E5%BA%A6/"

# 初始化 Hugging Face 嵌入模型
embedding_model = HuggingFaceEmbeddings(model_name="distilbert-base-uncased")

loader = WebBaseLoader(WEB_SITE)

# Document loader
documents = loader.load()

# print
print(documents)
