from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAI

WEB_SITE = "https://oghome.com.tw/%E7%9D%A1%E7%9C%A0%E6%BA%AB%E6%BF%95%E5%BA%A6/"

loader = WebBaseLoader(WEB_SITE)

# Document loader
documents = loader.load()

# Split documents
text_splitter = RecursiveCharacterTextSplitter(chunk_size = 1000, chunk_overlap = 200)
docs = text_splitter.split_documents(documents)

print(docs)
