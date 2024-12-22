from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.llms import OpenAI
from langchain.document_loaders import WikipediaLoader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma

# 文檔加載器
query_wiki = "Artificial Intelligence"
docs = WikipediaLoader(query = query_wiki, lang = "en", load_max_docs = 2).load()

print(docs[0].metadata)
print(docs[0].page_content[:400])
