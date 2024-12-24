from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.llms import OpenAI
from langchain.document_loaders import WikipediaLoader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma

# 文檔加載器
query_wiki = "Artificial Intelligence"
docs = WikipediaLoader(query = query_wiki, lang = "en", load_max_docs = 2).load()

# 分割器
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=100,  # 每個段落的字符數
    chunk_overlap=20,  # 段落間的重疊字符數
)

# 分割文本
split_texts = text_splitter.split_text(docs[0].page_content)

# 列印出分割後的文本段落
for i, chunk in enumerate(split_texts):
    print(f"Chunk {i+1}:\n{chunk}\n")

# 計算 Embeddings
# pip3 install tiktoken

embeddings_model = OpenAIEmbeddings()
embeddings = embeddings_model.embed_documents(docs[0].page_content)

# pip3 install chromadb
db = Chroma.from_documents(docs, embeddings_model)
