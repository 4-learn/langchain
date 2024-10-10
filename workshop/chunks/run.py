from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from collections import Counter

# 網頁 URL
WEB_SITE = "https://zh.wikipedia.org/w/index.php?title=%E4%BA%BA%E5%B7%A5%E7%A5%9E%E7%BB%8F%E7%BD%91%E7%BB%9C"

# 使用 WebBaseLoader 加載文檔
loader = WebBaseLoader(WEB_SITE)
documents = loader.load()

# 使用 RecursiveCharacterTextSplitter 進行文本分割
splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
chunks = splitter.split_documents(documents)

# 輸出分割後的文本塊數量
print(f"分割後的文本塊數量: {len(chunks)}")

# 列出每個文本塊的內容
for idx, chunk in enumerate(chunks, 1):
    print(f"文本塊 {idx}: {chunk.page_content}")

# 計算所有文本塊中出現頻率最高的前三個詞彙
word_counter = Counter()
for chunk in chunks:
    words = chunk.page_content.split()
    word_counter.update(words)

most_common_words = word_counter.most_common(10)
for word, count in most_common_words:
    print(f"詞彙: '{word}', 出現次數: {count}")

