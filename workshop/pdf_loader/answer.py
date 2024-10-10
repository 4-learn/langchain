from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from collections import Counter

# PDF 文件路徑
PDF_PATH = "./InnovationValue.pdf"

# 使用 PyPDFLoader 加載文檔
loader = PyPDFLoader(PDF_PATH)
documents = loader.load()

# 使用 RecursiveCharacterTextSplitter 進行文本分割
splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
chunks = splitter.split_documents(documents)

# 輸出分割後的文本塊數量
print(f"分割後的文本塊數量: {len(chunks)}")

# 合併所有文本塊內容
all_text = " ".join(chunk.page_content for chunk in chunks)

# 分割文本並計算詞頻，過濾掉短於 4 個字符的詞彙
words = all_text.split()
filtered_words = [word for word in words if len(word) >= 4]
word_counter = Counter(filtered_words)

# 找出出現頻率最高的前 10 個詞彙
most_common_words = word_counter.most_common(10)
for word, count in most_common_words:
    print(f"詞彙: '{word}', 出現次數: {count}")

