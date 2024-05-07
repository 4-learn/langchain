from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.retrieval import create_retrieval_chain
from langchain_community.document_loaders import WikipediaLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAI
from langchain_openai import OpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate

llm = OpenAI()
OPENAI_EMBEDDING_DEPLOYMENT_NAME = "text-embedding-3-small"

TOPIC = "保安宮"
loader = WikipediaLoader(query = TOPIC, lang="zh", load_max_docs = 2)

# Document loader
documents = loader.load()

# Split documents
text_splitter = RecursiveCharacterTextSplitter(chunk_size = 1000, chunk_overlap = 200)
docs = text_splitter.split_documents(documents)

# Get embeddings
embeddings = OpenAIEmbeddings(deployment = OPENAI_EMBEDDING_DEPLOYMENT_NAME, chunk_size = 1)
vector = FAISS.from_documents(docs, embeddings)
retriever = vector.as_retriever()

context = []
prompt = ChatPromptTemplate.from_messages([
    ('system', '請以中文回答，這間廟的主祀神\'s :\n\n{context}'),
    ('user', '你知道這間廟嗎？ {input}'),
])

document_chain = create_stuff_documents_chain(llm, prompt)
retrieval_chain = create_retrieval_chain(retriever, document_chain)

response = retrieval_chain.invoke({
    'input': "南廠保安宮",
    'context': context
})

print("DEBUG: response of llm invoke: " + str(response))