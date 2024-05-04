import os
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.retrieval import create_retrieval_chain
from langchain_community.document_loaders import WeatherDataLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAI
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate

llm = OpenAI()
OPENAI_EMBEDDING_DEPLOYMENT_NAME = "text-embedding-3-small"
embeddings = OpenAIEmbeddings(deployment=OPENAI_EMBEDDING_DEPLOYMENT_NAME,
                              chunk_size=1)

loader = WeatherDataLoader.from_params(
    ["Nantou"], openweathermap_api_key = os.getenv("OPENWEATHERMAP_API_KEY")
)
documents = loader.load()

print("DEBUG: content after document loader : " + str(documents))

text_splitter = RecursiveCharacterTextSplitter()
docs = text_splitter.split_documents(documents)

vector = FAISS.from_documents(docs, embeddings)
retriever = vector.as_retriever()

context = []
prompt = ChatPromptTemplate.from_messages([
    ('system', '請以中文回答，使用者所在地的氣象資訊\'s :\n\n{context}'),
    ('user', '問題: {input}'),
])

document_chain = create_stuff_documents_chain(llm, prompt)

retrieval_chain = create_retrieval_chain(retriever, document_chain)

response = retrieval_chain.invoke({
        'input': "南投縣",
        'context': context
    })

print("DEBUG: response of llm invoke: " + str(response))