import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from langchain_community.document_loaders import TextLoader
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import Chroma
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from fastapi.responses import FileResponse

from dotenv import load_dotenv
load_dotenv()

app = FastAPI()

# HTMLファイルから直接APIを叩けるようにCORSを設定
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class QueryRequest(BaseModel):
    question: str

# --- RAGのセットアップ（サーバー起動時に1度だけ実行） ---
# 1. ドキュメントの読み込み
loader = TextLoader("data.txt", encoding="utf-8")
docs = loader.load()

# 2. Embeddingモデルの指定とベクトルデータベース(Chroma)の構築
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
vectorstore = Chroma.from_documents(docs, embeddings)
retriever = vectorstore.as_retriever()

# 3. LLMとプロンプトの設定
llm = ChatOpenAI(model="gpt-4o-mini")

template = """以下のコンテキストを使用して、質問に答えてください。
コンテキストに答えがない場合は、「わかりません」と答えてください。

コンテキスト: {context}

質問: {question}
"""
prompt = ChatPromptTemplate.from_template(template)

# 検索してきた複数のドキュメントのテキストを1つにまとめる関数
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

# 4. LCELを使ったRAGチェーンの作成
rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

# --- APIエンドポイント ---
@app.post("/ask")
async def ask_question(request: QueryRequest):
    # LCELの場合、質問の文字列をそのままinvokeに渡します
    answer = rag_chain.invoke(request.question)
    return {"answer": answer}

# ルートURL ( http://127.0.0.1:8080/ ) にアクセスしたときにHTMLを返す
@app.get("/")
async def serve_html():
    return FileResponse("index.html")