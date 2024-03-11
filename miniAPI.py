import os
import configparser

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from datetime import datetime
import urllib.parse
import json

from redisConnector import RedisConnector
from mongoConnector import MongoConnector

import openai

from pydantic import BaseModel

from sentence_transformers import SentenceTransformer
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import TextLoader
from langchain.embeddings.sentence_transformer import SentenceTransformerEmbeddings
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma

# 项目配置
config = configparser.ConfigParser()
config.read('config.ini')
embedding_path = config.get('Settings', 'embedding_path')
redis_server = config.get('Settings', 'redis_server')
redis_port = config.get('Settings', 'redis_port')
mongo_server = config.get('Settings', 'mongo_server')
mongo_port = config.get('Settings', 'mongo_port')
mongo_username = config.get('Settings', 'mongo_username')
mongo_password = config.get('Settings', 'mongo_password')


def generate_timestamped_filename(original_filename):
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    base_name, extension = os.path.splitext(original_filename)
    encoded_filename = urllib.parse.quote_plus(base_name)  # 对文件名进行URL编码
    return f"{encoded_filename}_{timestamp}{extension}"

def decode_filename(encoded_filename):
    return urllib.parse.unquote_plus(encoded_filename)

def createDBFrom(filename):
    base_name, extension = os.path.splitext(filename)
    persist_directory = f"db/{base_name}_{extension[1::]}"
    loader = TextLoader(f"files/{filename}")
    documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=100)
    docs = text_splitter.split_documents(documents)
    
    #embedding_function = SentenceTransformerEmbeddings(model_name="paraphrase-MiniLM-L6-v2")
    embedding_function = HuggingFaceEmbeddings(model_name=embedding_path)
    db = Chroma.from_documents(documents=docs, embedding=embedding_function, persist_directory=persist_directory)
    db.persist()
    return f"{base_name}_{extension[1::]}"

def loadDB(dbname):
    if not os.path.exists(f"db/{dbname}"):
        return None
    embedding_function = HuggingFaceEmbeddings(model_name=embedding_path)
    db = Chroma(persist_directory=f"db/{dbname}", embedding_function=embedding_function)
    return db

def parseTemplate(template, vars):
    if template["vars"] != list(vars.keys()):
        return None
    
    for i in range(len(template["template"])):
        template["template"][i]["content"] = template["template"][i]["content"].format(**vars)
    
    return template


def makeRAGRes(message, retriever, client, history, user, cId):
    mId = redis_connector.addMessage(user, cId, message, "user", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    try:
        docs = retriever.similarity_search(message) 
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            stream=True,
            messages=history+[{"role": "user", "content": f"""
            请参考以下信息回答{message}：
            {'/n/n'.join([doc.page_content for doc in docs])}
            """}],
        )
        ans = ""
        for chunk in response:
            if chunk.choices[0].delta.content is not None:
                ans+=chunk.choices[0].delta.content
                yield chunk.choices[0].delta.content
        redis_connector.addMessage(user, cId, ans, "assistant", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
        #return response
    except Exception as e:
        redis_connector.delMessage(cId, mId)
        yield f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} - failed to connect"
        raise HTTPException(status_code=500, detail=str(e))

def makeRes(message, client, history, user, cId):
    mId = redis_connector.addMessage(user, cId, message, "user", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            stream=True,
            messages=history
        )
        ans = ""
        for chunk in response:
            if chunk.choices[0].delta.content is not None:
                ans+=chunk.choices[0].delta.content
                yield chunk.choices[0].delta.content
        redis_connector.addMessage(user, cId, ans, "assistant", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    except Exception as e:
        redis_connector.delMessage(cId, mId)
        yield f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} - failed to connect"
        raise HTTPException(status_code=500, detail=str(e))
    # return response


app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 测试
@app.get("/api/v1")
def hello():
    return "hello"

# 上传文档生成向量数据库
@app.post("/api/v1/upload/db")
async def upload_file(file: UploadFile = File(...)):
    new_filename = generate_timestamped_filename(file.filename)
    with open("files/"+new_filename, "wb") as f:
        f.write(file.file.read())
        f.close()
    with open("files/"+new_filename, "rb") as f:
        db_name = createDBFrom(new_filename)
    return {"filename": db_name}

# 列出所有向量数据库
@app.get("/api/v1/db")
def get_dbs():
    return os.listdir("db")

# 测试数据库
class TestDB(BaseModel):
    name: str
    query: str

@app.post("/api/v1/db/test")
def test_db(testDB: TestDB):
    if testDB.name not in [retriever["name"] for retriever in retrievers]:
        db = loadDB(testDB.name)
        if db is None:
            return {"error": "db not found"}
        else:
            retrievers.append({"name": testDB.name, "retriever": db})
    else:
        db = [retriever["retriever"] for retriever in retrievers if retriever["name"] == testDB.name][0]
    result = db.similarity_search(testDB.query)
    return result

# 上传模板
class TemplateUpload(BaseModel):
    name: str
    template: dict

@app.post("/api/v1/upload/template")
async def upload_template(template: TemplateUpload):
    with open(f"templates/{template.name}.json", "w") as f:
        f.write(json.dumps(template.template, indent=4, ensure_ascii=False))
    return {"template_name": template.name}

# 列出所有模板
@app.get("/api/v1/template")
def get_templates():
    templates = os.listdir("templates")
    return [os.path.splitext(t)[0] for t in templates]

# 获取模板
@app.get("/api/v1/template/{template_name}")
def get_template(template_name: str):
    if not os.path.exists(f"templates/{template_name}.json"):
        return {"error": "template not found"}
    with open(f"templates/{template_name}.json", "r") as f:
        return json.loads(f.read())

# 对话接口
# 每次请求携带以下参数
class ChatRequest(BaseModel):
    user: str
    base_url: str
    api_key: str
    message: str
    retriever: str
    cId: str

@app.post("/api/v1/chat")
def chat(chatRequest: ChatRequest):
    message = chatRequest.message
    client = openai.OpenAI(base_url=chatRequest.base_url, api_key=chatRequest.api_key)
    
    history = []
    for mId in redis_connector.getMessages(chatRequest.cId):
        m = redis_connector.getMessage(mId)
        history.append({"role": m["role"], "content": m["content"]})
    print(history)
    
    if chatRequest.retriever!="":
        if chatRequest.retriever not in [retriever["name"] for retriever in retrievers]:
            db = loadDB(chatRequest.retriever)
            if db is None:
                return {"error": "db not found"}
            else:
                retrievers.append({"name": chatRequest.retriever, "retriever": db})
        else:
            db = [retriever["retriever"] for retriever in retrievers if retriever["name"] == chatRequest.retriever][0]
            
        return StreamingResponse(makeRAGRes(message, db, client, history, chatRequest.user, chatRequest.cId), media_type="text/plain")
        #return makeRAGRes(message, db, client, history)
    else:
        return StreamingResponse(makeRes(message, client, history, chatRequest.user, chatRequest.cId), media_type="text/plain")
        #return makeRes(message, client, history)
        

@app.post("/api/v1/addConversation")
def addConversation(data:dict):
    cId = redis_connector.addConversation(data["user"])
    if (data["persona"]==""):
        data["persona"] = "你是一名乐于助人的的ai助手，请帮助我完成我的需求，请一定不要回答我的问题以外的内容。"
    #connector.addMessage(data["user"], cId, "你是一名乐于助人的的ai助手，请帮助我完成我的需求，请一定不要回答我的问题以外的内容。", "system", datetime.now().strftime("%Y-%m-%d %H:%M:%S"), True)
    redis_connector.addMessage(data["user"], cId, data["persona"], "system", datetime.now().strftime("%Y-%m-%d %H:%M:%S"), True)
    return {"cId": cId}

@app.post("/api/v1/addUser")
def addUser(data:dict):
    redis_connector.addUser(data["user"])
    return {"user": data["user"]}

@app.post("/api/v1/delConversation")
def delConversation(data:dict):
    redis_connector.delConversation(data["user"], data["cId"])
    return {"cId": data["cId"]}

@app.post("/api/v1/addAgent")
def addAgent(data:dict):
    aId = mongo_connector.addAgent(data["name"], data["user"], data["tId"], data["vdbs"], data["tools"])
    return {"aId": aId}

@app.get("/api/v1/agent/{user}/{aId}")
def getAgent(user:str, aId:str):
    return mongo_connector.getAgent(user, aId)

@app.get("/api/v1/agents/{user}")
def getAgents(user:str):
    return mongo_connector.getAgents(user)

# 初始化
if __name__ == "__main__":
    import uvicorn
    os.makedirs("files/", exist_ok=True)
    os.makedirs("db/", exist_ok=True)
    os.makedirs("templates/", exist_ok=True)
    retrievers = []
    
    try:
        #connector = RedisConnector('172.19.241.11', 6379)
        redis_connector = RedisConnector(redis_server, redis_port)
    except:
        raise Exception("Redis连接失败")
    
    try:
        mongo_connector = MongoConnector(mongo_server, mongo_port, mongo_username, mongo_password)
    except:
        raise Exception("Mongo连接失败")
    
    uvicorn.run(app, host="0.0.0.0", port=8901)
