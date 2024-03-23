import os
import configparser

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from datetime import datetime
import urllib.parse
import json
import logging

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
from langchain_community.embeddings import JinaEmbeddings
from langchain.vectorstores import Chroma
from langchain_community.embeddings import MiniMaxEmbeddings

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

os.environ["MINIMAX_GROUP_ID"] = "1768536437306691771"
os.environ["MINIMAX_API_KEY"] = "eyJhbGciOiJSUzI1NiIsInR5cCI6IkpXVCJ9.eyJHcm91cE5hbWUiOiJLZWl0aCIsIlVzZXJOYW1lIjoiS2VpdGgiLCJBY2NvdW50IjoiIiwiU3ViamVjdElEIjoiMTc2ODUzNjQzNzMxNTA4MDM3OSIsIlBob25lIjoiMTUxNjExNjU5NzMiLCJHcm91cElEIjoiMTc2ODUzNjQzNzMwNjY5MTc3MSIsIlBhZ2VOYW1lIjoiIiwiTWFpbCI6IiIsIkNyZWF0ZVRpbWUiOiIyMDI0LTAzLTIwIDIxOjM4OjU1IiwiaXNzIjoibWluaW1heCJ9.cpaXmCZuknX5D49hbnkm8cu2Sl-sOSeYPdPFIKPpVV0mhluS5koWMlLTOE_ortf3e6-g9xa5ivg_CZeOCVn6A2or9yKuCZ2nkEsyP1XFJFA6AoqW3GG2dqehuEDItQ_9BGk4NZ_dO0NnZowFwCejLtZkgXISZiBPOBOEYcJ2nSEFUKUoeKhuZuL8_-RdvNLHGZ82iogACw9Blneo8toxu43gHGIp0jMJfGx18aQp8rimuh2lCpdyxWc4OUduwEWeVXgSge35CrdA6m6h8ZcLRjh2PEXYGiE0tbtott7H4zxAPuTu1R0dH-QOjIWc_dn7kE6s3c9mLd4MmetkFKoreA"


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
    #embedding_function = JinaEmbeddings(
    #    jina_api_key="jina_be39166027554b7897831d22812be11bYeb8ozo9X_2NcZuj0dHEnPPzIrwD", model_name="jina-embeddings-v2-base-zh"
    #)
    embedding_function = MiniMaxEmbeddings()
    db = Chroma.from_documents(documents=docs, embedding=embedding_function, persist_directory=persist_directory)
    db.persist()
    return f"{base_name}_{extension[1::]}"

def loadDB(dbname):
    if not os.path.exists(f"db/{dbname}"):
        return None
    #embedding_function = JinaEmbeddings(
    #    jina_api_key="jina_be39166027554b7897831d22812be11bYeb8ozo9X_2NcZuj0dHEnPPzIrwD", model_name="jina-embeddings-v2-base-zh"
    #)
    embedding_function = MiniMaxEmbeddings()
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
        logging.error(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} - failed to connect")
        yield f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} - failed to connect"
        raise HTTPException(status_code=500, detail=str(e))

def makeRes(message, client, history, user, cId):
    mId = redis_connector.addMessage(user, cId, message, "user", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            stream=True,
            messages=history+[{"role": "user", "content": f"{message}"}],
        )
        ans = ""
        for chunk in response:
            if chunk.choices[0].delta.content is not None:
                ans+=chunk.choices[0].delta.content
                yield chunk.choices[0].delta.content
        redis_connector.addMessage(user, cId, ans, "assistant", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    except Exception as e:
        redis_connector.delMessage(cId, mId)
        logging.error(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} - failed to connect")
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
    #print(history)
    logging.info(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} - {history}")
    
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

@app.post("/api/v1/addConversationFromTemplate")
def addConversationFromTemplate(data:dict):
    templates = os.listdir("templates")
    tId = data["tId"]
    if (tId in [os.path.splitext(temp)[0] for temp in templates]):
        with open(f"templates/{tId}.json", "r", encoding='utf-8') as f:
            cId = redis_connector.addConversation(data["user"])
            try:
                template = json.loads(f.read())
                for t in template["template"]:
                    redis_connector.addMessage(data["user"], cId, t["content"], t["role"], datetime.now().strftime("%Y-%m-%d %H:%M:%S"), True)
            except:
                redis_connector.delConversation(data["user"], cId)
        return {"cId": cId}
    return {"error": "template not found"}

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
async def startup_event():
    os.makedirs("files/", exist_ok=True)
    os.makedirs("db/", exist_ok=True)
    os.makedirs("templates/", exist_ok=True)
    global retrievers
    retrievers = []
    
    #connector = RedisConnector('172.19.241.11', 6379)
    global redis_connector
    redis_connector = RedisConnector(redis_server, redis_port)
    if redis_connector.checkConncetion()=="Not Available":
        raise Exception("Redis连接失败")
    logging.info("Redis已连接")
    
    
    global mongo_connector
    mongo_connector = MongoConnector(mongo_server, mongo_port, mongo_username, mongo_password)
    if mongo_connector.checkConncetion()=="Not Available":
        raise Exception("Mongo连接失败")
    logging.info("Mongo已连接")
    
    logging.basicConfig(filename='miniAPI.log', level=logging.DEBUG)
    logging.info(f"API started at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
app.add_event_handler("startup", startup_event)

if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(app, host="0.0.0.0", port=8901)
