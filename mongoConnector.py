import pymongo
import uuid
import time
from datetime import datetime

def generate_unique_id():
    # 生成UUID
    unique_id = str(uuid.uuid4())

    # 获取当前时间戳（以秒为单位）
    timestamp = int(time.time())

    # 将UUID和时间戳组合成一个唯一ID
    combined_id = f"{unique_id}-{timestamp}"

    return combined_id

class MongoConnector():
    def __init__(self, host, port,username, password) -> None:
        self.mongo_client = client = pymongo.MongoClient(f"mongodb://{username}:{password}@{host}:{port}/")
        
    def checkConncetion(self):
        try:
            return self.mongo_client.server_info()
        except:
            return "Not Available"
        
    def addAgent(self, name, user, tId, vdbs, tools):
        
        aId = generate_unique_id()
        self.mongo_client.SEAI["agents"].insert_one(
            {
                "aId" : aId,
                "name": name,
                "user": user, 
                "tId": tId, 
                "vdbs": vdbs, 
                "tools": tools,
                "create_time": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            })
        return aId
        
    def getAgent(self, user, aId):
        query = {
            "user": user,
            "aId": aId
        }
        
        projection = {
            "_id": 0,
        }
        cursor = self.mongo_client["SEAI"]["agents"].find(query, projection)
        return list(cursor)
    
    def getAgents(self, user):
        query = {
            "user": user
        }
        
        projection = {
            "_id": 0,
        }
        cursor = self.mongo_client["SEAI"]["agents"].find(query, projection)
        return list(cursor)