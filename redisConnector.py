import redis
import uuid
import time

def generate_unique_id():
    # 生成UUID
    unique_id = str(uuid.uuid4())

    # 获取当前时间戳（以秒为单位）
    timestamp = int(time.time())

    # 将UUID和时间戳组合成一个唯一ID
    combined_id = f"{unique_id}-{timestamp}"

    return combined_id


class RedisConnector():
    def __init__(self, host, port, decode_responses=True) -> None:
        self.redis_client = redis.StrictRedis(host=host, port=port, decode_responses=decode_responses)
        
    def checkConncetion(self):
        try:
            return self.redis_client.ping()
        except:
            return "Not Available"

    def addUser(self, user:str):
        self.redis_client.rpush('users', user)
        
    def getUsers(self):
        return self.redis_client.lrange('users', 0, -1)
    
    def addConversation(self, user:str):
        cId = generate_unique_id()
        self.redis_client.rpush(f"u:{user}", cId)
        return cId
    
    def addMessage(self, user:str, cId:str, content:str, role, time, first=False):
        if (not first) and (cId not in self.redis_client.lrange(f"u:{user}", 0, -1)):
            return -1
        if (role not in ["user", "system", "assistant"]):
            return -2
        mId = generate_unique_id()
        self.redis_client.rpush(f"c:{cId}", mId)
        self.redis_client.hset(f"m:{mId}", "content", content)
        self.redis_client.hset(f"m:{mId}", "role", role)
        self.redis_client.hset(f"m:{mId}", "time", time)
        return mId

    def getCoversations(self, user:str):
        return self.redis_client.lrange(f"u:{user}", 0, -1)
    
    def getMessages(self, cId:str):
        return self.redis_client.lrange(f"c:{cId}", 0, -1)
    
    def getMessage(self, mId:str):
        return self.redis_client.hgetall(f"m:{mId}")

    def delConversation(self, user, cId:str):
        mIds = self.redis_client.lrange(f"c:{cId}", 0, -1)
        for mId in mIds:
            self.redis_client.delete(f"m:{mId}")
        self.redis_client.delete(f"c:{cId}")
        self.redis_client.lrem(f"u:{user}", 0, cId)
        return 0
    
    def delMessage(self, cId:str, mId:str):
        self.redis_client.delete(f"m:{mId}")
        self.redis_client.lrem(f"c:{cId}", 0, mId)
        return 0
    
    def flushAll(self):
        self.redis_client.flushall()