import redis

from base.config import config


client = redis.Redis(
    host=config.REDIS_HOST,
    port=config.REDIS_PORT,
    password=config.REDIS_PASSWORD,
    db=config.REDIS_DB,
    decode_responses=True
)

try:
    result = client.ping()
    print("Redis连接结果：", result)
finally:
    client.close()
#TODO 测试redis有没有连接上我们项目的配置
