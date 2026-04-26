from pymongo import MongoClient
from langgraph.checkpoint.mongodb import MongoDBSaver
from langchain_core.messages import HumanMessage, AIMessage

# 1. 建立连接
client = MongoClient("mongodb://localhost:27017")
checkpointer = MongoDBSaver(client,db_name="qiang_agent_db3")

# 2. 模拟你的图 (只需要用同样的 checkpointer 即可)
# 注意：不需要重新跑一遍逻辑，只需要获取状态
config = {"configurable": {"thread_id": "tomie_003"}}


# 3. 直接从数据库捞出当前状态
# ... 前面代码保持不变 ...

state = checkpointer.get(config)

if state:
    print("--- 数据库读取成功 ---")

    # 1. 提取消息列表
    messages = state.get("channel_values", {}).get("messages", [])
    summary = state.get("channel_values", {}).get("summary", "暂无摘要")

    # 2. 在这里打印消息数量
    print(f"📊 当前数据库中存储的消息总数: {len(messages)}")
    print(f"📝 当前存储的摘要: {summary}")

    print("\n--- 详细对话历史 ---")
    for msg in messages:
        role = "用户" if isinstance(msg, HumanMessage) else "AI"
        # 增加一个简单的类型判断，防止有其他类型的消息导致报错
        content = getattr(msg, 'content', str(msg))
        print(f"[{role}]: {content}")
else:
    print("❌ 没找到 tomie_003 的记录，请检查 thread_id 是否正确。")