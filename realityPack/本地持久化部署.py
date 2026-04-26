
import os
import pickle
import numpy as np
import faiss
import dashscope
from dashscope import TextEmbedding
from dotenv import load_dotenv
from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIChatModel

# 1. 加载环境变量
load_dotenv()
# 2. 配置模型 (确保环境变量中有 OPENAI_BASE_URL 和 OPENAI_API_KEY)
model = OpenAIChatModel(
    model_name='qwen-plus',
)

def embed(texts):
    if isinstance(texts, str):
        texts = [texts]

    resp = TextEmbedding.call(
        model="text-embedding-v2",
        input=texts
    )

    return [item["embedding"] for item in resp["output"]["embeddings"]]
