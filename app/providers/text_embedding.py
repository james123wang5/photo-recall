"""Small multilingual text embeddings served by the user's local Ollama."""
import json
import urllib.request
import numpy as np
from . import ollama
from .resources import inference


class TextEmbedding:
    model = "qwen3-embedding:0.6b"
    version = "qwen3-embedding:0.6b:1024:retrieval-v1"

    @property
    def installed(self):
        return self.model in ollama.installed_models()

    def encode(self,texts,query=False):
        if not texts:return []
        inputs = [("Instruct: Given a photo search query, retrieve relevant descriptions of photographs.\nQuery: "+s)
                  if query else s for s in texts]
        req=urllib.request.Request(ollama.BASE+"/api/embed",data=json.dumps({"model":self.model,
            "input":inputs,"truncate":True,"keep_alive":60,"options":{"num_ctx":2048}}).encode(),
            headers={"Content-Type":"application/json"})
        with inference("ollama"),ollama.MODEL_LOCK:
            with urllib.request.urlopen(req,timeout=180) as response:payload=json.load(response)
        values=payload.get("embeddings",[])
        if len(values)!=len(texts):raise RuntimeError("文字向量模型返回数量不完整")
        result=[]
        for value in values:
            v=np.asarray(value,dtype=np.float32)
            if v.shape!=(1024,) or not np.isfinite(v).all():raise RuntimeError("文字向量维度或数值异常")
            result.append(v/max(float(np.linalg.norm(v)),1e-12))
        return result
