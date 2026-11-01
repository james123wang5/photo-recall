import json,urllib.request,time
from pathlib import Path
for name in ['qwen3-embedding:0.6b','qwen3.5:4b']:
 print('准备本地模型',name,flush=True)
 req=urllib.request.Request('http://127.0.0.1:11434/api/pull',data=json.dumps({'model':name,'stream':True}).encode(),headers={'Content-Type':'application/json'})
 last=0
 with urllib.request.urlopen(req,timeout=3600) as response:
  for line in response:
   data=json.loads(line)
   if data.get('error'):raise RuntimeError(data['error'])
   if time.monotonic()-last>15 or data.get('status')=='success':
    print(name,data.get('status'),data.get('completed',0),data.get('total',0),flush=True);last=time.monotonic()
