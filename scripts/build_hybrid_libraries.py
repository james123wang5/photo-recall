"""Local one-shot indexing queue. Stops on pause; records errors and continues other libraries."""
import json,time,urllib.request
from pathlib import Path
BASE='http://127.0.0.1:8765'
def get(path):
 with urllib.request.urlopen(BASE+path,timeout=15) as r:return json.load(r)
def post(path):
 state=get('/api/state')
 req=urllib.request.Request(BASE+path,data=b'{}',headers={'Content-Type':'application/json','X-PhotoRecall-Token':state['token']})
 with urllib.request.urlopen(req,timeout=15) as r:return json.load(r)
libraries=[l for l in get('/api/state')['libraries'] if not l['is_demo']]
# Repair coverage in the most recently searched real library first.
import sqlite3
with sqlite3.connect(Path(__file__).resolve().parents[1]/'data/library.sqlite3') as con:
 row=con.execute("SELECT library_id FROM sessions WHERE library_id!='demo' AND conditions!='[]' ORDER BY updated DESC LIMIT 1").fetchone()
priority=row[0] if row else None
libraries.sort(key=lambda l:(l['id']!=priority,l['count']))
incomplete=[]
for library in libraries:
 lid=library['id'];print('开始本地增强索引',library['name'],library['count'],flush=True)
 post('/api/libraries/'+lid+'/hybrid-index')
 while True:
  job=get('/api/libraries/'+lid+'/job')
  print(job['status'],job['done'],job['total'],job['current'],flush=True)
  if job['status']=='complete':
   if job.get('errors'):
    incomplete.append(library['name']);print('部分文件失败，记录保留；继续其他图库。',flush=True)
   break
  if job['status']=='error':
   incomplete.append(library['name']);print('当前图库失败，保留记录并继续下一图库。',job.get('errors',[])[:2],flush=True);break
  if job['status'] in {'paused','idle'}:
   print('任务暂停或未完成，队列退出，不自动重试。',flush=True);raise SystemExit(0)
  time.sleep(30)
print('索引队列结束；仍需补建：'+('、'.join(incomplete) if incomplete else '无'),flush=True)
