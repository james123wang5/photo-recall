"""Optional OCR: small Apple Vision text gate, full recognition only if needed."""
import json
import queue
import subprocess
import sys
import tempfile
import threading
from PIL import Image, ImageOps


class LocalOCR:
    paddle_version = "ppocr6-medium:paddleocr3.7:exif2048:v1"
    vision_version = "apple-vision:detect960:accurate2048:v2"

    def __init__(self,root):
        self.root=root
        self.python=root/".venv-ocr/bin/python"
        self.vision=root/"data/bin/vision-ocr"
        self.process=None
        self.lock=threading.Lock()
        self.failed=False
        self.last_error=None

    @property
    def paddle_ready(self):
        return self.python.exists() and all((self.root/"data/models/paddleocr"/name/"inference.pdiparams").exists()
            for name in ("PP-OCRv6_medium_det","PP-OCRv6_medium_rec"))

    @property
    def version(self):
        if self.vision.exists() and sys.platform=="darwin":return self.vision_version
        return self.paddle_version if self.paddle_ready and not self.failed else self.vision_version

    def status(self):
        return {"paddle_ready":self.paddle_ready,"vision_ready":self.vision.exists(),
            "selected":"Apple Vision 按需文字识别" if self.vision.exists() and sys.platform=="darwin" else "PaddleOCR PP-OCRv6",
            "version":self.version,"last_error":self.last_error}

    def _start(self):
        if self.process and self.process.poll() is None:return
        self.responses=queue.Queue()
        log=open(self.root/"data/ocr-worker.log","a")
        self.process=subprocess.Popen([str(self.python),str(self.root/"scripts/ocr_worker.py")],
            stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=log,text=True,bufsize=1)
        log.close()
        def pump(proc,responses):
            for line in proc.stdout:
                if line.startswith("PHOTORECALL_OCR:"):responses.put(line.split(":",1)[1])
            responses.put(None)
        threading.Thread(target=pump,args=(self.process,self.responses),daemon=True).start()

    def close(self):
        with self.lock:
            if self.process:
                self.process.terminate()
                try:self.process.wait(timeout=5)
                except subprocess.TimeoutExpired:self.process.kill();self.process.wait(timeout=5)
                self.process=None

    def read(self,path,force=False):
        with Image.open(path) as source:
            image=ImageOps.exif_transpose(source).convert("RGB")
            image.thumbnail((2048,2048))
        with tempfile.TemporaryDirectory(prefix="photorecall-ocr-") as temp:
            from pathlib import Path
            normalized=Path(temp)/"image.png"
            # Text is an auxiliary signal. A negative gate is cached and never
            # excludes the image from visual search. Tiny text may be missed.
            if self.vision.exists() and sys.platform=="darwin":
                try:
                    small=image.copy();small.thumbnail((960,960));small.save(normalized)
                    result=subprocess.run([str(self.vision),str(normalized),"detect"],capture_output=True,text=True,timeout=10,check=True)
                    regions=json.loads(result.stdout)["regions"]
                    if not regions and not force:
                        return {"provider":"Apple Vision","version":self.vision_version,"lines":[],
                            "text_gate":"no_regions","gate_size":960}
                    image.save(normalized)
                    result=subprocess.run([str(self.vision),str(normalized)],capture_output=True,text=True,timeout=20,check=True)
                    return {"provider":"Apple Vision","version":self.vision_version,**json.loads(result.stdout),
                        "text_gate":"regions_found","gate_size":960}
                except Exception as exc:
                    self.last_error="Vision："+str(exc)[:260]
            image.save(normalized)
            if self.paddle_ready and not self.failed:
                try:
                    with self.lock:
                        self._start()
                        self.process.stdin.write(json.dumps({"path":str(normalized),"width":image.width,"height":image.height})+"\n")
                        self.process.stdin.flush()
                        response=self.responses.get(timeout=120)
                        if response is None:raise RuntimeError("PaddleOCR worker 提前退出")
                        result=json.loads(response)
                        if not result.get("ok"):raise RuntimeError(result.get("error","OCR失败"))
                    return {"provider":"PaddleOCR PP-OCRv6","version":self.paddle_version,"lines":result["lines"]}
                except Exception as exc:
                    self.last_error=str(exc)[:300]
                    self.failed=True
                    self.close()
            raise RuntimeError(self.last_error or "本地OCR尚未准备")
