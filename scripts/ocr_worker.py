"""Isolated PP-OCRv6 CPU worker; no downloads and no changes to another project."""
import contextlib
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK", "True")
os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("PADDLE_PDX_CACHE_HOME", str(Path(__file__).resolve().parents[1]/"data/paddle-cache"))

PREFIX = "PHOTORECALL_OCR:"
pipeline = None
for line in sys.stdin:
    try:
        request = json.loads(line)
        with contextlib.redirect_stdout(sys.stderr):
            if pipeline is None:
                from paddleocr import PaddleOCR
                root = Path(__file__).resolve().parents[1]/"data/models/paddleocr"
                pipeline = PaddleOCR(ocr_version="PP-OCRv6",
                    text_detection_model_name="PP-OCRv6_medium_det",
                    text_recognition_model_name="PP-OCRv6_medium_rec",
                    text_detection_model_dir=str(root/"PP-OCRv6_medium_det"),
                    text_recognition_model_dir=str(root/"PP-OCRv6_medium_rec"),
                    use_doc_orientation_classify=False,use_doc_unwarping=False,
                    use_textline_orientation=False,device="cpu",cpu_threads=4)
            output = list(pipeline.predict(request["path"]))
            raw = output[0].json if output else {}
            if isinstance(raw,str):raw=json.loads(raw)
            raw = raw.get("res",raw)
            texts,scores,polys = raw.get("rec_texts",[]),raw.get("rec_scores",[]),raw.get("rec_polys",[])
            lines = []
            for i,text in enumerate(texts):
                poly = polys[i].tolist() if hasattr(polys[i],"tolist") else polys[i]
                xs,ys = [float(p[0]) for p in poly],[float(p[1]) for p in poly]
                lines.append({"text":str(text),"confidence":float(scores[i]),
                    "box":[min(xs)/request["width"],min(ys)/request["height"],max(xs)/request["width"],max(ys)/request["height"]]})
        result={"ok":True,"lines":lines}
    except Exception as exc:
        result={"ok":False,"error":str(exc)[:500]}
    print(PREFIX+json.dumps(result,ensure_ascii=False),flush=True)
