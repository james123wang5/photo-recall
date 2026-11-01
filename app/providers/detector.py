"""Local, bounded object detections. Counts are lower bounds, not proof of absence."""
import importlib.util
import os
import threading


class Detector:
    version = "yolo26n-1280-hard040-soft015-v2"

    def __init__(self, data):
        self.data = data
        self.path = data / "models" / "yolo26n.pt"
        self.model = None
        self.lock = threading.Lock()

    @property
    def ready(self):
        return self.path.is_file() and importlib.util.find_spec("ultralytics") is not None

    def detect(self, path):
        if not self.ready:
            return {}
        with self.lock:
            if self.model is None:
                (self.data / "detector-config" / "Ultralytics").mkdir(parents=True, exist_ok=True)
                os.environ.setdefault("YOLO_CONFIG_DIR", str(self.data / "detector-config"))
                os.environ.setdefault("YOLO_AUTOINSTALL", "false")
                os.environ.setdefault("YOLO_OFFLINE", "true")
                from ultralytics import YOLO, settings
                settings.update({"sync": False})
                self.model = YOLO(str(self.path))
            result = self.model.predict(str(path), imgsz=1280, conf=.15, device="cpu", verbose=False, save=False)[0]
            facts = {}
            uncertain = []
            for box, label, confidence in zip(result.boxes.xyxy.tolist(), result.boxes.cls.tolist(), result.boxes.conf.tolist()):
                name = result.names[int(label)]
                if confidence < .4:
                    uncertain.append((name, box, confidence))
                    continue
                fact = facts.setdefault(name, {"lower_bound": 0, "boxes": [], "source": self.version})
                fact["lower_bound"] += 1
                fact["boxes"].append({"xyxy": [round(x, 1) for x in box], "confidence": round(confidence, 3)})
            for name, box, confidence in uncertain:
                fact = facts.setdefault(name, {"lower_bound": 0, "boxes": [], "source": self.version})
                # Overlapping low-score boxes often describe the same partly occluded subject.
                # Distinct low-score targets cannot prove a count, but they prevent a false exact-count claim.
                overlaps = False
                area = max(1, (box[2]-box[0])*(box[3]-box[1]))
                for known in fact["boxes"]:
                    b = known["xyxy"]
                    intersection = max(0,min(box[2],b[2])-max(box[0],b[0])) * max(0,min(box[3],b[3])-max(box[1],b[1]))
                    other_area = max(1,(b[2]-b[0])*(b[3]-b[1]))
                    if intersection/min(area,other_area) > .2:
                        overlaps = True
                        break
                if not overlaps:
                    fact.setdefault("possible_boxes", []).append({"xyxy": [round(x,1) for x in box], "confidence": round(confidence,3)})
            for fact in facts.values():
                fact["possible_extra"] = len(fact.get("possible_boxes", []))
            return facts
