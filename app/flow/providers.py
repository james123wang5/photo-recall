"""Optional local adapters. Missing weights are reported, never downloaded."""
import importlib.util
import json
from pathlib import Path
import numpy as np
from ..providers.resources import LOCK, register, inference, empty_cache
from .domain import digest


class QwenRetrieval:
    def __init__(self, root, role, pixels):
        self.root = Path(root)
        self.role = role
        self.pixels = pixels
        self.name = "Qwen3-VL-Embedding-2B" if role == "embedding" else "Qwen3-VL-Reranker-2B"
        self.directory = self.root / self.name
        self.backend = None
        register(self)

    @property
    def installed(self):
        return (self.directory / "config.json").is_file() and (self.directory / "tokenizer.json").is_file() and bool(list(self.directory.glob("*.safetensors")))

    @property
    def version(self):
        manifest = [(p.name, p.stat().st_size, p.stat().st_mtime_ns) for p in sorted(self.directory.glob("*")) if p.is_file()]
        return f"{self.name}:{digest(manifest)[:20]}:p{self.pixels}:official-v1"

    def unload(self):
        with LOCK:
            if self.backend is not None:
                self.backend = None
                empty_cache()

    def _load(self):
        if self.backend is not None and getattr(self, "loaded_version", None) == self.version:
            return
        if self.backend is not None:
            self.unload()
        if not self.installed:
            raise RuntimeError(f"请将完整 {self.name} 权重放到 data/models/{self.name}，程序不会自动下载")
        import torch
        device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
        dtype = torch.float16 if device != "cpu" else torch.float32
        if self.role == "embedding":
            from .vendor.qwen3_vl_embedding import Qwen3VLEmbedder as Backend
        else:
            from .vendor.qwen3_vl_reranker import Qwen3VLReranker as Backend
        self.backend = Backend(str(self.directory), max_length=8192, min_pixels=4096,
                               max_pixels=self.pixels, torch_dtype=dtype,
                               attn_implementation="sdpa", local_files_only=True)
        self.backend.model.to(device)
        if self.role == "reranker":
            self.backend.device = torch.device(device)
            self.backend.score_linear.to(device)
        self.loaded_version = self.version

    def encode(self, text=None, image=None):
        with inference(self):
            self._load()
            value = self.backend.process([{k: v for k, v in {"text": text, "image": image}.items() if v is not None}])
            result = value.float().cpu().numpy()[0].astype(np.float32)
            if not np.isfinite(result).all():
                raise RuntimeError("Qwen 编码返回非有限数值")
            return result / max(float(np.linalg.norm(result)), 1e-12)

    def image(self, image):
        return self.encode(image=image)

    def text(self, text):
        return self.encode(text=text)

    def score(self, query, image):
        with inference(self):
            self._load()
            score = float(self.backend.process({"query": {"text": query}, "documents": [{"image": image}]})[0])
            if not np.isfinite(score):
                raise RuntimeError("Qwen 精排返回非有限数值")
            return score


class GroundingTool:
    """Normalized boxes only. Detection does not certify absence or species."""
    def __init__(self, data):
        self.directory = Path(data) / "models" / "grounding-dino"
        self.model = self.processor = None
        register(self)

    @property
    def installed(self):
        return (self.directory / "config.json").is_file() and bool(list(self.directory.glob("*.safetensors")))

    def unload(self):
        with LOCK:
            if self.model is not None:
                self.model = self.processor = None
                empty_cache()

    def locate(self, image, phrase):
        if not self.installed:
            raise RuntimeError("开放对象定位权重未准备，仍可使用固定局部")
        with inference(self):
            import torch
            from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
            if self.model is None:
                self.processor = AutoProcessor.from_pretrained(self.directory, local_files_only=True)
                self.model = AutoModelForZeroShotObjectDetection.from_pretrained(self.directory, local_files_only=True).eval()
            text = phrase.strip().rstrip(".") + "."
            inputs = self.processor(images=image, text=text, return_tensors="pt")
            with torch.inference_mode():
                outputs = self.model(**inputs)
            result = self.processor.post_process_grounded_object_detection(
                outputs, inputs.input_ids, threshold=.35, text_threshold=.25,
                target_sizes=[(image.height, image.width)])[0]
            return [[max(0., min(1., float(v) / (image.width if i % 2 == 0 else image.height)))
                     for i, v in enumerate(box)] for box in result["boxes"].tolist()[:6]]


class FaceTool:
    version = "insightface-buffalo_l:normalized512:exif:v1"

    def __init__(self, data):
        self.root = Path(data) / "models" / "insightface"
        self.model = None

    @property
    def installed(self):
        directory = self.root / "models" / "buffalo_l"
        return all((directory / name).is_file() for name in (
            "det_10g.onnx", "w600k_r50.onnx", "1k3d68.onnx", "2d106det.onnx", "genderage.onnx")) and importlib.util.find_spec("insightface") is not None

    def faces(self, image):
        if not self.installed:
            raise RuntimeError("人脸组件未准备；请按开发说明安装依赖和本地 buffalo_l 权重")
        if self.model is None:
            from insightface.app import FaceAnalysis
            self.model = FaceAnalysis(name="buffalo_l", root=str(self.root), providers=["CPUExecutionProvider"])
            self.model.prepare(ctx_id=-1, det_size=(640, 640))
        return self.model.get(np.asarray(image)[:, :, ::-1].copy())
