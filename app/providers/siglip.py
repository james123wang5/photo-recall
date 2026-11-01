"""Optional, cache-only SigLIP 2 adapter. No implicit model downloads."""
from pathlib import Path
import threading
from functools import lru_cache
import numpy as np
from .resources import inference, register, empty_cache, LOCK


class SiglipProvider:
    name = "google/siglip2-base-patch16-224"

    def __init__(self, model_dir: Path):
        self.model_dir = model_dir
        self.model = self.processor = None
        self.lock = threading.Lock()
        self.device = None
        register(self)

    def unload(self):
        with LOCK:
            if self.model is not None:
                self.model = None
                empty_cache()

    @property
    def installed(self):
        required = ["config.json", "model.safetensors", "preprocessor_config.json", "tokenizer.json", "tokenizer_config.json"]
        return all((self.model_dir / name).is_file() and (self.model_dir / name).stat().st_size > 0 for name in required)

    def _load(self):
        if self.model is not None:
            return
        if not self.installed:
            raise RuntimeError("本地 SigLIP 2 尚未准备。请先按 README 安装模型。")
        try:
            import torch
            from transformers import AutoModel, AutoProcessor
        except ImportError as exc:
            raise RuntimeError("缺少 AI 依赖，请先安装 requirements-ai.txt。") from exc
        self.torch = torch
        self.device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
        self.processor = AutoProcessor.from_pretrained(str(self.model_dir), local_files_only=True)
        self.model = AutoModel.from_pretrained(str(self.model_dir), local_files_only=True).eval().to(self.device)

    def _encode(self, **kwargs):
        with inference(self), self.lock:
            self._load()
            inputs = self.processor(return_tensors="pt", **kwargs).to(self.device)
            with self.torch.inference_mode():
                if "images" in kwargs:
                    out = self.model.get_image_features(**inputs)
                else:
                    out = self.model.get_text_features(**inputs)
            # Transformers versions may expose a pooled output rather than a tensor.
            if hasattr(out, "pooler_output"):
                out = out.pooler_output
            vector = out.float().cpu().numpy().reshape(-1)
            return vector / max(float(np.linalg.norm(vector)), 1e-12)

    def image(self, image):
        return self._encode(images=image)

    @lru_cache(maxsize=256)
    def text(self, text):
        return self._encode(text=[text], padding="max_length", truncation=True)
