"""Pinned, local-only WeMM retrieval and CORE binary-logit reranking on Metal.

CORE formatting follows ms-swift's Qwen3VLRerankerTemplate. The binary
head is equivalent to softmax([no, yes]); scores are NOT calibrated probabilities.
"""
from functools import lru_cache
from pathlib import Path
import numpy as np
from PIL import Image, ImageOps
from .resources import register, inference, empty_cache, LOCK

WEMM_REVISION = "bbd6cd4bf52cfc6716f752a2df80b2706720bd95"
CORE_REVISION = "64e1a36cd6ecace459a4c7cd8f7d0ef5d6fee30b"
SYSTEM = ('Judge whether the Document meets the requirements based on the Query and the Instruct provided. '
          'Note that the answer can only be "yes" or "no".')


class LocalModel:
    weight_size = 0

    def __init__(self, directory):
        self.directory = Path(directory)
        self.model = self.processor = None
        self.device = None
        self.last_error = None
        register(self)

    @property
    def installed(self):
        weight = self.directory / "model.safetensors"
        return weight.is_file() and weight.stat().st_size == self.weight_size and all(
            (self.directory / name).is_file() for name in ("config.json", "tokenizer.json"))

    def unload(self):
        with LOCK:
            if self.model is not None:
                self.model = None
                if hasattr(self, "score_weight"):
                    self.score_weight = None
                empty_cache()

    def _environment(self):
        import torch
        self.torch = torch
        self.device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
        self.dtype = torch.float16 if self.device != "cpu" else torch.float32
        torch.set_num_threads(4)
        if not self.installed:
            raise RuntimeError(f"{self.name} 本地权重尚未完整下载")

    def _inputs(self, messages, generation=False):
        from qwen_vl_utils import process_vision_info
        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=generation)
        images, _, _ = process_vision_info(messages, image_patch_size=16, return_video_kwargs=True)
        inputs = self.processor(text=text, images=images, return_tensors="pt", padding=False,
                                do_resize=False if images else True)
        for key, value in inputs.items():
            if hasattr(value, "to"):
                inputs[key] = value.to(self.device, dtype=self.dtype) if value.is_floating_point() else value.to(self.device)
        return inputs


class WemmProvider(LocalModel):
    name = "tencent/WeMM-Embedding-2B"
    version = "wemm:" + WEMM_REVISION + ":fp16:p262144:2048:v1"
    weight_size = 5441695216
    max_pixels = 262144

    def _load(self):
        if self.model is not None:
            return
        self._environment()
        from transformers import AutoModel, AutoProcessor
        self.processor = AutoProcessor.from_pretrained(self.directory, local_files_only=True, trust_remote_code=True)
        # This local custom code was inspected and pinned when downloaded.
        self.model = AutoModel.from_pretrained(self.directory, local_files_only=True, trust_remote_code=True,
            dtype=self.dtype, attn_implementation="sdpa").eval().to(self.device)

    def encode(self, *, text=None, image=None):
        with inference(self):
            self._load()
            content = ([{"type": "image", "image": image, "min_pixels": 65536,
                         "max_pixels": self.max_pixels}] if image is not None else [{"type": "text", "text": text}])
            inputs = self._inputs([{"role": "user", "content": content}])
            embedding_id = self.processor.tokenizer.convert_tokens_to_ids("<embedding>")
            if int(inputs["input_ids"][0, -1]) != embedding_id:
                raise RuntimeError("WeMM 输入未以官方 embedding token 结尾，拒绝建立错误索引")
            with self.torch.inference_mode():
                out = self.model.embedding(**inputs, use_cache=False).float().cpu().numpy()[0]
            if not np.isfinite(out).all():
                raise RuntimeError("WeMM 返回非有限向量")
            return out.astype(np.float32) / max(float(np.linalg.norm(out)), 1e-12)

    def image(self, image):
        return self.encode(image=image)

    @lru_cache(maxsize=256)
    def text(self, text):
        return self.encode(text=text)


class CoreReranker(LocalModel):
    name = "Alibaba-NLP/core-reranker-2b"
    version = "core:" + CORE_REVISION + ":fp16:p262144:swift-template:exif:v2"
    weight_size = 4255140312
    max_pixels = 262144

    def _load(self):
        if self.model is not None:
            return
        self._environment()
        from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
        lm = Qwen3VLForConditionalGeneration.from_pretrained(self.directory, local_files_only=True,
                dtype=self.dtype, attn_implementation="sdpa").eval().to(self.device)
        self.processor = AutoProcessor.from_pretrained(self.directory, local_files_only=True)
        yes = self.processor.tokenizer.convert_tokens_to_ids("yes")
        no = self.processor.tokenizer.convert_tokens_to_ids("no")
        self.score_weight = (lm.lm_head.weight[yes] - lm.lm_head.weight[no]).detach().clone()
        self.model = lm.model
        # Do not retain the large vocabulary projection or allocate per-token logits.
        del lm

    @staticmethod
    def messages(query, image, max_pixels=262144):
        return [{"role": "system", "content": SYSTEM}, {"role": "user", "content": [
            {"type": "text", "text": "<Instruct>: Determine if the following image is relevant to the query.<Query>:" + query + "\n<Document>:"},
            {"type": "image", "image": image, "min_pixels": 65536, "max_pixels": max_pixels}]}]

    def score(self, query, image):
        with inference(self):
            self._load()
            if isinstance(image, (str, Path)):
                with Image.open(image) as source:
                    image = ImageOps.exif_transpose(source).convert("RGB")
            else:
                image = ImageOps.exif_transpose(image).convert("RGB")
            inputs = self._inputs(self.messages(query, image, self.max_pixels), generation=True)
            if inputs["input_ids"].shape[-1] > 2048:
                raise RuntimeError("图片与条件超过本机精排输入预算，请减少条件或分辨率")
            with self.torch.inference_mode():
                hidden = self.model(**inputs, use_cache=False).last_hidden_state[:, -1]
                logit = (hidden.float() @ self.score_weight.float()).item()
            if not np.isfinite(logit):
                raise RuntimeError("CORE 返回非有限分数")
            return float(1 / (1 + np.exp(-np.clip(logit, -50, 50))))

    def score_many(self, query, images):
        """At most two images per Metal pass; same pixels, prompt and score head.

        Left padding keeps each sample's last hidden state at its answer position.
        Fall back to serial inference on OOM, without lowering image resolution.
        """
        if not images:
            return []
        if getattr(self,"batch_disabled",False):
            return [self.score(query,image) for image in images]
        if len(images) == 1:
            return [self.score(query,images[0])]
        if len(images) > 2:
            raise ValueError("16GB 本机一次最多精排两张照片")
        try:
            with inference(self):
                self._load()
                from qwen_vl_utils import process_vision_info
                normalized = []
                for image in images:
                    if isinstance(image,(str,Path)):
                        with Image.open(image) as source:
                            normalized.append(ImageOps.exif_transpose(source).convert("RGB"))
                    else:
                        normalized.append(ImageOps.exif_transpose(image).convert("RGB"))
                messages = [self.messages(query,image,self.max_pixels) for image in normalized]
                text = [self.processor.apply_chat_template(m,tokenize=False,add_generation_prompt=True) for m in messages]
                vision,_,_ = process_vision_info(messages,image_patch_size=16,return_video_kwargs=True)
                previous = self.processor.tokenizer.padding_side
                try:
                    self.processor.tokenizer.padding_side = "left"
                    inputs = self.processor(text=text,images=vision,return_tensors="pt",padding=True,do_resize=False)
                finally:
                    self.processor.tokenizer.padding_side = previous
                if inputs["input_ids"].shape[-1] > 2048:
                    raise RuntimeError("图片与条件超过本机精排输入预算")
                for key,value in inputs.items():
                    if hasattr(value,"to"):
                        inputs[key] = value.to(self.device,dtype=self.dtype) if value.is_floating_point() else value.to(self.device)
                with self.torch.inference_mode():
                    hidden = self.model(**inputs,use_cache=False).last_hidden_state[:,-1]
                    logits = (hidden.float() @ self.score_weight.float()).cpu().numpy()
                if not np.isfinite(logits).all():
                    raise RuntimeError("CORE 返回非有限分数")
                return (1/(1+np.exp(-np.clip(logits,-50,50)))).astype(float).tolist()
        except RuntimeError as exc:
            if "out of memory" not in str(exc).lower():
                raise
            self.batch_disabled = True
            exc.__traceback__ = None
            inputs = hidden = logits = None
            empty_cache()
            return [self.score(query,image) for image in images]
