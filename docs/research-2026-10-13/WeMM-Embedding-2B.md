[![Hugging Face's logo](https://huggingface.co/front/assets/huggingface_logo-noborder.svg) Hugging Face](https://huggingface.co/)
- [ Models ](https://huggingface.co/models)
- [ Datasets ](https://huggingface.co/datasets)
- [ Spaces ](https://huggingface.co/spaces)
- [ Buckets new](https://huggingface.co/storage)
- [ Docs ](https://huggingface.co/docs)
- [ Enterprise ](https://huggingface.co/enterprise) 
- [Pricing](https://huggingface.co/pricing) 
- 
  - 
Website
    - [ Tasks](https://huggingface.co/tasks) 
    - [ HuggingChat](https://huggingface.co/chat) 
    - [ Collections](https://huggingface.co/collections) 
    - [ Languages](https://huggingface.co/languages) 
    - [ Organizations](https://huggingface.co/organizations) 
  - 
Community
    - [ Blog](https://huggingface.co/blog) 
    - [ Posts](https://huggingface.co/posts) 
    - [ Daily Papers](https://huggingface.co/papers) 
    - [ Hardware](https://huggingface.co/hardware) 
    - [ Learn](https://huggingface.co/learn) 
    - [ Discord](https://huggingface.co/join/discord) 
    - [ Forum](https://discuss.huggingface.co/) 
    - [ GitHub](https://github.com/huggingface) 
  - 
Solutions
    - [ Team & Enterprise](https://huggingface.co/enterprise) 
    - [ Hugging Face PRO](https://huggingface.co/pro) 
    - [ Enterprise Support](https://huggingface.co/support) 
    - [ Inference Providers](https://huggingface.co/inference/models) 
    - [ Inference Endpoints](https://huggingface.co/inference-endpoints) 
    - [ Storage Buckets](https://huggingface.co/storage) 
- 
---
- [Log In](https://huggingface.co/login) 
- [Sign Up](https://huggingface.co/join)

# ![](https://cdn-avatars.huggingface.co/v1/production/uploads/5dd96eb166059660ed1ee413/Lp3m-XLpjQGwBItlvn69q.png) 
[https://huggingface.co/tencent](https://huggingface.co/tencent)
[tencent](https://huggingface.co/tencent)  
/
[WeMM\-Embedding\-2B](https://huggingface.co/tencent/WeMM-Embedding-2B)
Like  118
Follow
![](https://cdn-avatars.huggingface.co/v1/production/uploads/5dd96eb166059660ed1ee413/Lp3m-XLpjQGwBItlvn69q.png) Tencent 12.5k
Feature Extraction  
[https://huggingface.co/models?pipeline_tag=feature-extraction](https://huggingface.co/models?pipeline_tag=feature-extraction)
Transformers  
[https://huggingface.co/models?library=transformers](https://huggingface.co/models?library=transformers)
Safetensors  
[https://huggingface.co/models?library=safetensors](https://huggingface.co/models?library=safetensors)
sentence\-transformers  
[https://huggingface.co/models?library=sentence-transformers](https://huggingface.co/models?library=sentence-transformers)
Chinese  
[https://huggingface.co/models?language=zh](https://huggingface.co/models?language=zh)
English  
[https://huggingface.co/models?language=en](https://huggingface.co/models?language=en)
qwen3\_5  
[https://huggingface.co/models?other=qwen3_5](https://huggingface.co/models?other=qwen3_5)
image\-text\-to\-text  
[https://huggingface.co/models?other=image-text-to-text](https://huggingface.co/models?other=image-text-to-text)
multimodal\-embedding  
[https://huggingface.co/models?other=multimodal-embedding](https://huggingface.co/models?other=multimodal-embedding)
text\-embedding  
[https://huggingface.co/models?other=text-embedding](https://huggingface.co/models?other=text-embedding)
image\-embedding  
[https://huggingface.co/models?other=image-embedding](https://huggingface.co/models?other=image-embedding)
video\-embedding  
[https://huggingface.co/models?other=video-embedding](https://huggingface.co/models?other=video-embedding)
mrl  
[https://huggingface.co/models?other=mrl](https://huggingface.co/models?other=mrl)
custom\_code  
[https://huggingface.co/models?other=custom_code](https://huggingface.co/models?other=custom_code)
arxiv: 2608.24053  
License: apache\-2.0  
[Model card  ](https://huggingface.co/tencent/WeMM-Embedding-2B)
Files Files and versions  
xet
[https://huggingface.co/tencent/WeMM-Embedding-2B/tree/main](https://huggingface.co/tencent/WeMM-Embedding-2B/tree/main)
Community 
3
[https://huggingface.co/tencent/WeMM-Embedding-2B/discussions](https://huggingface.co/tencent/WeMM-Embedding-2B/discussions)
Deploy   
Copy to bucket new   
Use this model  

### Instructions to use tencent/WeMM\-Embedding\-2B with libraries, inference providers, notebooks, and local apps. Follow these links to get started.
- Libraries 
- [ Transformers](https://huggingface.co/tencent/WeMM-Embedding-2B?library=transformers) 

How to use tencent/WeMM\-Embedding\-2B with Transformers:

```
# Use a pipeline as a high-level helper
from transformers import pipeline

pipe = pipeline("feature-extraction", model="tencent/WeMM-Embedding-2B", trust_remote_code=True)
```

```
# pip install -U transformers accelerate
# Load model directly
from transformers import AutoProcessor, AutoModelForMultimodalLM

processor = AutoProcessor.from_pretrained("tencent/WeMM-Embedding-2B", trust_remote_code=True)
model = AutoModelForMultimodalLM.from_pretrained("tencent/WeMM-Embedding-2B", trust_remote_code=True, device_map="auto")
```
- [ sentence\-transformers](https://huggingface.co/tencent/WeMM-Embedding-2B?library=sentence-transformers) 

How to use tencent/WeMM\-Embedding\-2B with sentence\-transformers:

```
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("tencent/WeMM-Embedding-2B", trust_remote_code=True)

sentences = [
    "The weather is lovely today.",
    "It's so sunny outside!",
    "He drove to the stadium."
]
embeddings = model.encode(sentences)

similarities = model.similarity(embeddings, embeddings)
print(similarities.shape)
# [3, 3]
```
- Notebooks 
- [ Google Colab](https://huggingface.co/tencent/WeMM-Embedding-2B/colab)  
- [ Kaggle](https://huggingface.co/tencent/WeMM-Embedding-2B/kaggle)      
- [WeMM\-Embedding\-2B](https://huggingface.co/tencent/WeMM-Embedding-2B#wemm-embedding-2b) 
  - [Installation](https://huggingface.co/tencent/WeMM-Embedding-2B#installation) 
  - [Transformers](https://huggingface.co/tencent/WeMM-Embedding-2B#transformers) 
  - [Sentence Transformers](https://huggingface.co/tencent/WeMM-Embedding-2B#sentence-transformers) 
  - [Matryoshka Embeddings](https://huggingface.co/tencent/WeMM-Embedding-2B#matryoshka-embeddings) 
  - [Serving](https://huggingface.co/tencent/WeMM-Embedding-2B#serving) 
  - [Evaluation](https://huggingface.co/tencent/WeMM-Embedding-2B#evaluation) 
    - [MMEB\-v2](https://huggingface.co/tencent/WeMM-Embedding-2B#mmeb-v2)
    - [MMEB\-v3](https://huggingface.co/tencent/WeMM-Embedding-2B#mmeb-v3)
  - [Citation](https://huggingface.co/tencent/WeMM-Embedding-2B#citation) 
  - [License](https://huggingface.co/tencent/WeMM-Embedding-2B#license) 

# [](https://huggingface.co/tencent/WeMM-Embedding-2B#wemm-embedding-2b)  WeMM\-Embedding\-2B  

[![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97-Hugging%20Face-yellow)](https://huggingface.co/collections/tencent/wemm-embedding) [![Technical Report](https://img.shields.io/badge/%F0%9F%93%84-Technical%20Report-red)](https://arxiv.org/abs/2608.24053) [![GitHub](https://img.shields.io/badge/GitHub-WeMM--Embedding-black?logo=github)](https://github.com/Tencent/WeMM-Embedding)

WeMM\-Embedding\-2B is a universal multimodal embedding model built on Qwen3.5. It accepts text, images, videos, visual documents, and interleaved multimodal inputs, and returns a 2,048\-dimensional L2\-normalized embedding. Audio input is not supported.

## [](https://huggingface.co/tencent/WeMM-Embedding-2B#installation)  Installation  

```bash
pip install torch transformers==5.2.0 "qwen-vl-utils[decord]==0.0.14" \
  "sentence-transformers>=5.7.0" "accelerate>=1.1.0"
```

## [](https://huggingface.co/tencent/WeMM-Embedding-2B#transformers)  Transformers  

```python
import torch
from qwen_vl_utils import process_vision_info
from transformers import AutoModel, AutoProcessor

model_id = "tencent/WeMM-Embedding-2B"
processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
model = AutoModel.from_pretrained(
    model_id, trust_remote_code=True, dtype=torch.bfloat16
).cuda().eval()

messages = [{"role": "user", "content": [
    {"type": "image", "image": "/path/to/image.jpg"},
    {"type": "video", "video": "/path/to/video.mp4"},
    {"type": "text", "text": "This can be any text input."},
]}]
text = processor.apply_chat_template(
    messages, tokenize=False, add_generation_prompt=False
)
images, videos, video_kwargs = process_vision_info(
    messages,
    image_patch_size=16,
    return_video_kwargs=True,
    return_video_metadata=True,
)
if videos is not None:
    videos, video_metadata = zip(*videos)
    videos, video_metadata = list(videos), list(video_metadata)
else:
    video_metadata = None
inputs = processor(
    text=text,
    images=images,
    videos=videos,
    video_metadata=video_metadata,
    return_tensors="pt",
    **video_kwargs,
).to("cuda")

with torch.inference_mode():
    embedding = model.embedding(**inputs)
```

Use any subset of the content items to encode text, image, or video independently.

## [](https://huggingface.co/tencent/WeMM-Embedding-2B#sentence-transformers)  Sentence Transformers  

```python
from sentence_transformers import SentenceTransformer

model_id = "tencent/WeMM-Embedding-2B"
model = SentenceTransformer(model_id, trust_remote_code=True)

queries = [
    "Which Llama 4 model variants are available?",
    "How is mapo tofu prepared?",
]
documents = [
    "Mapo tofu is a Sichuan dish of soft tofu simmered in a spicy, numbing sauce of chili bean paste and Sichuan peppercorn.",
    {
        "image": "https://huggingface.co/datasets/sentence-transformers/example-documents/resolve/main/llama4_hgf.png",
    },
    {
        "video": "https://huggingface.co/datasets/sentence-transformers/example-documents/resolve/main/mapo_tofu.mp4",
    },
]

query_embeddings = model.encode_query(queries)
document_embeddings = model.encode_document(documents)
print(query_embeddings.shape, document_embeddings.shape)
# (2, 2048) (3, 2048)

similarities = model.similarity(query_embeddings, document_embeddings)
print(similarities)
# tensor([[0.0685, 0.5379, 0.0423],
#         [0.7829, 0.1852, 0.4729]])
```

Each input is a string, a URL or path, a `PIL.Image`, or a dict combining `image`, `video`, and `text` keys. Chat messages such as `\{"role": "user", "content": \[\{"type": "image", "image": ...\}, \{"type": "text", "text": ...\}\]\}` are also accepted, which is the way to interleave several images or videos in one input.

## [](https://huggingface.co/tencent/WeMM-Embedding-2B#matryoshka-embeddings)  Matryoshka Embeddings  

```python
embedding_256 = torch.nn.functional.normalize(embedding[..., :256], dim=-1)
```

With Sentence Transformers, pass `truncate\_dim` and let it renormalize:

```python
embeddings_256 = model.encode_document(documents, truncate_dim=256, normalize_embeddings=True)
```

Use a dimension listed in `model.config.matryoshka\_dimensions`. On MMEB\-v2, 256\-dimensional embeddings retain 98.7% of the full\-dimensional image and video performance.

## [](https://huggingface.co/tencent/WeMM-Embedding-2B#serving)  Serving  

vLLM `0.27.0`:

```bash
MODEL_PATH=/path/to/WeMM-Embedding-2B
vllm serve "$MODEL_PATH" \
  --runner pooling \
  --chat-template "$MODEL_PATH/embedding_chat_template.jinja"
```

SGLang `0.5.9`:

```bash
MODEL_PATH=/path/to/WeMM-Embedding-2B
python patch_sglang_video.py
python -m sglang.launch_server \
  --model-path "$MODEL_PATH" \
  --is-embedding \
  --enable-precise-embedding-interpolation
```

## [](https://huggingface.co/tencent/WeMM-Embedding-2B#evaluation)  Evaluation  

### [](https://huggingface.co/tencent/WeMM-Embedding-2B#mmeb-v2)  MMEB\-v2  

Results on 78 datasets from Table 1 of the [technical report](https://github.com/Tencent/WeMM-Embedding/blob/main/assets/WeMM_Embedding_tech_report.pdf). Image and video tasks use Hit@1, while visual\-document tasks use NDCG@5. Higher is better.

| Model |  Size |  AVG |  Image |  Video |  VisDoc | 
|---|---|---|---|---|---|
| VLM2Vec |  2B |  47.8 |  59.7 |  29.0 |  44.0 | 
| GME |  2B |  55.4 |  51.9 |  33.9 |  76.8 | 
| VLM2Vec\-V2 |  2B |  59.3 |  64.9 |  34.9 |  69.2 | 
| Qwen3\-VL\-Embedding |  2B |  73.2 |  75.0 |  61.9 |  79.2 | 
| DME\-Small† |  2B |  74.8 |  75.9 |  65.6 |  79.9 | 
| **WeMM\-Embedding** |  **2B** |  **77.9** |  **79.6** |  **70.8** |  **80.7** | 
| **WeMM\-Embedding** |  **4B** |  **79.2** |  **80.8** |  **72.1** |  **82.0** | 
| VLM2Vec |  8B |  53.2 |  65.5 |  34.0 |  49.1 | 
| GME |  8B |  59.2 |  56.0 |  38.6 |  79.3 | 
| Qwen3\-VL\-Embedding |  8B |  77.8 |  80.1 |  67.1 |  82.4 | 
| DME\-Medium† |  9B |  78.4 |  79.8 |  70.8 |  82.0 | 
| **WeMM\-Embedding** |  **9B** |  **80.6** |  **81.9** |  **74.3** |  **83.3** | 

† Closed\-source leaderboard submission without publicly released model weights or a public inference endpoint.

### [](https://huggingface.co/tencent/WeMM-Embedding-2B#mmeb-v3)  MMEB\-v3  

Results on all 190 tasks from Table 2 of the [technical report](https://github.com/Tencent/WeMM-Embedding/blob/main/assets/WeMM_Embedding_tech_report.pdf). V3\-All includes the 78 MMEB\-v2 tasks, 53 text tasks, 47 agent tasks, 11 audio tasks, and MCMR. Unsupported tasks are assigned a score of zero.

| Model |  Size |  V3\-All |  Text |  Agent |  MCMR |  Audio | 
|---|---|---|---|---|---|---|
| VLM2Vec\-V2 |  2B |  38.3 |  24.5 |  28.7 |  4.1 |  0.0 | 
| Omni\-Embed\-Nemotron |  3B |  43.5 |  39.2 |  36.5 |  26.1 |  36.5 | 
| E5\-Omni |  3B |  44.6 |  26.7 |  36.9 |  31.9 |  30.8 | 
| Qwen3\-VL\-Embedding |  2B |  50.9 |  39.2 |  39.3 |  42.0 |  0.0 | 
| **WeMM\-Embedding** |  **2B** |  **56.0** |  **45.3** |  **45.1** |  **42.5** |  **0.0** | 
| **WeMM\-Embedding** |  **4B** |  **58.2** |  **47.9** |  **49.0** |  **41.9** |  **0.0** | 
| WAVE |  7B |  26.3 |  13.7 |  11.3 |  8.9 |  31.8 | 
| VLM2Vec |  8B |  32.9 |  22.2 |  19.7 |  0.9 |  0.0 | 
| LCO\-Embedding\-Omni |  7B |  40.6 |  32.4 |  27.8 |  20.0 |  43.2 | 
| GME |  8B |  43.6 |  37.1 |  35.6 |  27.3 |  0.0 | 
| E5\-Omni |  7B |  47.1 |  26.9 |  36.7 |  41.1 |  43.0 | 
| Tianmu\-Emb\-Uni |  8B |  53.3 |  43.6 |  39.4 |  38.8 |  38.9 | 
| Qwen3\-VL\-Embedding |  8B |  53.5 |  42.5 |  38.4 |  38.0 |  0.0 | 
| **WeMM\-Embedding** |  **9B** |  **59.5** |  **48.8** |  **51.0** |  **49.3** |  **0.0** | 

Text results use NDCG@5; agent, MCMR, and audio results use Hit@1.

## [](https://huggingface.co/tencent/WeMM-Embedding-2B#citation)  Citation  

If you find this repository useful, please consider giving a star ⭐ and citation

```bibtex
@article{wemm-embedding,
      title={WeMM-Embedding: WeChat Multi-Modal Embedding Technical Report}, 
      author={Junjie Zhou and Ke Mei and Lei Li and Tianyi Wang and Fengyun Rao and Jing Lyu},
      year={2026},
      eprint={2608.24053},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2608.24053}, 
}
```

## [](https://huggingface.co/tencent/WeMM-Embedding-2B#license)  License  

WeMM\-Embedding\-2B, including the code, model parameters, and weights made publicly available by Tencent, is licensed under the [Apache License 2.0](https://huggingface.co/tencent/WeMM-Embedding-2B/blob/main/LICENSE). Third\-party components remain subject to their respective original licenses.
Downloads last month 16,398 
Safetensors[](https://huggingface.co/docs/safetensors)
Model size
3B params
Tensor type
BF16 
·

Chat template

Files info
Inference Providers 
[NEW](https://huggingface.co/docs/inference-providers)
Feature Extraction
[Learn more about feature-extraction](https://huggingface.co/tasks/feature-extraction)
This model isn't deployed by any Inference Provider. 
[🙋  Ask for provider support](https://huggingface.co/spaces/huggingface/InferenceSupport/discussions/new?title=tencent/WeMM-Embedding-2B&description=React%20to%20this%20comment%20with%20an%20emoji%20to%20vote%20for%20%5Btencent%2FWeMM-Embedding-2B%5D(%2Ftencent%2FWeMM-Embedding-2B)%20to%20be%20supported%20by%20Inference%20Providers.%0A%0A(optional)%20Which%20providers%20are%20you%20interested%20in%3F%20(Novita%2C%20Hyperbolic%2C%20Together%E2%80%A6)%0A)

## Model tree for tencent/WeMM\-Embedding\-2B [](https://huggingface.co/docs/hub/model-cards#specifying-a-base-model)

Base model
[Qwen/Qwen3.5\-2B\-Base](https://huggingface.co/Qwen/Qwen3.5-2B-Base)
Finetuned
[Qwen/Qwen3.5\-2B](https://huggingface.co/Qwen/Qwen3.5-2B)
Finetuned
\([492](https://huggingface.co/models?other=base_model:finetune:Qwen/Qwen3.5-2B)\) 
this model
Finetunes
[4 models](https://huggingface.co/models?other=base_model:finetune:tencent/WeMM-Embedding-2B)
Quantizations
[8 models](https://huggingface.co/models?other=base_model:quantized:tencent/WeMM-Embedding-2B)

## Spaces using tencent/WeMM\-Embedding\-2B 6
📈
tardellirs/model\-pulse
[https://huggingface.co/spaces/tardellirs/model-pulse](https://huggingface.co/spaces/tardellirs/model-pulse)
📊
mteb/leaderboard
[https://huggingface.co/spaces/mteb/leaderboard](https://huggingface.co/spaces/mteb/leaderboard)
⚡
fastgpu/vram\-and\-gpu\-cost\-calculator
[https://huggingface.co/spaces/fastgpu/vram-and-gpu-cost-calculator](https://huggingface.co/spaces/fastgpu/vram-and-gpu-cost-calculator)
📈
emanforhelp/model\-pulse
[https://huggingface.co/spaces/emanforhelp/model-pulse](https://huggingface.co/spaces/emanforhelp/model-pulse)
🧮
aquanode/gpu\-vram\-calculator
[https://huggingface.co/spaces/aquanode/gpu-vram-calculator](https://huggingface.co/spaces/aquanode/gpu-vram-calculator)
🔍
jw78/wemm\-embedding\-demo
[https://huggingface.co/spaces/jw78/wemm-embedding-demo](https://huggingface.co/spaces/jw78/wemm-embedding-demo)
\+ 1 Spaces 

## Collection including tencent/WeMM\-Embedding\-2B

#### WeMM\-Embedding
Collection
4 items • Updated 3 days ago •  33
[https://huggingface.co/collections/tencent/wemm-embedding](https://huggingface.co/collections/tencent/wemm-embedding)

## Paper for tencent/WeMM\-Embedding\-2B

#### WeMM\-Embedding: WeChat Multi\-Modal Embedding Technical Report
Paper • 2608.24053 • Published Aug 25 •  71 
[https://huggingface.co/papers/2608.24053](https://huggingface.co/papers/2608.24053)
System theme   
Company
[TOS](https://huggingface.co/terms-of-service)
[Privacy](https://huggingface.co/privacy)
[About](https://huggingface.co/huggingface)
[Careers](https://apply.workable.com/huggingface/)
[Hugging Face](https://huggingface.co/)
Website
[Models](https://huggingface.co/models)
[Datasets](https://huggingface.co/datasets)
[Spaces](https://huggingface.co/spaces)
[Pricing](https://huggingface.co/pricing)
[Docs](https://huggingface.co/docs)
Inference providers allow you to run inference using different serverless providers.
