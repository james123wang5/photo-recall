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

# ![](https://cdn-avatars.huggingface.co/v1/production/uploads/63fc4c00a3c067e62899d32b/dfd_EcIfylvu3sdc2WMqX.png) 
[https://huggingface.co/Alibaba-NLP](https://huggingface.co/Alibaba-NLP)
[Alibaba\-NLP](https://huggingface.co/Alibaba-NLP)  
/
[core\-reranker\-2b](https://huggingface.co/Alibaba-NLP/core-reranker-2b)
Like  2
Follow
![](https://cdn-avatars.huggingface.co/v1/production/uploads/63fc4c00a3c067e62899d32b/dfd_EcIfylvu3sdc2WMqX.png) Alibaba\-NLP 1.71k
Feature Extraction  
[https://huggingface.co/models?pipeline_tag=feature-extraction](https://huggingface.co/models?pipeline_tag=feature-extraction)
Transformers  
[https://huggingface.co/models?library=transformers](https://huggingface.co/models?library=transformers)
Safetensors  
[https://huggingface.co/models?library=safetensors](https://huggingface.co/models?library=safetensors)
qwen3\_vl  
[https://huggingface.co/models?other=qwen3_vl](https://huggingface.co/models?other=qwen3_vl)
image\-text\-to\-text  
[https://huggingface.co/models?other=image-text-to-text](https://huggingface.co/models?other=image-text-to-text)
embeddings  
[https://huggingface.co/models?other=embeddings](https://huggingface.co/models?other=embeddings)
multimodal  
[https://huggingface.co/models?other=multimodal](https://huggingface.co/models?other=multimodal)
retrieval  
[https://huggingface.co/models?other=retrieval](https://huggingface.co/models?other=retrieval)
compositional\-reasoning  
[https://huggingface.co/models?other=compositional-reasoning](https://huggingface.co/models?other=compositional-reasoning)
vision  
[https://huggingface.co/models?other=vision](https://huggingface.co/models?other=vision)
reranker\-distillation  
[https://huggingface.co/models?other=reranker-distillation](https://huggingface.co/models?other=reranker-distillation)
arxiv: 2609.04083  
arxiv: 2305.03689  
arxiv: 2406.11171  
License: cc\-by\-4.0  
[Model card  ](https://huggingface.co/Alibaba-NLP/core-reranker-2b)
Files Files and versions  
xet
[https://huggingface.co/Alibaba-NLP/core-reranker-2b/tree/main](https://huggingface.co/Alibaba-NLP/core-reranker-2b/tree/main)
[Community  ](https://huggingface.co/Alibaba-NLP/core-reranker-2b/discussions)
Deploy   
Copy to bucket new   
Use this model  

### Instructions to use Alibaba\-NLP/core\-reranker\-2b with libraries, inference providers, notebooks, and local apps. Follow these links to get started.
- Libraries 
- [ Transformers](https://huggingface.co/Alibaba-NLP/core-reranker-2b?library=transformers) 

How to use Alibaba\-NLP/core\-reranker\-2b with Transformers:

```
# Use a pipeline as a high-level helper
from transformers import pipeline

pipe = pipeline("feature-extraction", model="Alibaba-NLP/core-reranker-2b")
```

```
# pip install -U transformers accelerate
# Load model directly
from transformers import AutoProcessor, AutoModelForMultimodalLM

processor = AutoProcessor.from_pretrained("Alibaba-NLP/core-reranker-2b")
model = AutoModelForMultimodalLM.from_pretrained("Alibaba-NLP/core-reranker-2b", device_map="auto")
```
- Notebooks 
- [ Google Colab](https://huggingface.co/Alibaba-NLP/core-reranker-2b/colab)  
- [ Kaggle](https://huggingface.co/Alibaba-NLP/core-reranker-2b/kaggle)      
- [Model Family](https://huggingface.co/Alibaba-NLP/core-reranker-2b#model-family) 
- [Highlights](https://huggingface.co/Alibaba-NLP/core-reranker-2b#highlights) 
- [Training](https://huggingface.co/Alibaba-NLP/core-reranker-2b#training) 
- [Results](https://huggingface.co/Alibaba-NLP/core-reranker-2b#results) 
- [Usage](https://huggingface.co/Alibaba-NLP/core-reranker-2b#usage) 
- [Citation](https://huggingface.co/Alibaba-NLP/core-reranker-2b#citation) 
- [Acknowledgements](https://huggingface.co/Alibaba-NLP/core-reranker-2b#acknowledgements) 

# Core\-Embed: Improving Compositional Reasoning in MLLM Embedding via Reranker Distillation

[![arXiv](https://img.shields.io/badge/arXiv-CORE-b31b1b.svg)](https://arxiv.org/abs/2609.04083) [![GitHub](https://img.shields.io/badge/GitHub-CODE-black.svg)](https://github.com/) ![License: CC\-BY\-4.0](https://img.shields.io/badge/License-CC--BY--4.0-blue.svg) 

Core\-Embed is an MLLM\-based multimodal embedding model that resolves fine\-grained attribute\-object bindings by distilling a reranker's compositional judgments into the embedding space.

## [](https://huggingface.co/Alibaba-NLP/core-reranker-2b#model-family)  Model Family  

| Model |  Backbone |  Parameters |  Outputs |  Modalities | 
|---|---|---|---|---|
| core\-emb\-2b |  VL\-Emb \(Qwen3\-VL\) |  2B |  Dense embedding |  Text, image | 
| core\-emb\-8b |  VL\-Emb \(Qwen3\-VL\) |  8B |  Dense embedding |  Text, image | 
| core\-reranker\-2b |  Qwen3\-VL\-Reranker |  2B |  Relevance score |  Text, image | 
| core\-reranker\-8b |  Qwen3\-VL\-Reranker |  8B |  Relevance score |  Text, image | 

## [](https://huggingface.co/Alibaba-NLP/core-reranker-2b#highlights)  Highlights  
- **Compositional retrieval**: distinguishes scenes with the same concepts but different attribute\-object bindings \(e.g., "a white plate and a black chair" vs. "a black plate and a white chair"\). 
- **Rank\-KL distillation**: trained to reproduce the reranker teacher's fine\-grained ranking over a five\-level compositional matching spectrum \(full match, partial presence, attribute error, object error, full mismatch\), instead of collapsing all negatives into one class as InfoNCE does. 
- **General retrieval preserved**: continual training from a strong MLLM embedding backbone, so COCO and Flickr30k retrieval quality is retained while compositional accuracy improves. 

## [](https://huggingface.co/Alibaba-NLP/core-reranker-2b#training)  Training  

Candidate lists are synthesized from LAION\-400M seed images: Qwen3\-VL\-32B extracts structured scene representations and generates queries plus five captions spanning the matching levels, Z\-Image\-Turbo generates the candidate images, and MLLM\-based verification filters low\-quality tuples. The student \(VL\-Emb\) is then trained with the Rank\-KL objective — a KL divergence between temperature\-softened teacher reranker scores and student cosine similarities over each candidate list. Core\-Reranker models are fine\-tuned from Qwen3\-VL\-Reranker on the same synthesized data.

## [](https://huggingface.co/Alibaba-NLP/core-reranker-2b#results)  Results  

On compositional reasoning benchmarks \(COLA, SugarCrepe\+\+, NegBench\):
- **Core\-Reranker\-8B**: 82.7% total average, \+10.7 points over Jina\-Reranker, while recovering negation sensitivity that standard reranker fine\-tuning erodes. 
- **Core\-Embed\-8B**: best total average \(0.666\) among all evaluated embedding models, \+5.7 points over its VL\-Emb\-8B backbone. 
- Gains transfer to the MCMR multi\-condition retrieval benchmark \(R@1 0.375 → 0.412\) with COCO/Flickr30k performance fully preserved. 

See the paper for the full evaluation across 12 embedding baselines and 5 reranker baselines.

## [](https://huggingface.co/Alibaba-NLP/core-reranker-2b#usage)  Usage  

Requires a recent `transformers` build with Qwen3\-VL support:

```bash
pip install "transformers>=4.57.0" torch qwen-vl-utils pillow
```

The model is loaded through the wrapper classes provided in the [GitHub repository](https://github.com/):

```python
import torch
from models.qwen3_vl.qwen3_vl_wrapper import Qwen3VLForEmbedding, Qwen3VLForEmbeddingProcessor

model_path = "core-emb-8b"
model = Qwen3VLForEmbedding.from_pretrained(
    model_path, device_map="cuda", torch_dtype=torch.bfloat16
)
processor = Qwen3VLForEmbeddingProcessor.from_pretrained(
    model_path,
    instruction_standalone=True,
    max_length=3024,
    min_pixels=32*32*4,
    max_pixels=32*32*1280,
    total_pixels=32*32*4500,
    num_frames=48,
)

query_inst = "Find me an everyday image that matches the given caption: "
psg_inst = "Represent the given image."

queries = ["a white plate and a black chair"]
inputs = processor(
    texts=queries, images=[None], videos=[None],
    instruction=[query_inst], is_query=True,
)
with torch.inference_mode():
    query_emb = model(**inputs.to(model.device)).last_hidden_state[:, -1]

images = ["path/to/image.jpg"]
inputs = processor(
    texts=[None], images=images, videos=[None],
    instruction=[psg_inst], is_query=False,
)
with torch.inference_mode():
    img_emb = model(**inputs.to(model.device)).last_hidden_state[:, -1]

print(query_emb @ img_emb.T)
```

For FlashAttention acceleration, pass `attn\_implementation="flash\_attention\_2"` when loading the model.

For evaluation on COLA, SugarCrepe\+\+, NegBench, COCO, Flickr30k, and MCMR, use the evaluation code in the [GitHub repository](https://github.com/).

## [](https://huggingface.co/Alibaba-NLP/core-reranker-2b#citation)  Citation  

If you use Core\-Embed, please cite the paper:

```bibtex
@misc{song2026core,
      title={CORE: Improving Compositional Reasoning in MLLM Embedding via Reranker Distillation}, 
      author={Tingyu Song and Mingxin Li and Yanzhao Zhang and Dingkun Long and Chu Liu and Pengjun Xie and Yilun Zhao and Shu Wu},
      year={2026},
      eprint={2609.04083},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2609.04083}, 
}
```

## [](https://huggingface.co/Alibaba-NLP/core-reranker-2b#acknowledgements)  Acknowledgements  

We thank the authors of [COLA](https://arxiv.org/abs/2305.03689), [SugarCrepe\+\+](https://arxiv.org/abs/2406.11171), and [NegBench](https://github.com/m1k2zoo/negbench) for their benchmarks, and the [Qwen3\-VL\-Embedding](https://github.com/QwenLM/Qwen3-VL-Embedding) project for the evaluation framework.
Downloads last month 145 
Safetensors[](https://huggingface.co/docs/safetensors)
Model size
2B params
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
[🙋  Ask for provider support](https://huggingface.co/spaces/huggingface/InferenceSupport/discussions/new?title=Alibaba-NLP/core-reranker-2b&description=React%20to%20this%20comment%20with%20an%20emoji%20to%20vote%20for%20%5BAlibaba-NLP%2Fcore-reranker-2b%5D(%2FAlibaba-NLP%2Fcore-reranker-2b)%20to%20be%20supported%20by%20Inference%20Providers.%0A%0A(optional)%20Which%20providers%20are%20you%20interested%20in%3F%20(Novita%2C%20Hyperbolic%2C%20Together%E2%80%A6)%0A)

## Collection including Alibaba\-NLP/core\-reranker\-2b

#### core\-emb
Collection
5 items • Updated Sep 4 
[https://huggingface.co/collections/Alibaba-NLP/core-emb](https://huggingface.co/collections/Alibaba-NLP/core-emb)

## Papers for Alibaba\-NLP/core\-reranker\-2b

#### CORE: Improving Compositional Reasoning in MLLM Embedding via Reranker Distillation
Paper • 2609.04083 • Published Sep 3 •  26 
[https://huggingface.co/papers/2609.04083](https://huggingface.co/papers/2609.04083)

#### SUGARCREPE\+\+ Dataset: Vision\-Language Model Sensitivity to Semantic and Lexical Alterations
Paper • 2406.11171 • Published Jun 17, 2024  
[https://huggingface.co/papers/2406.11171](https://huggingface.co/papers/2406.11171)

#### COLA: How to adapt vision\-language models to Compose Objects Localized with Attributes?
Paper • 2305.03689 • Published May 5, 2023 •  5 
[https://huggingface.co/papers/2305.03689](https://huggingface.co/papers/2305.03689)
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
