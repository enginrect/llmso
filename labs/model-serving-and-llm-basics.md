# 모델 서빙 입문과 LLM 실행 원리 — 실습 기록

모든 실행은 RTX 4090 24GB / WSL2 Ubuntu 24.04에서 했다 ([환경 구축 기록](environment-setup.md), [개념 정리](../model-serving-and-llm-basics.md)).

서빙 개론 파트는 개념뿐이라 실습이 없다. 실습은 전부 **LLM 실행 원리** 쪽에 몰려 있고, 교재 리포의 Jupyter 노트북 5개로 구성된다.

- 소스: https://github.com/orca3/llm-model-inference — `ch02/`
- 원래는 **Google Colab (T4)** 기준으로 작성돼 있다.
- **4090 판정: ✅ 전부 여유롭게 가능.** 사용 모델이 Qwen2.5-0.5B(가중치 약 1GB)라 VRAM이 전혀 문제되지 않는다.

---

## 개요 — 이론에서 프레임워크까지

### 실습 흐름 한눈에

```
① Transformer 내부 뜯어보기   →  모델은 어떻게 생겼나 (config, layer, attention)
② 밑바닥부터 추론 돌려보기     →  KV cache 없이 / 있이 → 성능 차이 체감
③ vLLM으로 서빙              →  ①②를 프레임워크가 대신 해주면 얼마나 빨라지나
④ 스트리밍                   →  TTFT를 어떻게 체감 개선하나
⑤ 배칭                       →  처리량을 어떻게 올리나
```

**"이론 → 스크래치 구현 → 실전 도구 → 성능 전략"** 순서다. 순서를 바꾸면 왜 vLLM이 필요한지 감이 안 온다.

---

## 실습 기록 (Labs)

### Lab 1 — Transformer 내부 들여다보기 ✅

**노트북**: [`ch02/ch2_Inside_the_Mind_of_a_Transformer.ipynb`](https://github.com/orca3/llm-model-inference/blob/main/ch02/ch2_Inside_the_Mind_of_a_Transformer.ipynb)
**모델**: Qwen2.5-0.5B

| 세부 실습 | 확인할 것 |
|---|---|
| **[실습1]** `model.config` 출력 | hidden_size, num_hidden_layers, num_attention_heads, vocab_size |
| **[실습2]** 디코더 레이어 구조 확인 | `Qwen2DecoderLayer` 안에 attention + MLP + norm이 어떻게 쌓였나 |
| **[실습2.5]** attention layer 확인 → **GQA 검증** | `num_attention_heads: 14` vs `num_key_value_heads: 2` → 7배 차이 |
| **[실습3]** BertViz로 attention 시각화 | 특정 토큰이 **어떤 토큰에 더 주목**하는지 head_view로 확인 |


#### 실행

```bash
git clone https://github.com/orca3/llm-model-inference
cd llm-model-inference/ch02
uv venv --python 3.12 .venv && source .venv/bin/activate
uv pip install torch transformers jupyter bertviz
jupyter lab
```

#### 왜 이걸 먼저 하나

[병목/최적화 노트](../serving-bottlenecks-and-optimization.md)의 **KV cache 용량 계산이 전부 이 config 값에서 나온다.**
`2 × layers × kv_heads × head_dim × 2bytes` — 이 식의 변수를 여기서 눈으로 확인해두면 이후 실습이 훨씬 수월하다.

GQA 검증은 계산으로 확인 가능하다 (hidden 896, head_dim 64):
```
MHA였다면 (KV 헤드 14개): 2 × (896 × 896) = 1,605,632 파라미터
GQA 실제  (KV 헤드  2개): 2 × (896 × 128) =   229,376 파라미터
→ 정확히 7.0배. config의 num_key_value_groups: 7 과 일치
```

---


#### 실행 기록

환경 구성 (문서의 `jupyter lab` 대신 `nbconvert --execute`로 노트북을 통째로 실행하고 출력을 보존했다.
vLLM용 CUDA 수정 3종(nvcc 13.0 정렬·심링크·CUDA_HOME)은 [environment-setup](environment-setup.md) 시도 4~6과 동일):

```shell
$ uv venv --python 3.12 --seed .venv
Using CPython 3.12.3 interpreter at: /usr/bin/python3.12
Creating virtual environment with seed packages at: .venv
 + pip==26.2.1
Activate with: source .venv/bin/activate

$ uv pip install vllm bertviz jupyter nbconvert ipykernel matplotlib accelerate tiktoken transformers_stream_generator
 + webcolors==25.10.0
 + webencodings==0.6.1
 + websocket-client==1.9.0
 + websockets==17.0.1
 + widgetsnbextension==4.0.16
 + xgrammar==0.2.3
 + yarl==1.24.5
 + z3-solver==4.15.4.0

$ nvcc 13.0 정렬 + 심링크 (셋업 노트 Step 11~12와 동일)
Installed 3 packages in 5ms
 - nvidia-cuda-crt==13.3.73
 + nvidia-cuda-crt==13.0.88
 - nvidia-cuda-nvcc==13.3.73
 + nvidia-cuda-nvcc==13.0.88
 - nvidia-nvvm==13.3.73
 + nvidia-nvvm==13.0.88
심링크 적용 완료: /home/enginrect/llm-model-inference/ch02/.venv/lib/python3.12/site-packages/nvidia/cu13/lib64
```

노트북 실행:

```shell
$ jupyter nbconvert --to notebook --execute ch2_Inside_the_Mind_of_a_Transformer.ipynb --output executed_lab1.ipynb --ExecutePreprocessor.timeout=1800
[NbConvertApp] Converting notebook ch2_Inside_the_Mind_of_a_Transformer.ipynb to notebook
[NbConvertApp] Writing 4320927 bytes to executed_lab1.ipynb
```

셀별 실행 결과 발췌 (In = 노트북 셀 소스, Out = 실제 출력, 반복·노이즈는 `...`로 생략):

```shell
# ───────── In [1] (cell 3) ─────────
!pip install --quiet transformers tiktoken transformers_stream_generator bertviz

# ───────── In [2] (cell 4) ─────────
...
# Unload models and clean up gpu memory cache
...
# ───────── In [3] (cell 6) ─────────
...
# Print all configuration parameters
...
# Architecture parameters
...
print(f"Hidden size: {config.hidden_size}")  # Size of the hidden layers
print(f"Number of layers: {config.num_hidden_layers}")  # Number of transformer blocks
print(f"Number of attention heads: {config.num_attention_heads}")  # Number of attention heads
print(f"Intermediate size: {config.intermediate_size}")  # Size of the MLP intermediate layer
...
# Tokenizer parameters
...
print(f"Vocabulary size: {config.vocab_size}")  # Size of the vocabulary
print(f"Maximum position embeddings: {config.max_position_embeddings}")  # Maximum sequence length
...
# Print model size
...
# Model-specific parameters
...
    if key not in ['architectures', 'model_type', 'torch_dtype']:
...
# Free GPU memory
...
# ───────── Out ─────────
...
vocab_size: 151936
hidden_size: 896
intermediate_size: 4864
num_hidden_layers: 24
num_attention_heads: 14
num_key_value_heads: 2
...
max_position_embeddings: 32768
...
tie_word_embeddings: True
...
# ───────── In [4] (cell 9) ─────────
...
# Load the model
...
# pprint(model.config.to_dict())
...
          if hasattr(module, 'head_dim'):
              print(f"Head dimension: {module.head_dim}")
          if hasattr(module, 'hidden_size'):
              print(f"Hidden size: {module.hidden_size}")
...
# ───────── Out ─────────
...
... (레이어 1~23 동일 구조 생략)
...
# ───────── In [5] (cell 11) ─────────
...
# Find all attention layers
...
# Analyze each attention layer
...
    if hasattr(module, 'head_dim'):
        print(f"Head dimension: {module.head_dim}")
    if hasattr(module, 'hidden_size'):
        print(f"Hidden size: {module.hidden_size}")
...
# Print model's attention-related configuration
...
# ───────── Out ─────────
...
==================================================
...
  "hidden_size": 896,
...
  "intermediate_size": 4864,
...
  "max_position_embeddings": 32768,
...
  "model_type": "qwen2",
  "num_attention_heads": 14,
  "num_hidden_layers": 24,
  "num_key_value_heads": 2,
...
  "tie_word_embeddings": true,
...
  "vocab_size": 151936
...
  head_dim: 64
...
==================================================
...
  "hidden_size": 896,
...
  "intermediate_size": 4864,
...
  "max_position_embeddings": 32768,
...
  "model_type": "qwen2",
  "num_attention_heads": 14,
  "num_hidden_layers": 24,
  "num_key_value_heads": 2,
...
  "tie_word_embeddings": true,
...
  "vocab_size": 151936
...
  head_dim: 64
...
==================================================
...
  "hidden_size": 896,
...
  "intermediate_size": 4864,
...
  "max_position_embeddings": 32768,
...
  "model_type": "qwen2",
  "num_attention_heads": 14,
  "num_hidden_layers": 24,
  "num_key_value_heads": 2,
...
  "tie_word_embeddings": true,
...
  "vocab_size": 151936
...
  head_dim: 64
...
==================================================
...
  "hidden_size": 896,
...
  "intermediate_size": 4864,
...
  "max_position_embeddings": 32768,
...
  "model_type": "qwen2",
  "num_attention_heads": 14,
  "num_hidden_layers": 24,
  "num_key_value_heads": 2,
...
  "tie_word_embeddings": true,
...
  "vocab_size": 151936
...
  head_dim: 64
...
==================================================
...
  "hidden_size": 896,
...
  "intermediate_size": 4864,
...
  "max_position_embeddings": 32768,
...
  "model_type": "qwen2",
  "num_attention_heads": 14,
  "num_hidden_layers": 24,
  "num_key_value_heads": 2,
...
  "tie_word_embeddings": true,
...
  "vocab_size": 151936
...
  head_dim: 64
...
==================================================
...
  "hidden_size": 896,
...
  "intermediate_size": 4864,
...
  "max_position_embeddings": 32768,
...
  "model_type": "qwen2",
  "num_attention_heads": 14,
  "num_hidden_layers": 24,
  "num_key_value_heads": 2,
...
  "tie_word_embeddings": true,
...
  "vocab_size": 151936
...
  head_dim: 64
...
==================================================
...
  "hidden_size": 896,
...
  "intermediate_size": 4864,
...
  "max_position_embeddings": 32768,
...
  "model_type": "qwen2",
  "num_attention_heads": 14,
  "num_hidden_layers": 24,
  "num_key_value_heads": 2,
...
  "tie_word_embeddings": true,
...
  "vocab_size": 151936
...
  head_dim: 64
...
==================================================
...
  "hidden_size": 896,
...
  "intermediate_size": 4864,
...
  "max_position_embeddings": 32768,
...
  "model_type": "qwen2",
  "num_attention_heads": 14,
  "num_hidden_layers": 24,
  "num_key_value_heads": 2,
...
  "tie_word_embeddings": true,
...
  "vocab_size": 151936
...
  head_dim: 64
...
==================================================
...
  "hidden_size": 896,
...
  "intermediate_size": 4864,
...
  "max_position_embeddings": 32768,
...
  "model_type": "qwen2",
  "num_attention_heads": 14,
  "num_hidden_layers": 24,
  "num_key_value_heads": 2,
...
  "tie_word_embeddings": true,
...
  "vocab_size": 151936
...
  head_dim: 64
...
==================================================
...
  "hidden_size": 896,
...
  "intermediate_size": 4864,
...
  "max_position_embeddings": 32768,
...
  "model_type": "qwen2",
  "num_attention_heads": 14,
  "num_hidden_layers": 24,
  "num_key_value_heads": 2,
...
  "tie_word_embeddings": true,
...
  "vocab_size": 151936
...
  head_dim: 64
...
==================================================
...
  "hidden_size": 896,
...
  "intermediate_size": 4864,
...
  "max_position_embeddings": 32768,
...
  "model_type": "qwen2",
  "num_attention_heads": 14,
  "num_hidden_layers": 24,
  "num_key_value_heads": 2,
...
  "tie_word_embeddings": true,
...
  "vocab_size": 151936
...
  head_dim: 64
...
==================================================
...
  "hidden_size": 896,
...
  "intermediate_size": 4864,
...
  "max_position_embeddings": 32768,
...
  "model_type": "qwen2",
  "num_attention_heads": 14,
  "num_hidden_layers": 24,
  "num_key_value_heads": 2,
...
  "tie_word_embeddings": true,
...
  "vocab_size": 151936
...
  head_dim: 64
...
==================================================
...
  "hidden_size": 896,
...
  "intermediate_size": 4864,
...
  "max_position_embeddings": 32768,
...
  "model_type": "qwen2",
  "num_attention_heads": 14,
  "num_hidden_layers": 24,
  "num_key_value_heads": 2,
...
  "tie_word_embeddings": true,
...
  "vocab_size": 151936
...
  head_dim: 64
...
==================================================
...
  "hidden_size": 896,
...
  "intermediate_size": 4864,
...
  "max_position_embeddings": 32768,
...
  "model_type": "qwen2",
  "num_attention_heads": 14,
  "num_hidden_layers": 24,
  "num_key_value_heads": 2,
...
  "tie_word_embeddings": true,
...
  "vocab_size": 151936
...
  head_dim: 64
...
==================================================
...
  "hidden_size": 896,
...
  "intermediate_size": 4864,
...
  "max_position_embeddings": 32768,
...
  "model_type": "qwen2",
  "num_attention_heads": 14,
  "num_hidden_layers": 24,
  "num_key_value_heads": 2,
...
  "tie_word_embeddings": true,
...
  "vocab_size": 151936
...
  head_dim: 64
...
==================================================
...
  "hidden_size": 896,
...
  "intermediate_size": 4864,
...
  "max_position_embeddings": 32768,
...
  "model_type": "qwen2",
  "num_attention_heads": 14,
  "num_hidden_layers": 24,
  "num_key_value_heads": 2,
...
  "tie_word_embeddings": true,
...
  "vocab_size": 151936
...
  head_dim: 64
...
==================================================
...
  "hidden_size": 896,
...
  "intermediate_size": 4864,
...
  "max_position_embeddings": 32768,
...
  "model_type": "qwen2",
  "num_attention_heads": 14,
  "num_hidden_layers": 24,
  "num_key_value_heads": 2,
...
  "tie_word_embeddings": true,
...
  "vocab_size": 151936
...
  head_dim: 64
...
==================================================
...
  "hidden_size": 896,
...
  "intermediate_size": 4864,
...
  "max_position_embeddings": 32768,
...
  "model_type": "qwen2",
  "num_attention_heads": 14,
  "num_hidden_layers": 24,
  "num_key_value_heads": 2,
...
  "tie_word_embeddings": true,
...
  "vocab_size": 151936
...
  head_dim: 64
...
==================================================
...
  "hidden_size": 896,
...
  "intermediate_size": 4864,
...
  "max_position_embeddings": 32768,
...
  "model_type": "qwen2",
  "num_attention_heads": 14,
  "num_hidden_layers": 24,
  "num_key_value_heads": 2,
...
  "tie_word_embeddings": true,
...
  "vocab_size": 151936
...
  head_dim: 64
...
==================================================
...
  "hidden_size": 896,
...
  "intermediate_size": 4864,
...
  "max_position_embeddings": 32768,
...
  "model_type": "qwen2",
  "num_attention_heads": 14,
  "num_hidden_layers": 24,
  "num_key_value_heads": 2,
...
  "tie_word_embeddings": true,
...
  "vocab_size": 151936
...
  head_dim: 64
...
==================================================
...
  "hidden_size": 896,
...
  "intermediate_size": 4864,
...
  "max_position_embeddings": 32768,
...
  "model_type": "qwen2",
  "num_attention_heads": 14,
  "num_hidden_layers": 24,
  "num_key_value_heads": 2,
...
  "tie_word_embeddings": true,
...
  "vocab_size": 151936
...
  head_dim: 64
...
==================================================
...
  "hidden_size": 896,
...
  "intermediate_size": 4864,
...
  "max_position_embeddings": 32768,
...
  "model_type": "qwen2",
  "num_attention_heads": 14,
  "num_hidden_layers": 24,
  "num_key_value_heads": 2,
...
  "tie_word_embeddings": true,
...
  "vocab_size": 151936
...
  head_dim: 64
...
==================================================
...
  "hidden_size": 896,
...
  "intermediate_size": 4864,
...
  "max_position_embeddings": 32768,
...
  "model_type": "qwen2",
  "num_attention_heads": 14,
  "num_hidden_layers": 24,
  "num_key_value_heads": 2,
...
  "tie_word_embeddings": true,
...
  "vocab_size": 151936
...
  head_dim: 64
...
==================================================
...
  "hidden_size": 896,
...
  "intermediate_size": 4864,
...
  "max_position_embeddings": 32768,
...
  "model_type": "qwen2",
  "num_attention_heads": 14,
  "num_hidden_layers": 24,
  "num_key_value_heads": 2,
...
  "tie_word_embeddings": true,
...
  "vocab_size": 151936
...
  head_dim: 64
...
 'num_attention_heads': 14,
...
# ───────── In [6] (cell 14) ─────────
...
    hidden_size = 896
...
    intermediate_size = 4864
    vocab_size = 151936
...
    embedding_memory = vocab_size * hidden_size * 4  # 4 bytes per float32
...
    qkv_memory = hidden_size * hidden_size * 3 * 4  # Q, K, V projections
    attention_output_memory = hidden_size * hidden_size * 4  # Output projection
...
    mlp_input_memory = hidden_size * intermediate_size * 4  # First MLP layer
    mlp_output_memory = intermediate_size * hidden_size * 4  # Second MLP layer
...
    norm_memory = hidden_size * 4  # Layer normalization parameters
...
# Calculate and print memory usage
...
# ───────── Out ─────────
...
# ───────── In [7] (cell 15) ─────────
...
# ───────── In [8] (cell 16) ─────────
# prompt: use bertviz library to visualize the attention result of the input prompt "write a short introduction about US capital city"
...
# Your input text
...
# Tokenize input and get token strings
...
# Generate outputs with attention
...
# Get attention weights
...
# Use bertviz to visualize
...
# ───────── Out ─────────
...
```

- key point: `model.config`에서 **hidden_size 896 / layers 24 / heads 14 / KV heads 2** 를 직접 확인했다. num_attention_heads(14) ÷ num_key_value_heads(2) = 7 — 문서의 GQA 7배 계산과 config 실측이 일치한다. BertViz head_view는 headless 실행이라 위젯 HTML 객체 생성까지만 확인했다(시각 확인은 브라우저 필요).


### Lab 2 — 밑바닥부터 추론 + KV Cache 효과 ✅

**노트북**: [`ch02/ch2_Workthrough_LLM_execution.ipynb`](https://github.com/orca3/llm-model-inference/blob/main/ch02/ch2_Workthrough_LLM_execution.ipynb)

| 세부 실습 | 내용 |
|---|---|
| **[실습1]** HuggingFace `pipeline`으로 텍스트 생성 | 가장 쉬운 경로. 기준선 |
| **[실습2]** 예측을 한 줄씩 뜯어보기 (**KV cache 미사용**) | 토크나이징 → forward → argmax → 다시 붙이기 루프를 직접 작성 |
| **[실습3]** **KV cache 적용** 후 비교 | `past_key_values`를 넘겨서 재계산 제거 |


#### 이 실습의 핵심

실습2는 매 스텝마다 **전체 시퀀스를 다시 forward**한다. 실습3은 직전 스텝의 K/V를 재사용한다.
길이가 늘어날수록 격차가 벌어지는 걸 직접 봐야 한다 — **지연시간과 처리량 양쪽 모두에서.**

> 이게 [서빙 엔진 실습](serving-engine-and-production-architecture.md)의 `WorkloadManager`, [최적화 실습](serving-bottlenecks-and-optimization.md)의 "decode는 산술 강도가 항상 1.00"으로 이어진다.
> **KV cache는 연산을 줄이는 대신 메모리를 먹는 거래**라는 걸 여기서 체감하는 게 목적이다.

---


#### 실행 기록

```shell
$ jupyter nbconvert --to notebook --execute ch2_Workthrough_LLM_execution.ipynb --output executed_lab2.ipynb --ExecutePreprocessor.timeout=1800 --allow-errors
```

셀별 실행 결과 발췌:

```shell
# ───────── In [1] (cell 1) ─────────
!pip install --quiet transformers tiktoken transformers_stream_generator bertviz

# ───────── In [2] (cell 2) ─────────
...
# Unload models and clean up gpu memory cache
...
# ───────── In [3] (cell 4) ─────────
...
# Initialize the text generation pipeline
...
# Define your prompt
...
# Generate text
...
# Print the generated text
...
# ───────── Out ─────────
...
# ───────── In [4] (cell 6) ─────────
...
# (1) Specify the model and load tokenizer and model
...
# (2) Define the input prompt - a text about communication history
...
Today, digital platforms allow billions of people to share messages, media, and experiences in real time. Social media, messaging apps, and video conferencing have broken down geographical barriers and created new ways of building communities. At the same time, these technologies raise important questions about privacy, information overload, and the nature of human interaction.
...
# (3) Convert (Tokenize) prompt to the input format that model understands
...
# tokenize the input prompt for the first output token
# PS: prompt is the initial input sequence for LLM generation
...
# (4) Main generation loop - generate tokens one by one
...
# Decode the entire generated sequence
...
# Free GPU memory
...
# ───────── Out ─────────
...
Today, digital platforms allow billions of people to share messages, media, and experiences in real time. Social media, messaging apps, and video conferencing have broken down geographical barriers and created new ways of building communities. At the same time, these technologies raise important questions about privacy, information overload, and the nature of human interaction.
...
# ───────── In [5] (cell 8) ─────────
...
# First chart: First bar red, others blue
...
# plt.subplot(1, 2, 1)
...
# # Second chart: Exclude the first element
# plt.subplot(1, 2, 2)
# plt.bar(range(len(times) - 1), times[1:])
# plt.xlabel("Token ID")
# plt.ylabel("Time Spent in Token Generation")
# plt.title("Token Generation Times (Excluded the initial prompt tokens)")
...
# plt.tight_layout()
# plt.show()
# ───────── Out ─────────
...
# ───────── In [6] (cell 9) ─────────
...
Today, digital platforms allow billions of people to share messages, media, and experiences in real time. Social media, messaging apps, and video conferencing have broken down geographical barriers and created new ways of building communities. At the same time, these technologies raise important questions about privacy, information overload, and the nature of human interaction.
...
# (1) Define Key/Value Cache for faster generation
...
# ───────── Out ─────────
...
Today, digital platforms allow billions of people to share messages, media, and experiences in real time. Social media, messaging apps, and video conferencing have broken down geographical barriers and created new ways of building communities. At the same time, these technologies raise important questions about privacy, information overload, and the nature of human interaction.
...
# ───────── In [7] (cell 11) ─────────
...
# Display the time cost of LLM token generation without KV Cache
...
# Display the time cost of LLM token generation with KV Cache
...
# ───────── Out ─────────
...
# ───────── In [8] (cell 14) ─────────
...
Today, digital platforms allow billions of people to share messages, media, and experiences in real time. Social media, messaging apps, and video conferencing have broken down geographical barriers and created new ways of building communities. At the same time, these technologies raise important questions about privacy, information overload, and the nature of human interaction.
...
# Free GPU memory
...
# ───────── Out ─────────
...
Today, digital platforms allow billions of people to share messages, media, and experiences in real time. Social media, messaging apps, and video conferencing have broken down geographical barriers and created new ways of building communities. At the same time, these technologies raise important questions about privacy, information overload, and the nature of human interaction.
...
How might the next wave of communication tools shape our relationships, societies, and sense of identity? Which tools will resonate the deepest while others appear vague and unwelcoming? And how might human interaction evolve alongside the seams of digital screens and living organisms?

• Check: [Source](http://www.wired.com/wiredscience/155
```

#### ⚠️ 실측에서 드러난 문제 — 원본 루프는 비교가 성립하지 않는다

위 출력을 보면 **실습2(no-cache 루프)와 실습3(KV cache 루프) 모두 1토큰 만에 EOS**가 나왔다
(프롬프트가 질문으로 끝나 multinomial 샘플링 첫 토큰이 EOS). 그래서 "0.0950s vs 0.0193s"는
KV cache 효과가 아니라 **prefill 1회씩의 비교**일 뿐이다. 교재 의도(길수록 벌어지는 격차)를 보려면
생성 길이를 강제해야 한다 → 보강 실험.

#### 보강 실험 — EOS 무시하고 200/500/1500 토큰 생성 (greedy, 웜업 포함)

`kv_cache_comparison.py` 작성 (같은 모델·프롬프트, use_cache만 토글, greedy로 출력 동일 보장):

```shell
$ python kv_cache_comparison.py
Warning: You are sending unauthenticated requests to the HF Hub. Please set a HF_TOKEN to enable higher rate limits and faster downloads.

Loading weights:   0%|          | 0/290 [00:00<?, ?it/s]
Loading weights: 100%|██████████| 290/290 [00:00<00:00, 3268.80it/s]
=== KV cache OFF ===
총 시간 (200 tokens): 3.829s
토큰당 평균: 19.15ms
초반 10토큰 평균: 66.78ms | 마지막 10토큰 평균: 17.19ms
시퀀스 최종 길이: 211 tokens

=== KV cache ON ===
총 시간 (200 tokens): 2.980s
토큰당 평균: 14.90ms
초반 10토큰 평균: 17.07ms | 마지막 10토큰 평균: 13.97ms
시퀀스 최종 길이: 211 tokens
```

```shell
$ python kv_cache_comparison.py  # max_new_tokens=500, 워밍업 추가

Loading weights:   0%|          | 0/290 [00:00<?, ?it/s]
Loading weights: 100%|██████████| 290/290 [00:00<00:00, 3467.24it/s]
=== KV cache OFF ===
총 시간 (500 tokens): 8.414s
토큰당 평균: 16.83ms
초반 10토큰 평균: 16.83ms | 마지막 10토큰 평균: 16.23ms
시퀀스 최종 길이: 511 tokens

=== KV cache ON ===
총 시간 (500 tokens): 7.474s
토큰당 평균: 14.95ms
초반 10토큰 평균: 14.78ms | 마지막 10토큰 평균: 14.92ms
시퀀스 최종 길이: 511 tokens
```

```shell
$ python kv_cache_comparison.py  # max_new_tokens=1500

Loading weights:   0%|          | 0/290 [00:00<?, ?it/s]
Loading weights: 100%|██████████| 290/290 [00:00<00:00, 3476.91it/s]
[W822 20:50:19.622805861 CUDACachingAllocator.cpp:3933] memory allocation failed with OOM on device 0 while trying to allocate 394264576 bytes (free: 0, total: 25756696576).
=== KV cache OFF ===
총 시간 (1500 tokens): 441.441s
토큰당 평균: 294.29ms
초반 10토큰 평균: 15.14ms | 마지막 10토큰 평균: 772.46ms
시퀀스 최종 길이: 1511 tokens

=== KV cache ON ===
총 시간 (1500 tokens): 20.774s
토큰당 평균: 13.85ms
초반 10토큰 평균: 20.91ms | 마지막 10토큰 평균: 13.77ms
시퀀스 최종 길이: 1511 tokens
```

| max_new_tokens | cache OFF | cache ON | 배율 |
|---|---|---|---|
| 200 (웜업 없음) | 3.83s | 2.98s | 1.29x |
| 500 (웜업 후) | 8.41s | 7.47s | 1.13x |
| **1500** | **441.4s** | **20.8s** | **21.2x** |

- key point: **KV cache 효과는 시퀀스 길이의 함수다.** 500토큰까지는 0.5B 모델의 스텝 시간이 커널 런치 오버헤드(~15ms)에 지배돼 차이가 1.1배에 그치지만, 1500토큰에서는 OFF의 토큰당 시간이 15ms → 772ms로 폭증(전체 재계산 + eager attention의 n² 행렬, 도중 CUDA allocator OOM 경고까지)해 21배 차이가 났다. 4090처럼 빠른 GPU는 짧은 시퀀스에서 KV cache 부재를 숨긴다 — 교재(T4)의 큰 배수가 로컬에서 재현되지 않은 이유.


### Lab 3 — vLLM으로 서빙 ✅

**노트북**: [`ch02/ch2_Run_LLM_With_vLLM.ipynb`](https://github.com/orca3/llm-model-inference/blob/main/ch02/ch2_Run_LLM_With_vLLM.ipynb)

앞의 실습 2에서 **손으로 만든 것**(토크나이징 → 루프 → KV cache 관리)을 vLLM이 전부 대신한다.
교재 실습 결과 기준 **약 17배** 차이가 난다 (1.12s vs 19.58s).


#### ⚠️ 버전 주의

노트북이 `vllm==0.6.6.post1`로 고정돼 있다. 2025년 7월 시점에 이미 Colab 런타임 호환성 문제로 조정이 필요했던 버전이다.

**4090 + WSL2에서는 핀을 풀고 최신 vLLM을 쓰는 걸 권한다.**
`LLM()` / `SamplingParams` 기본 API는 안정적이라 노트북 코드가 거의 그대로 동작한다.

```bash
uv pip install vllm          # 버전 핀 없이
```

만약 최신 버전에서 API가 깨지면 그때 핀을 맞추면 된다. 옛 버전을 4090(sm89)에 억지로 맞추는 것보다 낫다.

---


#### 실행 기록

```shell
$ jupyter nbconvert --to notebook --execute ch2_Run_LLM_With_vLLM.ipynb --output executed_lab3.ipynb --ExecutePreprocessor.timeout=1800 --allow-errors
```

셀별 실행 결과 발췌 (vLLM 0.27.1 — 문서 권고대로 핀을 풀고 최신 사용, `LLM()`/`SamplingParams` API는 그대로 동작했다):

```shell
# ───────── In [1] (cell 1) ─────────
import torch
import gc
...
# Unload models and clean up gpu memory cache
...
# ───────── In [2] (cell 2) ─────────
...
# ───────── Out ─────────
...
# ───────── In [3] (cell 4) ─────────
...
# Load model with vLLM.
...
# Define the prompt.
...
# Create sampling parameters.
...
# Time the model generation.
...
# Print the results.
...
# ───────── Out ─────────
...
INFO 08-22 20:52:01 [scheduler.py:242] Chunked prefill is enabled with max_num_batched_tokens=8192.
...
(EngineCore pid=20075) INFO 08-22 20:52:23 [weight_utils.py:867] Filesystem type for checkpoints: EXT4. Checkpoint size: 0.92 GiB. Available RAM: 26.94 GiB.
...
(EngineCore pid=20075) INFO 08-22 20:52:25 [model_runner.py:329] Model loading took 0.93 GiB and 6.206722 seconds
...
(EngineCore pid=20075) INFO 08-22 20:52:36 [gpu_worker.py:563] Available KV cache memory: 20.52 GiB
(EngineCore pid=20075) INFO 08-22 20:52:36 [kv_cache_utils.py:2235] GPU KV cache size: 1,793,232 tokens
...
(EngineCore pid=20075) INFO 08-22 20:52:39 [model_runner.py:791] Graph capturing finished in 2 secs, took 0.00 GiB
(EngineCore pid=20075) INFO 08-22 20:52:39 [gpu_worker.py:789] Free memory on device (22.45/23.99 GiB) on startup. Desired GPU memory utilization is (0.92, 22.07 GiB). Actual usage is 1.23 GiB for consumed memory (weights + non-torch), 0.31 GiB for peak activation, and 0.0 GiB for CUDAGraph memory. Replace gpu_memory_utilization config with `--kv-cache-memory=21878056530` (20.38 GiB) to fit into requested memory, or `--kv-cache-memory=22286560768` (20.76 GiB) to fully utilize gpu memory. Current kv cache memory in use is 20.52 GiB.
...
(EngineCore pid=20075) INFO 08-22 20:53:33 [core.py:348] init engine (profile, create kv cache, warmup model) took 67.80 s (compilation: 8.08 s)
...
# ───────── In [4] (cell 5) ─────────
...
# ───────── In [5] (cell 6) ─────────
...
# ───────── Out ─────────
ERROR: SyntaxError: invalid syntax (2268419272.py, line 1)
...
SyntaxError: invalid syntax
...
# ───────── In [6] (cell 7) ─────────
# --- Basic Model Serving (transformers) ---
...
# Load model and tokenizer
...
# Create the pipeline.
...
# ───────── Out ─────────
...
# ───────── In [7] (cell 8) ─────────
...
# ───────── Out ─────────

Latency difference: 8.42 seconds
```

- key point: **vLLM 1.16s vs HF pipeline 9.58s = 8.3배** (교재 17배). 단 원 노트북 코드가 HF 쪽 측정에 모델·토크나이저 로딩을 포함하므로 순수 추론 비교가 아니다. 격차가 교재보다 작은 것은 4090에서 HF eager 경로도 충분히 빨라서다. vLLM 로그에서 chunked prefill(8192)과 prefix caching이 **기본 활성**임을 확인 — [최적화 실습](serving-bottlenecks-and-optimization.md) 도전과제의 전제와 일치.


### Lab 4 — 스트리밍 ✅

**노트북**: [`ch02/ch2_Streaming.ipynb`](https://github.com/orca3/llm-model-inference/blob/main/ch02/ch2_Streaming.ipynb)
**추가 예제**: [`ch02/streaming.py`](https://github.com/orca3/llm-model-inference/blob/main/ch02/streaming.py)

**해결하는 문제**: 응답 전체가 완성될 때까지 사용자가 빈 화면을 본다 → 체감 UX 최악.
**해결 방식**: 토큰이 생성되는 즉시 하나씩(또는 작은 청크로) 흘려보낸다.

확인할 것: **총 생성 시간은 그대로인데 TTFT만 줄어든다는 것.** 스트리밍은 처리량 최적화가 아니라 **체감 지연시간** 최적화다.

---


#### 실행 기록 — 1차: 원본 그대로 (실패 2건 재현)

```shell
$ jupyter nbconvert --to notebook --execute ch2_Streaming.ipynb --output executed_lab4.ipynb --ExecutePreprocessor.timeout=1800 --allow-errors
```

원본 노트북에서 에러 2건이 그대로 재현됐다:

```shell
ERROR: TypeError: AsyncEngineArgs.__init__() got an unexpected keyword argument 'disable_log_requests'
  → vLLM 0.27에서 제거된 인자 (0.11+부터 로깅 기본 비활성)

ERROR: NameError: name 'requtest_id' is not defined
  → 원본 노트북 자체의 오타 (requtest_id ≠ request_id)
```

#### 2차: 수정본 실행 (편차 2건 — 인자 제거, 오타 수정)

```shell
$ python - <<'EOF'   # ch2_Streaming_fixed.ipynb 생성
수정 1: cell 6 — disable_log_requests=True 줄 삭제 (vLLM 0.27에서 제거된 인자)
수정 2: cell 10 — requtest_id → request_id 오타 수정
EOF
$ jupyter nbconvert --to notebook --execute ch2_Streaming_fixed.ipynb --output executed_lab4_fixed.ipynb --ExecutePreprocessor.timeout=1800 --allow-errors
```

> **⚠️ 왜 실패했나** — 2건의 원인이 다르다.
> ① `disable_log_requests`: **버전 차이.** 노트북은 vLLM 0.6.6(2025.07 Colab 기준)으로 작성됐고,
> 이 환경은 0.27.1이다. 가이드 시점과 실행 시점의 버전 격차가 원인.
> ② `requtest_id`: **리포 자체의 오타.** 어떤 환경에서 돌려도 실패한다 — 환경 탓이 아니다.

수정본은 끝까지 실행됐다. 단 노트북의 `stop=["\n"]` + temperature 0.0 조합 때문에 첫 토큰이 곧바로
stop에 걸려 **가시적 출력 없이 종료** (chunk 1회 방출, finish_reason=stop, token_ids=[151643]=EOS).
스트리밍 메커니즘 동작은 확인됐지만 체감 데모가 안 된다 → 보강.

#### 보강 실험 — streaming_fixed.py (stop 제거 + TTFT 측정)

리포의 `streaming.py`를 0.27에 맞게 수정: ①`disable_log_requests` 제거 ②`stop=["\n"]` 제거
③TTFT 측정 추가 ④**`__main__` 가드 추가** — vLLM 0.27은 spawn 멀티프로세싱이라 모듈 레벨에서
엔진을 만들면 `RuntimeError: An attempt has been made to start a new process before ... bootstrapping phase`가
난다 (가드 없이 실행해 실제로 재현했다. Jupyter에서는 안 터지고 스크립트에서만 터지는 함정).

```shell
$ python streaming_fixed.py
 The capital of the United States is Washington, D.C.

--- 측정 ---
TTFT (첫 청크 도착): 29.5ms
총 생성 시간: 0.097s / 13 tokens (13 chunks)
스트리밍이 없다면 사용자는 첫 글자까지 0.097s를 기다린다. 스트리밍은 그걸 29.5ms로 줄인다.
```

- key point: 13토큰이 **13개 청크로 증분 도착** — 토큰 단위 스트리밍 확인. TTFT 29.5ms vs 총 97ms. 총 생성 시간은 그대로고 **첫 응답까지의 대기만 줄어든다**는 문서의 명제가 수치로 확인됐다. 부가 발견 2건: vLLM 0.27에서 `disable_log_requests` 제거, 스크립트 실행 시 `__main__` 가드 필수.


### Lab 5 — 배칭 ✅

**노트북**: [`ch02/ch2_Batching.ipynb`](https://github.com/orca3/llm-model-inference/blob/main/ch02/ch2_Batching.ipynb)

여러 prompt를 묶어 한 번에 처리해 GPU 활용률을 올린다.

> **예습 포인트**: 배칭의 진짜 이유는 "GPU가 놀아서"가 아니라
> **모델 가중치를 HBM에서 SRAM으로 읽는 횟수를 줄이기 때문**이다.
> 요청 3개를 따로 처리하면 가중치 전송 3회, 묶으면 1회.
> 이 실습에서 배치 크기를 키우며 토큰/초를 재보면 그 효과가 그대로 보인다.

---


#### 실행 기록

```shell
$ jupyter nbconvert --to notebook --execute ch2_Batching.ipynb --output executed_lab5.ipynb --ExecutePreprocessor.timeout=1800 --allow-errors
```

셀별 실행 결과 발췌:

```shell
# ───────── In [1] (cell 1) ─────────
!pip install --quiet vllm transformers tiktoken

# ───────── In [2] (cell 2) ─────────
...
# Unload models and clean up gpu memory cache
...
# ───────── In [3] (cell 3) ─────────
# Define the prompt.
...
# ───────── In [4] (cell 4) ─────────
...
# Load model with vLLM
...
# ───────── Out ─────────
...
INFO 08-22 20:58:22 [scheduler.py:242] Chunked prefill is enabled with max_num_batched_tokens=8192.
...
(EngineCore pid=22621) INFO 08-22 20:58:35 [weight_utils.py:867] Filesystem type for checkpoints: EXT4. Checkpoint size: 0.92 GiB. Available RAM: 27.69 GiB.
...
(EngineCore pid=22621) INFO 08-22 20:58:35 [model_runner.py:329] Model loading took 0.93 GiB and 2.092727 seconds
...
(EngineCore pid=22621) INFO 08-22 20:58:44 [gpu_worker.py:563] Available KV cache memory: 20.52 GiB
(EngineCore pid=22621) INFO 08-22 20:58:44 [kv_cache_utils.py:2235] GPU KV cache size: 1,793,232 tokens
...
(EngineCore pid=22621) INFO 08-22 20:58:47 [model_runner.py:791] Graph capturing finished in 2 secs, took 0.00 GiB
(EngineCore pid=22621) INFO 08-22 20:58:47 [gpu_worker.py:789] Free memory on device (22.45/23.99 GiB) on startup. Desired GPU memory utilization is (0.92, 22.07 GiB). Actual usage is 1.23 GiB for consumed memory (weights + non-torch), 0.31 GiB for peak activation, and 0.0 GiB for CUDAGraph memory. Replace gpu_memory_utilization config with `--kv-cache-memory=21878056530` (20.38 GiB) to fit into requested memory, or `--kv-cache-memory=22286560768` (20.76 GiB) to fully utilize gpu memory. Current kv cache memory in use is 20.52 GiB.
...
(EngineCore pid=22621) INFO 08-22 20:58:49 [core.py:348] init engine (profile, create kv cache, warmup model) took 13.44 s (compilation: 7.49 s)
...
# ───────── In [5] (cell 5) ─────────
...
# Prompts for batch generation, 4 input sequences
...
# process four input sequences (prompts) together in one batch
...
# process prompt one by one
...
# ───────── Out ─────────
...
vLLM generation time for 4 prompts one by one: 1.2743 seconds
```

#### 보강 실험 — 배치 크기 스윕 (Self-Check: 1 → 4 → 16)

`batch_sweep.py` 작성 (같은 프롬프트 복제, max_tokens=200 고정, ignore_eos로 출력량 통제, 웜업 1회):

```shell
$ python batch_sweep.py
batch |    총 시간(s) |      총 출력 토큰 |     처리량(tok/s) |      요청당 시간(s)
----------------------------------------------------------------------
    1 |      0.441 |          200 |          453.4 |          0.441
    4 |      0.575 |          800 |         1390.7 |          0.144
   16 |      0.607 |         3200 |         5268.0 |          0.038
```

- key point: 노트북 원본 — 4개 배치 0.5991s vs 순차 1.2743s = **2.13배**. 스윕 — batch 1→16에서 총 시간은 0.44s→0.61s로 거의 그대로인데 처리량은 453→5,268 tok/s로 **11.6배**. 배치를 키워도 벽시계 시간이 안 늘어나는 것이 "가중치 HBM 전송 1회를 여러 요청이 나눠 쓴다"는 명제의 직접 증거다. decode가 memory-bound 구간에 있는 한 배칭은 공짜 점심에 가깝다.


## 도전과제 (선택)

노션에 제시된 11개. 4090 기준 난이도/필요환경을 붙였다.

| # | 과제 | 환경 | 판정 |
|---|---|---|---|
| 1 | [Branch Education] CPU 동작 vs GPU 동작 영상 비교 정리 | 없음 | ✅ 정리형 |
| 2 | *Attention Is All You Need* 학습 후 주요 동작 정리 | 없음 | ✅ 정리형 |
| 3 | [Transformer Explainer](https://poloclub.github.io/transformer-explainer/) 로 각 동작 과정 정리 | 브라우저 | ✅ |
| 4 | GQA & MQA 영상 학습 후 동작 정리 | 없음 | ✅ 실습1과 연결 |
| 5 | **ipynb 실습을 로컬 PC 주피터에 설치 후 실행** | WSL2 | ✅ **이 문서가 그 과제다** |
| 6 | *Inference Engineering(2026).pdf* 에서 발췌 정리 | 없음 | ✅ 정리형 |
| 7 | *'밑바닥부터 만들면서 배우는 LLM'* 실습으로 추론/학습 | WSL2 | ⚠️ 학습(training)까지 하면 시간 소요 큼 |
| 8 | GPT 미니멀 구현체(파이썬 builtin만) 따라해보기 | CPU | ✅ GPU 불필요 |
| 9 | KV Cache / Prefill+Decode / P/D Disaggregation 정리 + vLLM·SGLang 실습 | WSL2 | ⚠️ **P/D 분리는 GPU 2장 이상 필요** — 개념 정리만 |
| 10 | FlashAttention, PagedAttention 등 최적화 기능 정리 + 실습 | WSL2 | ⚠️ FA2까지만. **FA3는 Hopper 전용** |
| 11 | Streaming/Batch Serving 동작 가능 이유와 내부 동작 정리 | 없음 | ✅ 실습4·5와 연결 |

### 4090에서 걸리는 것

- **도전과제 9의 P/D Disaggregation**: prefill 담당 GPU와 decode 담당 GPU를 분리하는 구조라 **최소 2장**이 필요하다. 1장으로는 개념 정리까지만.
- **도전과제 10의 FlashAttention 3**: sm90(Hopper) 전용. 4090은 FA2까지다. FlashInfer의 Ada 커널은 사용 가능.

---

## 정리

1. **GQA를 config에서 실측했다** — Qwen2.5-0.5B: hidden 896, layers 24, Q heads 14 vs KV heads 2 (7:1). KV cache 계산식 `2 × layers × kv_heads × head_dim × dtype`의 변수가 전부 여기서 나온다.
2. **KV cache 효과는 시퀀스 길이의 함수다** — 1500토큰 생성에서 OFF 441.4s vs ON 20.8s (21.2배). 500토큰 이하에서는 4090의 속도가 재계산 비용을 숨겨 1.1배에 그쳤다. OFF의 토큰당 시간은 15ms→772ms로 길이에 따라 폭증했다.
3. **vLLM vs HF pipeline: 8.3배** (1.16s vs 9.58s, 모델 로드 포함 조건). 교재 17배보다 작은 것은 4090에서 baseline 자체가 빨라서다.
4. **스트리밍은 TTFT 최적화다** — 13토큰이 13청크로 증분 도착, TTFT 29.5ms vs 총 97ms. 총 생성 시간은 변하지 않는다.
5. **배칭은 처리량 최적화다** — batch 1→16에서 총 시간 거의 불변(0.44→0.61s), 처리량 453→5,268 tok/s (11.6배). 가중치 전송 상각의 직접 증거.
6. **vLLM 0.27 이식 비용 실측** — `disable_log_requests` 인자 제거됨, 스크립트 실행 시 `__main__` 가드 필수(spawn), 원본 노트북 오타 1건(`requtest_id`). `LLM()`/`SamplingParams` 핵심 API는 그대로 동작.

---

## Self-Check

- [x] WSL2 + Ubuntu에서 `nvidia-smi`로 4090 인식 확인 (사전 준비 — [environment-setup](environment-setup.md))
- [x] `ch02` 가상환경 구성, 노트북 5개 실행 (Lab 1~5 — nbconvert로 실행)
- [x] `model.config` 값으로 **KV cache 토큰당 크기 직접 계산**해보기 (Lab 1 — 2×24×2×64×2 = 12KiB/token, config 실측값과 일치)
- [x] KV cache ON/OFF 생성 시간 비교 (Lab 2 — 1500토큰에서 21.2배)
- [x] HuggingFace vs vLLM 속도 비교 (교재는 17배) (Lab 3 — 실측 8.3배, 모델 로드 포함 조건)
- [x] 배치 크기를 1 → 4 → 16으로 올리며 토큰/초 변화 기록 (Lab 5 — 453 / 1,391 / 5,268 tok/s)

