# 서빙 엔진 해부와 프로덕션 서빙 아키텍처 — 실습 기록

모든 실행은 RTX 4090 24GB / WSL2 Ubuntu 24.04 + Docker Desktop에서 했다 ([환경 구축 기록](environment-setup.md), [개념 정리](../serving-engine-and-production-architecture.md)).

실습 섹션은 2개고, 성격이 서로 다르다.

| | 서빙 엔진 | 프로덕션 아키텍처 |
|---|---|---|
| 성격 | **서빙 엔진을 밑바닥부터 직접 만든다** | 만들어진 것을 **프로덕션에 어떻게 얹나** |
| 실습 | 6개 (`ch03/`) | Bedrock, Knowledge Agent, RayService |
| GPU | **거의 불필요** (실습4만 필요) | 옵션 실습에서만 |
| 외부 의존 | 없음 | **AWS 계정 / OpenAI API 키** |

**4090 판정: 서빙 엔진은 ✅ 전부 가능. 프로덕션 파트는 ⚠️ 절반이 외부 계정에 묶여 있다.**

---

# Part 1. 서빙 시스템 직접 만들기

- 소스: https://github.com/orca3/llm-model-inference — `ch03/`
- 노션 명시: **"로컬 PC로 실습 가능 — GPU 없이 실습 가능, Docker로 Triton 실행"**
- 구조: `ch03/single_model_llm_serving/` (실습1~4), `ch03/multi_model_serving/` (실습5~6)

> **목표는 vLLM을 배우는 게 아니다.** 프레임워크가 숨겨놓은 것 — 요청 추적, 배칭, 스트리밍,
> 프로세스 격리, 라우팅 — 을 손으로 만들어보고, 마지막에 "vLLM은 이걸 20줄로 끝낸다"를 체감하는 게 목적이다.

## 사전 준비

```bash
cd ~/llm-model-inference/ch03/single_model_llm_serving
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
```

`requirements.txt` 핵심:
```
fastapi==0.115.12   uvicorn==0.24.0   pydantic==2.11.5
transformers==4.52.4   torch==2.7.0   numpy==1.26.4
pytest==8.4.0   httpx==0.27.0   pytest-asyncio==1.0.0
```

### ⚠️ 두 가지 주의점

1. **torch/vLLM 버전 충돌**: 리포가 `torch==2.7.0` + `vllm==0.9.0.1`을 고정하는데, vLLM 설치 시 자기가 원하는 torch를 끌어오려 한다. 설치 로그를 확인하고, 충돌 나면 vLLM을 나중에 별도로 설치한다.
2. **모델을 두 번 로드한다**: `LLMEngine.__init__`이 `facebook/opt-125m`을 **transformers 기반 ModelExecutor(별도 프로세스)** 와 **vLLM 엔진** 양쪽에 각각 로드한다. 두 경로를 비교 학습시키려는 의도지만, **기동 시 GPU 메모리를 이중으로 쓴다.** 4090에서는 opt-125m이 워낙 작아 문제없다.

---

## 실습 기록 (Labs)

### Lab 1 — 서버 기동, 단일 요청 ✅

```bash
python main.py              # 포트 8000
```

프로세스를 확인해보는 게 이 실습의 핵심이다:
```bash
ps -ef --forest | grep "python main.py"
```
부모 1개 + **자식 2개**가 뜬다. uvicorn/FastAPI 메인이 부모고, ModelExecutor와 ModelWorker가 자식이다.

요청:
```bash
curl -s -X POST http://localhost:8000/basic_generate \
  -H "Content-Type: application/json" \
  -d '{"prompt": "Hello, I am"}' | jq
```

**확인할 것**: 왜 CPU 작업(API 서버)과 GPU 작업(모델 실행)을 **별도 프로세스로 격리**했는가.
Python GIL 때문이다. 한 프로세스에 두면 모델 forward가 이벤트 루프를 막는다.

**GPU 판정: ✅ 불필요.** CPU만으로도 opt-125m은 돈다.

---


#### 실행 기록

venv 구성 (핀 그대로 — torch 2.7.0 + vllm 0.9.0.1이 충돌 없이 설치됐다. 문서의 주의점 1은 이 조합에선 발생하지 않았다):

```shell
$ uv venv --python 3.12 --seed venv && uv pip install -r requirements.txt
Using CPython 3.12.3 interpreter at: /usr/bin/python3.12
Creating virtual environment with seed packages at: venv
 + pip==26.2.1
Activate with: source venv/bin/activate
 + transformers==4.52.4
 + triton==3.3.0
 + truststore==0.10.4
 + typer==0.27.1
 + typing-extensions==4.16.0
 + typing-inspection==0.4.4
 + urllib3==2.7.0
 + uvicorn==0.24.0
 + uvloop==0.22.1
 + vllm==0.9.0.1
 + watchfiles==1.2.0
 + websockets==17.0.1
 + xformers==0.0.30
 + xgrammar==0.1.19
 + yarl==1.24.5
```

서버 기동 후 첫 요청 — **리포의 잠재 버그 발견**:

```shell
$ python main.py   # 포트 8000, 백그라운드
$ curl -s -X POST http://localhost:8000/basic_generate -H "Content-Type: application/json" -d '{"prompt": "Hello, I am"}' | jq
(응답 없음 — 워커 프로세스가 죽어 요청이 영원히 걸린다)
```

서버 로그의 원인 traceback:

```shell
Traceback (most recent call last):
  File "/usr/lib/python3.12/multiprocessing/process.py", line 314, in _bootstrap
    self.run()
...
    return torch.embedding(weight, input, padding_idx, scale_grad_by_freq, sparse)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
RuntimeError: Expected all tensors to be on the same device, but found at least two devices, cpu and cuda:0! (when checking argument for argument index in method wrapper_CUDA__index_select)
```

**원인**: `ModelManager.load_model()`은 모델을 CPU에 두는데(`.to(device)` 없음), `ModelWorker.generate()`는
입력만 `self.device`(cuda)로 옮긴다. **GPU가 없는 환경(노션 전제 "GPU 없이 실습 가능")에선 둘 다 CPU라 안
터지고, GPU 머신에서만 터지는 버그다.** 수정(편차): `ModelWorker.__init__`에 `self.model = self.model.to(self.device)` 1줄 추가.

> **⚠️ 왜 실패했나** — **가이드 전제 환경과의 차이.** 노션은 "GPU 없이 실습 가능"이라 안내했고,
> 리포도 CPU 환경에서만 검증된 것으로 보인다. CPU에선 모델도 입력도 CPU라 안 터지고,
> GPU 머신에서만 device 불일치가 발현한다. 가이드가 틀린 게 아니라 **가이드의 검증 범위 밖 환경**이었던 것.

패치 후 재기동 → 성공:

```shell
$ curl -s -X POST http://localhost:8000/basic_generate -H "Content-Type: application/json" -d '{"prompt": "Hello, I am"}' | jq
{
  "generated_text": "Hello, I am a student at the University of California, Berkeley. I am a graduate student in the Department of Psychology. I am a graduate student in the Department of Psychology. I am a graduate student in the Department of Psychology. I am a graduate student in the"
}
```

프로세스 계층 (부모 1 + 자식 2 — 문서 그대로):

```shell
$ ps -ef --forest | grep "main.py" (요약: 서버 프로세스 계층만)
enginre+   24625   24622 13 21:04 ?        00:00:06  |           \_ python main.py
enginre+   24664   24625 11 21:04 ?        00:00:05  |               \_ python main.py
enginre+   24790   24625 33 21:04 ?        00:00:13  |               \_ python main.py

$ nvidia-smi --query-gpu=memory.used --format=csv,noheader
23035 MiB
```

- key point: uvicorn 부모(24625) + 자식 2개(24664, 24790) 구조 확인 — API 서버와 모델 실행을 프로세스로 격리한 형태. vLLM 0.9 엔진이 붙으면서 VRAM 23,035MiB 점유(기본 gpu_memory_utilization 0.9). 그리고 **이 리포는 CPU 환경에서만 검증된 코드였다** — device 불일치 버그가 GPU 머신에서만 발현.


### Lab 2 — 배칭 ✅

```bash
curl -s -X POST http://localhost:8000/generate \
  -H "Content-Type: application/json" \
  -d '{"prompts": ["Hello, I am", "The weather is", "Once upon a time"]}' | jq
```

**확인할 것**: `WorkloadManager`가 요청들을 어떻게 모으고, 응답을 **원래 요청자에게 다시 되돌려주는지**(Sequence ID 복원).
배치로 묶는 순간 "누가 보낸 요청인지"가 사라지기 때문에, 이걸 추적하는 자료구조가 필요해진다. 이게 서빙 엔진의 본질적인 복잡도다.

로그를 같이 본다:
```bash
tail -f server_run.log
```

---


#### 실행 기록

```shell
$ curl -s -X POST http://localhost:8000/generate -H "Content-Type: application/json" -d '{"prompts": ["Hello, I am", "The weather is", "Once upon a time"]}' | jq
{
  "generated_texts": [
    "Hello, I am a student at the University of California, Berkeley. I am a graduate student in the Department of Psychology. I am a graduate student in the Department of Psychology. I am a graduate student in the Department of Psychology. I am a graduate student in the",
    "The weather is, of course, a factor in the weather.\n\nThe weather is a factor in the weather.\n\nThe weather is a factor in the weather.\n\nThe weather is a factor in the weather.\n\nThe weather is a factor",
    "Once upon a time, I was a student at the University of California, Berkeley. I was a student at the University of California, Berkeley. I was a student at the University of California, Berkeley. I was a student at the University of California, Berkeley. I"
  ]
}
```

- key point: 프롬프트 3개가 배치로 처리된 뒤 **각자 원래 순서의 응답으로 정확히 복원**됐다. opt-125m의 출력 품질(반복 루프)은 낮지만 그건 모델 문제고, 여기서 확인할 것은 Sequence ID 기반 라우팅이 동작한다는 사실이다.


### Lab 3 — 스트리밍 + 배칭 동시 ✅

```bash
curl -N -H "Accept: text/event-stream" \
  -X POST http://localhost:8000/generate_stream \
  -H "Content-Type: application/json" \
  -d '{"prompt": "Once upon a time"}'
```

**동시에 2개 요청**을 날려서 각 토큰이 **자기 client_stream(이벤트 큐)로 정확히 라우팅**되는지 확인하는 게 진짜 포인트다.

> 배칭과 스트리밍은 언뜻 상충한다 — 배칭은 "모아서 한 번에", 스트리밍은 "나오는 즉시 하나씩".
> 이 둘이 공존하려면 **배치 결과를 요청별 큐로 분배하는 계층**이 필요하다. 그걸 여기서 만든다.

---


#### 실행 기록

동시 2건을 프롬프트를 다르게 해 요청하고, 스트림 출처를 [A]/[B]로 태깅했다:

```shell
$ 동시 요청 2건: curl -N .../generate_stream ("Once upon a time" | "The capital of France is")
[A] data: {"token": "ical", "sequence_id": "105e7cb4-a89f-4057-a497-c0ceccf4565f"}
[A] 
...
[B] 
[B] data: {"token": " million", "sequence_id": "d5e6e93f-82fc-40d1-9f50-783c680c56a9"}
[B]
```

토큰 수·인터리빙 분석 (위 파일 기준):

```shell
$ grep -c "^\[A\] data" → 21    $ grep -c "^\[B\] data" → 21
$ awk '{{print $1}}' | uniq -c   →   42 [A]  /  42 [B]   (A 전부 → B 전부, 인터리빙 없음)
sequence_id: A=105e7cb4-...  B=d5e6e93f-...  (두 개로 정확히 분리)
```

- key point: 관찰 3건. ①각 토큰이 자기 sequence_id를 달고 자기 클라이언트 스트림으로만 흘러갔다 — 배치→요청별 큐 분배 계층이 동작한다. ②그러나 **A 42줄이 모두 끝난 뒤 B 42줄이 시작** — 두 요청이 한 배치로 합쳐지지 않고 순차 처리됐다. ③스트리밍 경로(`generate_forward_batch`)는 매 스텝 `use_cache=False`로 독립 샘플링이라 A 출력이 비문("ical", "iew", …)이다 — 상태(past_key_values)를 잇지 않는 나이브 구현의 한계.


### Lab 4 — vLLM으로 배치 서빙 ✅ *(GPU 필요)*

```bash
curl -s -X POST http://localhost:8000/generate_vllm \
  -H "Content-Type: application/json" \
  -d '{"prompts": ["Hello, I am", "The weather is", "Once upon a time"]}' | jq
```

검증 포인트: `generated_texts` 배열 길이가 3.

**이 실습이 서빙 엔진 파트의 결론이다.** 실습1~3에서 수백 줄로 만든 것을 vLLM은 몇 줄로 대체한다.

**4090 판정: ✅ 가능. 단 WSL2 필수** (vLLM은 Windows 네이티브 미지원).

> 노션에 기록된 관찰 하나: **Uvicorn이 단일 이벤트 루프인데 `generate_vllm`이 `await` 없는 동기 호출**이라,
> 동시 요청 2개가 실제로는 순차 처리됐다. 프레임워크를 써도 **감싸는 쪽을 잘못 만들면 병렬성이 죽는다**는 좋은 예시다. 직접 재현해볼 만하다.

---


#### 실행 기록

```shell
$ curl -s -X POST http://localhost:8000/generate_vllm -H "Content-Type: application/json" -d '{"prompts": ["Hello, I am", "The weather is", "Once upon a time"]}' | jq
{
  "generated_texts": [
    " new to this.  What does a \"Vaccine passport\" look like?  I'm",
    " awful, but I have some pretty good friends who like to go to the beach. I live in",
    ", the EU had a reputation for being a fair and impartial country. They were very well-respected"
  ]
}
```

노션 관찰("동기 호출이라 동시 요청이 순차 처리") 재현 시도:

```shell
$ 단일 요청 시간 측정 후, 동시 2건 시간 측정 (generate_vllm 동기 호출 병렬성 검증)
--- 단일 1건 ---

real	0m0.189s
user	0m0.003s
sys	0m0.000s
--- 동시 2건 ---

real	0m0.112s
user	0m0.005s
sys	0m0.000s
```

- key point: `generated_texts` 배열 길이 3 확인 — 실습 1~3에서 수백 줄로 만든 것을 vLLM 경로는 엔드포인트 하나로 대체한다. 동기 호출 병렬성 문제는 **재현 판별 불가** — opt-125m가 4090에서 너무 빨라(요청당 ~100ms) 순차/병렬 차이가 노이즈에 묻힌다. 동시 2건(0.112s)이 단일(0.189s)보다 빠르게 나온 것이 그 증거다. 격차를 보려면 생성이 초 단위인 모델이 필요하다.


### Lab 5 — 멀티 모델 서빙 ✅

```bash
cd ../multi_model_serving
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
python -m app.server          # 포트 8001
```

모델 3종이 등록돼 있다 (프레임워크가 서로 다르다는 게 핵심):

| 용도 | 모델 | framework |
|---|---|---|
| sentiment | `distilbert-base-uncased-finetuned-sst-2-english` | transformers |
| spam | (transformers 계열) | transformers |
| image | `pytorch/vision:mobilenet_v2` | torchvision |

```bash
curl -s http://localhost:8001/models | jq

curl -X POST http://localhost:8001/predict \
  -H "Content-Type: application/json" \
  -d '{"model_id": "550e8400-e29b-41d4-a716-446655440000", "input_data": "I love this"}'
```


#### ⚠️ 반드시 curl을 **두 번** 실행할 것

**첫 호출은 모델(가중치) 로딩에서 끝난다. 두 번째 호출에서야 결과가 리턴된다.**
이게 바로 **cold start**다. 노션에도 명시된 함정이니 "고장났나?" 하고 넘어가지 말 것.

#### 확인할 것

- `ModelManager`의 **LRU 캐시** — `/models`를 호출 사이사이에 계속 찍어보면 캐시에 뭐가 올라가고 뭐가 빠지는지 보인다
- `ModelEngine`의 **Factory 패턴** — `framework` 값(`transformers` / `torchvision` / `triton`)에 따라 워커 서브클래스가 분기된다
- `/predict`가 **얇은 라우팅 레이어**라는 것 — 모델이 어떤 프레임워크인지 전혀 모른다. 대신 전처리/후처리 책임이 클라이언트로 넘어간다

**GPU 판정: ✅ 불필요.**

---


#### 실행 기록

```shell
$ curl -s http://localhost:8001/models | jq
{
  "available_models": {
    "550e8400-e29b-41d4-a716-446655440000": {
      "id": "550e8400-e29b-41d4-a716-446655440000",
      "name": "distilbert-base-uncased-finetuned-sst-2-english",
      "type": "text",
      "framework": "transformers",
      "version": "1.0.0",
      "description": "Sentiment analysis model"
    },
    "6ba7b810-9dad-11d1-80b4-00c04fd430c8": {
      "id": "6ba7b810-9dad-11d1-80b4-00c04fd430c8",
      "name": "mrm8488/bert-tiny-finetuned-sms-spam-detection",
      "type": "text",
      "framework": "transformers",
      "version": "1.0.0",
      "description": "Spam detection model"
    },
    "7c9e6679-7425-40de-944b-e07fc1f90ae7": {
      "id": "7c9e6679-7425-40de-944b-e07fc1f90ae7",
      "name": "pytorch/vision:mobilenet_v2",
      "type": "image",
      "framework": "torchvision",
      "version": "1.0.0",
      "description": "Image classification model"
    },
    "8ba7b810-9dad-11d1-80b4-00c04fd430c9": {
      "id": "8ba7b810-9dad-11d1-80b4-00c04fd430c9",
      "name": "densenet_onnx",
      "type": "image",
      "framework": "triton",
      "version": "1.0.0",
      "description": "DenseNet image classification model served via Triton"
    }
  },
  "loaded_models": {}
}
```

cold start 검증 (같은 요청 2회):

```shell
$ [1회차] curl -X POST http://localhost:8001/predict -d '{"model_id": "550e...", "input_data": "I love this"}'
{"predictions":[[0.0001180444160127081,0.9998819828033447]]}
real	0m7.378s
user	0m0.003s
sys	0m0.000s


$ [2회차] 같은 요청 반복 (cold start 검증)
{"predictions":[[0.0001180444160127081,0.9998819828033447]]}
real	0m0.009s
user	0m0.000s
sys	0m0.003s


$ curl -s http://localhost:8001/models | jq .loaded_models
{
  "550e8400-e29b-41d4-a716-446655440000": "distilbert-base-uncased-finetuned-sst-2-english"
}
```

LRU 축출 (`max_models=2` — manager.py에서 확인):

```shell
$ [spam 모델 로드] curl /predict (2번째 모델 → 캐시 [sentiment, spam])
{"predictions":[[0.9315177798271179,0.06848226487636566]]}

$ curl /models | jq .loaded_models
{
  "550e8400-e29b-41d4-a716-446655440000": "distilbert-base-uncased-finetuned-sst-2-english",
  "6ba7b810-9dad-11d1-80b4-00c04fd430c8": "mrm8488/bert-tiny-finetuned-sms-spam-detection"
}

$ [image 모델 로드 → max_models=2 초과 → LRU(sentiment) 축출 예상]
{"predictions":[[0.000556908780708909,0.0005300698685459793,0.00042819976806640625,0.0004879587213508785,0.0007765536429360509,0.00048326622345484793,0.0003736938815563917,0.00028169856523163617,0.000517645908985287,0.0006431405781768262,0.0005171019001863897,0.0008888092124834657,0.0003733298799488 ...(1000차원 벡터 축약)

$ curl /models | jq .loaded_models  # sentiment가 빠졌는지
{
  "6ba7b810-9dad-11d1-80b4-00c04fd430c8": "mrm8488/bert-tiny-finetuned-sms-spam-detection",
  "7c9e6679-7425-40de-944b-e07fc1f90ae7": "pytorch/vision:mobilenet_v2"
}
```

- key point: cold start **7.378s → 0.009s (820배)**. 단 문서 경고("첫 호출은 로딩에서 끝난다")와 달리 이 구현은 첫 호출이 블로킹된 채 로딩 후 결과까지 반환한다. LRU는 교과서적으로 동작: [sentiment, spam] 상태에서 image 로드 → 가장 오래 안 쓴 sentiment가 축출되고 [spam, image]가 남았다.


### Lab 6 — Triton을 백엔드로 연동 ✅

```bash
mkdir -p model_dir/densenet_onnx/1

# config.pbtxt 작성 (densenet_onnx, onnxruntime_onnx, input data_0 [3,224,224], output fc6_1 [1000])

docker run -p8009:8000 -p8010:8001 -p8011:8002 \
    -v $(pwd)/model_dir:/models \
    nvcr.io/nvidia/tritonserver:24.12-py3 \
    tritonserver --model-repository=/models --model-control-mode=explicit
```

`--model-control-mode=explicit`이라 **모델을 명시적으로 로드해야** 한다:
```bash
curl -X POST http://localhost:8009/v2/repository/models/densenet_onnx/load
curl -v http://localhost:8009/v2/models/densenet_onnx/ready
curl -X POST http://localhost:8009/v2/models/densenet_onnx/infer -d @payload.json
```

#### ⚠️ 디스크

`nvcr.io/nvidia/tritonserver:24.12-py3` 이미지는 **압축 9.63GB / 전개 후 27.4GB**다.
WSL2의 vhdx가 그만큼 커지고, **한 번 커지면 자동으로 줄지 않는다.** 실습 후 정리 계획을 세워두는 게 좋다.

```bash
docker rm -f triton-densenet && docker rmi nvcr.io/nvidia/tritonserver:24.12-py3
```

**4090 판정: ✅ 가능.** densenet_onnx는 CPU 추론이라 `--gpus=1`은 선택사항이다. Docker Desktop(WSL2 백엔드)으로 충분하다.

---


#### 실행 기록

docker는 Docker Desktop 대신 **WSL2 안에 docker.io를 직접 설치**했다 (systemd 활성 확인 후):

```shell
$ sudo apt-get install -y docker.io
Created symlink /etc/systemd/system/multi-user.target.wants/docker.service → /usr/lib/systemd/system/docker.service.

Created symlink /etc/systemd/system/sockets.target.wants/docker.socket → /usr/lib/systemd/system/docker.socket.

Setting up dnsmasq-base (2.90-2ubuntu0.4) ...
Setting up ubuntu-fan (0.12.16+24.04.1) ...
Created symlink /etc/systemd/system/multi-user.target.wants/ubuntu-fan.service → /usr/lib/systemd/system/ubuntu-fan.service.

Processing triggers for man-db (2.12.0-4build2) ...
Processing triggers for dbus (1.14.10-4ubuntu4.1) ...
Processing triggers for libc-bin (2.39-0ubuntu8.8) ...

$ sudo usermod -aG docker enginrect
docker 그룹 추가 완료

$ sudo docker --version
Docker version 29.1.3, build 29.1.3-0ubuntu3~24.04.2
```

이미지 pull (압축 9.63GB — 문서 예고대로):

```shell
$ sudo docker pull nvcr.io/nvidia/tritonserver:24.12-py3
(로그 축약 없음 — 레이어 다운로드 진행 표시 후)
Status: Downloaded newer image for nvcr.io/nvidia/tritonserver:24.12-py3
```

컨테이너 기동 → 모델 로드 → ready → 메타데이터:

```shell
$ curl -X POST http://localhost:8009/v2/repository/models/densenet_onnx/load
HTTP 200

$ curl -v http://localhost:8009/v2/models/densenet_onnx/ready
HTTP 200

$ curl -s http://localhost:8009/v2/models/densenet_onnx | jq
{
  "name": "densenet_onnx",
  "versions": [
    "1"
  ],
  "platform": "onnxruntime_onnx",
  "inputs": [
    {
      "name": "data_0",
      "datatype": "FP32",
      "shape": [
        3,
        224,
        224
      ]
    }
  ],
  "outputs": [
    {
      "name": "fc6_1",
      "datatype": "FP32",
      "shape": [
        1000
      ]
    }
  ]
}
```

추론 검증은 리포의 pytest로 (tritonclient http 경유, cat1.jpg 전처리 포함):

```shell
$ python -m pytest tests/test_triton_densenet.py -v
============================= test session starts ==============================
platform linux -- Python 3.12.3, pytest-8.0.0, pluggy-1.6.0 -- /home/enginrect/llm-model-inference/ch03/multi_model_serving/venv/bin/python
cachedir: .pytest_cache
rootdir: /home/enginrect/llm-model-inference/ch03/multi_model_serving
plugins: asyncio-0.23.5, anyio-3.7.1
asyncio: mode=Mode.STRICT
collecting ... collected 3 items

tests/test_triton_densenet.py::TestTritonDenseNet::test_model_inference PASSED [ 33%]
tests/test_triton_densenet.py::TestTritonDenseNet::test_model_loading PASSED [ 66%]
tests/test_triton_densenet.py::TestTritonDenseNet::test_model_unloading PASSED [100%]

============================== 3 passed in 0.31s ===============================
```

정리 (Self-Check "이미지 삭제까지"):

```shell
$ sudo docker rm -f triton-densenet && sudo docker rmi nvcr.io/nvidia/tritonserver:24.12-py3
Untagged: nvcr.io/nvidia/tritonserver:24.12-py3
Deleted: sha256:e6d844f6cfd96bf91f021798c20bcdb59b75e490d2c778daec724deaf5221052

$ sudo docker system df
TYPE            TOTAL     ACTIVE    SIZE      RECLAIMABLE
Images          0         0         0B        0B
Containers      0         0         0B        0B
Local Volumes   0         0         0B        0B
Build Cache     0         0         0B        0B
```

- key point: `--model-control-mode=explicit`라 load API 호출 전에는 모델이 없다 — POST /load → ready 200 → 추론 3테스트 PASS(0.31s). densenet은 CPU 추론이라 `--gpus` 없이 돌았다. 정리 후 `docker system df` 전부 0B — 27.4GB 회수 확인.


# Part 2. 프로덕션 아키텍처

노션 명시: **"로컬 PC로 실습 가능(GPU 없이), AWS 계정(Bedrock API) 필요 + (옵션) KubeRay GPU 노드 1대"**

## 실습 기록 (Labs)

### Lab A — Amazon Bedrock (Option 1) ⬜ *(AWS 계정 필요)*

**노트북**: [`ch04/bedrock/aws_bedrock_examples.ipynb`](https://github.com/orca3/llm-model-inference/blob/main/ch04/bedrock/aws_bedrock_examples.ipynb)

세 단계다:
1. AWS 계정(**us-west-2 오레곤 리전**)에서 **Bedrock API 키** 생성
2. 모델 카탈로그에서 선택 — Anthropic 모델은 **최초 1회 사용 사례 제출**이 필요하다
3. API 호출

```bash
aws bedrock list-foundation-models --region us-west-2 | grep -E 'modelArn|modelId'
```

**4090 판정: ❌ 로컬 GPU와 무관.** 완전관리형 서비스라 대체 경로가 없다. AWS 계정이 있어야 한다.

> 비용은 토큰 과금이라 학습용으로는 소액이다. 다만 **API 키 관리에 주의** — 공개 저장소에 절대 커밋하지 말 것.


#### 진행 기록

**미진행 (환경 제약).** Bedrock은 완전관리형 서비스라 로컬 대체가 없고, 이 데스크탑 세션에는 AWS 계정
자격증명이 없다. 문서 판정(❌ 로컬 불가) 그대로. AWS 계정을 물릴 때 진행한다.

- key point: (미진행 — AWS 계정 필요)


### Lab B — Knowledge Agent (RAG) ✅ *(로컬 vLLM으로 완주)*

**구성**: `agent.py`(오케스트레이터) → `rag_system.py`(PDF 처리·임베딩·벡터 검색) / `llm_manager.py`(OpenAI API·토큰 관리) / `planner.py`(실행 계획) / `actions.py`(질의·요약·분석) / `config.py`

```
User Query → Agent → OpenAI API
              │
              └→ RAG System / LLM Manager / Planner / Actions
```

**4090 대체안**: 백엔드를 **로컬 vLLM의 OpenAI 호환 서버**로 바꾸면 키 없이 돌릴 수 있다.
```bash
vllm serve Qwen/Qwen2.5-7B-Instruct --max-model-len 8192

# config.py의 base_url을 http://localhost:8000/v1 로, api_key는 아무 문자열로
```
임베딩 모델도 로컬로 띄우면(`vllm serve BAAI/bge-m3` 등) 완전 오프라인 RAG가 된다.
**이게 4090을 가진 사람만 할 수 있는 확장 실습이다.** 스터디 노트 소재로도 좋다.


#### 실행 기록 ✅ (4090 확장 실습 — 완전 로컬 RAG)

서버 2개 동시 기동 — GPU 1장을 쪼개 썼다:

```shell
$ vllm serve Qwen/Qwen2.5-7B-Instruct --max-model-len 8192 --gpu-memory-utilization 0.80 --port 8000
$ vllm serve BAAI/bge-m3 --gpu-memory-utilization 0.10 --port 8001

# 1차 시도는 0.75 배분으로 실패했다:
# ValueError: No available memory for the cache blocks.
#   → 0.75×24GB=18.4GB에서 7B BF16 가중치 15.2GB + 오버헤드를 빼면 KV cache 블록이 안 나온다.
#   → chat 0.80 / embed 0.10 재배분으로 해결
```

임베딩 엔드포인트 확인:

```shell
$ curl -s http://localhost:8001/v1/embeddings -d '{"model": "BAAI/bge-m3", "input": "hello world"}' | jq (차원 확인)
{
  "model": "BAAI/bge-m3",
  "dims": 1024,
  "usage": {
    "prompt_tokens": 5,
    "total_tokens": 5,
    "completion_tokens": 0,
    "prompt_tokens_details": null
  }
}
```

KnowledgeAgent 패치 2건 (편차):
- `rag_system.py`: 임베딩 클라이언트에 `EMBEDDING_BASE_URL` 분리 (chat과 다른 서버라서), `tiktoken.encoding_for_model("Qwen/...")` KeyError → `cl100k_base` 폴백
- 코드 외 설정은 전부 env로: `OPENAI_API_KEY=local-dummy OPENAI_BASE_URL=http://localhost:8000/v1 LLM_MODEL=Qwen/Qwen2.5-7B-Instruct EMBEDDING_MODEL=BAAI/bge-m3`

1차 실행 — KB 구축(PDF 4종 임베딩)은 성공, chat이 전부 400:

```shell
ERROR:llm_manager:Error generating OpenAI response: Error code: 400 - {'error': {'message': "This model's
maximum context length is 8192 tokens. However, you requested 4096 output tokens and your prompt contains
at least 4097 input tokens, for a total of at least 8193 tokens. ..."}}
→ RAG 프롬프트(검색 청크 포함 ~4100 토큰) + MAX_TOKENS 4096 = 8193 > max-model-len 8192
```

> **⚠️ 왜 실패했나** — 두 건 모두 **가이드가 전제한 관리형 API와 로컬 서빙의 차이.**
> ① 0.75 배분 실패: OpenAI API에는 "GPU 메모리 배분"이라는 개념 자체가 없다. 로컬은 가중치+KV cache 합산을 직접 계산해야 한다.
> ② 8193 > 8192: OpenAI 서버는 컨텍스트 예산을 알아서 관리하지만, 로컬 vLLM은 input+output ≤ max-model-len을 클라이언트가 지켜야 한다.
> 가이드(OpenAI 키 전제)를 로컬로 옮기면서 새로 생긴 제약이지, 코드 결함이 아니다.

`MAX_TOKENS=1024`로 재실행 — 4개 쿼리 전부 성공:

```shell
$ (동일 env) + MAX_TOKENS=1024 python -c "from example_usage import example_basic_usage; example_basic_usage()"
INFO:agent:Initializing Agent components...
INFO:llm_manager:LLM Manager initialized
...
INFO:rag_system:Successfully processed 5-Level Paging and 5-Level EPT - Intel - Revision 1.0 (December, 2016).pdf into 18 chunks
...
INFO:rag_system:Successfully processed A Brief Tutorial on Database Queries, Data Mining, and OLAP (hamel-197-manuscript-final).pdf into 6 chunks
...
INFO:rag_system:Successfully processed A Case Study in Optimizing HTM-Enabled Dynamic Data Structures - Patricia Tries (2015).pdf into 13 chunks
...
INFO:rag_system:Successfully processed A Brief Introduction to the Standard Annotation Language (SAL) - 2006.pdf into 5 chunks
...
INFO:actions:Successfully generated RAG-based response
...
INFO:actions:Successfully generated RAG-based response
...
INFO:actions:Successfully generated summary
...
INFO:actions:Successfully generated RAG-based response
...
INFO:actions:Successfully generated RAG-based response
...
🔍 Query: Tell me about Standard Annotation Language (SAL)
✅ Response: Standard Annotation Language (SAL) is a meta-language designed by Microsoft Research to assist static analysis tools in finding bugs, particularly security bugs, in C or C++ code at compile time. Intr...
📋 Plan: This plan will use the Rich Answer Generation (RAG) system with context to provide a detailed and accurate description of Standard Annotation Language (SAL).
```

- key point: **OpenAI 키 없이 RAG 파이프라인 전체가 로컬에서 돌았다** — bge-m3(1024차원) 임베딩 → 벡터 검색 → 7B 생성. 응답이 PDF 근거와 부합한다(5-level paging=57-bit linear address, SAL=Microsoft Research의 정적 분석 메타언어). 발견 2건: ①GPU 1장 분할 서빙은 가중치+KV cache 합이 배분을 넘는 순간 기동 실패 — 배분 계산을 먼저 해야 한다 ②로컬 서빙은 컨텍스트 예산(input+output ≤ max-model-len)을 클라이언트가 직접 관리해야 한다 — OpenAI API에는 없는 제약.


### Lab C — Ray Serve on K8s (kind) ⬜ *(조건부)*

노션 도전과제: **"로컬 PC에 kind(k8s)로 RayService 배포 테스트"**

사전 준비:
- K8s (kind 또는 k3s)
- NVIDIA GPU Operator (GPU 사용 시)
- kube-prometheus-stack + DCGM Exporter + Grafana 대시보드 **12239**

**4090 판정**:
- **CPU-only 예제는 ✅ 쉽다.** Ray 공식 문서의 MobileNet 이미지 분류 서빙, Fashion MNIST 학습 예제가 CPU로 동작한다. RayService의 배포/스케일링 개념 학습에는 이걸로 충분하다.
- **GPU 노출은 ⚠️ 까다롭다.** kind는 컨테이너 안의 컨테이너라 GPU 패스스루 설정이 번거롭다. WSL2까지 겹치면 3중이다.
  → **대안: kind 대신 WSL2에 k3s를 직접 설치**하면 GPU device plugin 연결이 훨씬 단순하다.

Ray Dashboard는 `localhost:8265`.


#### 진행 기록

**미진행 (시간 배분).** docker는 이미 WSL2에 설치돼 있어(Lab 6) kind/k3s 경로 모두 열려 있다.
핵심 실습(GPU 벤치마크)을 우선했고, RayService는 스터디 후반의 AWS EKS 워크숍과 겹치는 주제라
그때 클라우드 환경에서 다루는 편이 낫다고 판단했다.

- key point: (미진행 — AWS EKS 워크숍과 통합 예정)


### Lab D — SageMaker (Option 2, 3) ⬜ *(노션에서 Skip 지정)*

노션에 **[실습Skip]** 으로 명시돼 있다. JumpStart / DLC(Deep Learning Containers) 경로는 읽고 넘어가면 된다.

단, 옵션으로 **"SageMaker 없이 서빙 컨테이너만 떼어서 로컬 GPU PC에서 재현"** 이 제시돼 있다.
LMI 컨테이너가 내부적으로 vLLM을 쓰기 때문에, 4090이 있으면 이건 시도해볼 만하다.

---


#### 진행 기록

**노션 지정대로 Skip.** LMI 컨테이너 로컬 재현은 선택 과제로 남긴다.

- key point: (Skip — 노션 지정)


## 도전과제

| # | 과제 | 판정 |
|---|---|---|
| 1 | 로컬 PC에 kind(k8s)로 RayService 배포 테스트 | ⚠️ CPU-only는 가능, GPU는 k3s 권장 |
| 2 | AWS EKS에 Ray Serve 설치 후 'Chat with Mistral' | ❌ AWS 필요 — **EKS 워크숍에서 다룰 예정** |

---

## 정리

1. **서빙 엔진 리포는 CPU 환경에서만 검증된 코드였다** — ModelManager가 모델을 CPU에 두고 Worker가 입력만 cuda로 옮겨, GPU 머신에서만 device 불일치로 죽는다. 1줄 패치(`model.to(device)`)로 해결. "GPU 없이 실습 가능"의 이면.
2. **배칭과 스트리밍의 공존 계층을 실측했다** — 토큰마다 sequence_id가 붙어 자기 클라이언트 스트림으로 라우팅된다. 단 이 구현은 동시 요청을 한 배치로 합치지 못하고 순차 처리했고(A 42줄 → B 42줄), 스트리밍 경로는 상태를 잇지 않아 출력이 비문이다.
3. **cold start 820배** (7.378s → 0.009s) — 첫 호출에 다운로드+로딩이 전부 얹힌다. LRU(max 2)는 3번째 모델 로드 시 가장 오래 안 쓴 것을 정확히 축출했다.
4. **Triton은 explicit 모드에서 load API 호출 전까지 모델이 없다** — load → ready → 추론(tritonclient) 3테스트 PASS. 이미지 27.4GB는 실습 후 즉시 회수했다.
5. **Bedrock 없이 RAG 완주** — vLLM 서버 2개(7B chat 0.80 + bge-m3 embed 0.10)로 GPU 1장을 분할, OpenAI 키 없이 PDF 4종 RAG가 동작. 대신 배분 계산(가중치+KV cache)과 컨텍스트 예산(input+output ≤ max-model-len) 관리가 클라이언트 몫이 된다.
6. **opt-125m는 동시성 실험 도구로 부적합하다** — 4090에서 요청당 ~100ms라 순차/병렬 차이가 노이즈에 묻힌다. 동기 호출 병목 재현은 초 단위 생성 모델이 필요.

---

## Self-Check

- [x] `single_model_llm_serving` 기동 후 **프로세스 3개** 확인 (부모 + 자식 2) (Lab 1 — 24625/24664/24790)
- [x] `/basic_generate` → `/generate` → `/generate_stream` 순서로 진행하며 코드 변화 추적 (Lab 1~3)
- [x] 동시 스트리밍 요청 2개로 **토큰 라우팅** 확인 (Lab 3 — sequence_id 분리, 단 순차 처리 관찰)
- [x] `/generate_vllm`로 vLLM 경로 비교 (WSL2) (Lab 4)
- [x] `multi_model_serving`에서 **cold start** 직접 확인 (Lab 5 — 7.378s → 0.009s)
- [x] `/models`를 반복 호출하며 **LRU 캐시** 변화 관찰 (Lab 5 — sentiment 축출 확인)
- [x] Triton 컨테이너로 densenet_onnx 추론 후 **이미지 삭제까지** (Lab 6)
- [x] (선택) Knowledge Agent를 **로컬 vLLM 백엔드로** 돌려보기 (Lab B — OpenAI 키 없이 완주)

