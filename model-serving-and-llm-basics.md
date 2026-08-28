# 모델 서빙 입문과 LLM 실행 원리

## Part 1. 모델 서빙의 큰 그림

### 모델은 파일 하나가 아니다

가장 먼저 교정해야 할 인식. `model.safetensors` 하나를 모델이라고 부르면 서빙을 이해할 수 없다.
모델은 **세 요소가 합쳐진 실행 가능한 소프트웨어**다.

| 구성요소 | 내용 | 대표 파일 |
|---|---|---|
| **모델 데이터** | 학습으로 얻은 weight, bias, embedding, 입출력 텐서 정의 | `model.safetensors`, `pytorch_model.bin` |
| **모델 아키텍처** | 레이어 종류·개수·연결 순서 (계산 경로 설계도) | `config.json` |
| **모델 실행 코드** | 아키텍처 초기화 → weight 로드 → inference 실행 | 모델 클래스 정의 + 추론 코드 |

```python
model = TheModelClass(*args, **kwargs)                              # 아키텍처
model.load_state_dict(torch.load("model_weights.pt"))               # 데이터
model.eval()
pred = model(inputs)                                                # 실행 코드
```

아키텍처와 데이터를 **분리해서 저장하는 이유**가 곧 운영 편의성이다. 버전 변경, 부분 로딩, fine-tuning,
레이어 추가 같은 시나리오에서 key가 일부 안 맞아도 호환되는 파라미터만 선택적으로 로드할 수 있다.

### 학습과 서빙은 목표 함수가 다르다

```
데이터 수집 → 학습 → 평가 → 배포 → [ 서빙 ] → 최적화 → 재학습
```

- **학습**: 정확도, loss, 학습 효율이 목표
- **서빙**: latency, throughput, 안정성, 보안, 비용, 모니터링이 목표

같은 모델이라도 이 두 국면에서 잘한다는 기준이 완전히 다르다. 그래서 "학습 잘 끝났으니 배포만 하면 됨"이
성립하지 않는다.

### 왜 최적화가 필수인가

LLM은 연산 요구량이 커서 데모에서는 멀쩡하다가 실사용자가 들어오는 순간 무너진다.

- 부하가 걸리면 latency 증가
- throughput이 하드웨어 성능보다 훨씬 낮은 지점에서 정체
- 사용량 증가에 비례해(혹은 그 이상으로) 비용 증가

Alphabet 회장 John Hennessy는 2023년 인터뷰에서 **LLM 요청 1건 처리 비용이 전통적 키워드 검색보다
10배 비쌀 수 있다**고 언급했다. 수십억 달러 규모의 추가 비용으로 이어질 수 있다는 뜻이다.

반대로 최적화 효과도 크다. 스터디 자료 기준:

| 사례 | 개선폭 |
|---|---|
| vLLM 팀 실험 (2023, Llama-7B/13B) | HF Transformers 대비 최대 24배, HF TGI 대비 최대 3.5배 |
| 책 저자 DeepSeek R1 실험 | 설정 2개(FP8 MLA 커널 + 배치 크기)만 조정해 38 TPS → 600 TPS (약 15배) |

> MLA(Multi-head Latent Attention)는 DeepSeek이 도입한 어텐션 방식으로,
> K/V 헤드를 줄이는 대신 **압축(latent)해서 캐싱**한다. 자세한 건 [최적화 노트](serving-bottlenecks-and-optimization.md)의 어텐션 절에서 다룬다.

**추가 인프라 투자 없이 설정만으로 배수 개선이 나온다는 게 핵심**이다. GPU가 비싼 자원이라는 점을 생각하면
이건 성능 튜닝이 아니라 비용 구조의 문제다. 극단적으로 표현하면 **GPU 노드 50대로 처리해야 할 워크로드를
10대로 처리하게 만드는 일**이다.

그리고 이 분야는 기술 회전이 유난히 빠르다. 대략 이런 주기로 돈다.

```
논문 발표 → 1~3개월 내 주요 런타임에 베타 기능으로 추가 → 6개월 내외로 프로덕션 적용
```

그래서 "한 번 배워두고 끝"이 성립하지 않는다. 새로 나온 기법의 장단점을 스스로 판단하려면
결국 **기반이 되는 Transformer 동작을 알아야 한다**는 결론으로 되돌아온다.

### 모델 서빙 4가지 방안

| 방식 | 핵심 특징 | 적합한 상황 |
|---|---|---|
| **On-device / Edge** | 단말 안에서 직접 실행 | 초저지연, 오프라인, 개인정보 보호 |
| **Single-model** | 컨테이너 하나에 모델 하나 | 최고 성능, 독립적 확장 |
| **Multi-model** | 컨테이너 하나에 여러 모델 공유 | 모델 수가 많고 트래픽이 낮거나 불규칙 |
| **Serving platform** | 여러 앱·모델·워크플로 통합 | 대규모 조직, 다단계 AI 서비스 |

#### On-device

앱 로직 → **모델 래퍼**(전/후처리 캡슐화) → **모델 런타임**(LiteRT, ONNX Runtime, Core ML) → 로컬 하드웨어.
배포 과정은 `변환 → 정확도 검증 → 성능 측정 → 패키징`.

제약이 분명하다. 연산·저장 공간 한계, 전력 소모, **업데이트를 모든 기기에 개별 배포해야 하는 문제**,
기기별 NPU 지원 불일치(iPhone Neural Engine 최적화 모델이 Snapdragon에선 비효율적일 수 있음).

#### Single-model

컨테이너 3대 구성요소: **API-Server**(HTTP/gRPC 노출) + **Model Management**(다운로드·로딩·갱신) +
**Inference Backend**(TF Serving, TorchServe, vLLM, TensorRT-LLM).

라우팅에서 단순 라운드로빈은 한계가 있다. 요청마다 처리 시간이 다르기 때문이다
(5,000토큰 요청 vs 100토큰 요청). 그래서 가중 라운드로빈, Least Connections, Least Response Time,
동적 로드밸런싱, **Cache-Aware Routing** 같은 전략이 필요해진다.

스케일링은 수평(인스턴스 추가, K8s HPA)과 수직(더 큰 GPU 또는 다중 GPU 분산) 두 축.

```bash
python -m vllm.entrypoints.api_server \
  --model meta-llama/Llama-2-13b-hf \
  --tensor-parallel-size 4
```

**중요 원칙: 여러 머신에 분산(inter-node)하기보다 한 머신에 여러 GPU(intra-node)를 우선하라.**
숫자로 보면 이유가 명확하다.

| 인터커넥트 | 대역폭 |
|---|---|
| NVLink (노드 내부) | 900 GB/s |
| IB / RoCEv2 400Gbps (노드 간) | 50 GB/s |

400Gbps를 8로 나누면 50GB/s다. **약 18배 차이.**
노드를 넘는 순간 네트워크 오버헤드와 동기화 복잡도가 지연시간을 지배한다.

구체적으로 어디서 터지는가. GPU 노드는 보통 **내부에 GPU 8장**으로 구성된다.
`--tensor-parallel-size 8`까지는 NVLink로 900GB/s를 쓰지만, **9장이 필요해지는 순간 노드를 넘어가고**
그 순간부터 GPU들이 all-to-all로 주고받는 통신이 50GB/s 구간을 타면서 병목이 된다.

(GB200/GB300의 랙스케일 아키텍처는 NVLink 도메인으로 72 GPU를 묶어 이 제약을 밀어낸다.
다만 국내 도입 사례가 아직 많지 않아서, 당분간은 "노드당 8장" 기준으로 설계하는 게 현실적이다.)

관련해서 K8s 스케줄링도 짚고 넘어갈 부분이 있다. 기본 스케줄러는 Pod을 노드에 **골고루 분산**한다.
GPU 워커 노드 5개에 1 GPU 요구·replica 5인 워크로드를 올리면 노드마다 하나씩 흩어진다.
이러면 이후 2 GPU를 요구하는 워크로드가 들어올 자리가 없어진다 — **GPU 파편화**다.
**Bin Packing** 스케줄링은 반대로 노드를 빈틈없이 채워서 이 파편화를 완화한다.
서로 통신이 잦은 Pod들을 같은 노드에 모아두면 느린 노드 간 링크를 탈 일도 줄어든다.

이런 이유로 GPU 워크로드 전용 스케줄러가 계속 나오고 있다 — **Kai Scheduler**, **Volcano Gang Scheduler** 등.

한계도 분명하다. 고객 100명 × 모델 10개 = **1,000개 개별 서비스**를 운영해야 하는 상황이면
유지보수·패치·모니터링 부담이 감당 불가능해지고, 안 쓰이는 모델도 자원을 계속 점유한다.

#### Multi-model

컨테이너 하나가 여러 모델을 공유하고, 요청이 올 때만 동적으로 로드/언로드한다.

```
Container 1
├─ Model A
├─ Model B
└─ Model D
```

- **Model Server Inference Backend**: 프레임워크가 달라도 통합 예측 API로 처리 (예: NVIDIA Triton)
- **Model Cache Management**: 다운로드·로드 + **LRU 캐시**로 자원 임계치(예: 80%) 초과 시 가장 덜 쓰인 모델 언로드

이 방식은 두 가지 새로운 문제를 만든다.

1. **라우팅 문제** — 모델이 어느 컨테이너에 있을지 모르니, 이미 로드된 컨테이너로 보내야 한다.
   아니면 콜드 스타트나 모델 스와핑으로 지연이 발생한다.
2. **모델별 스케일링 문제** — hot 모델은 더 많은 인스턴스가 필요하다.

해결책은 라우팅 계층에 **replica 속성** + **route map**(어떤 모델이 어떤 컨테이너에 있는지)을 두는 것.
모델 A의 replica가 2면 라우터가 C1, C3에 배치하고 이후 A 요청을 두 곳에 분배한다.
새 replica를 어느 컨테이너에 넣을지는 일종의 bin-packing 문제가 된다.

→ 요즘 이 역할을 하는 게 **AI Gateway** 제품군이다. Envoy AI Gateway, LiteLLM 등.

#### Model Serving Platform

앱이 늘고 요청 하나가 여러 모델을 거치기 시작하면 위 두 방식을 나열하는 것만으로는 부족하다.

```
Gateway → Graph Execution → Routing → Resource Group → Single/Multi-model Service
```

- **Graph Execution**: 다단계 추론 워크플로 실행 (Airflow, Ray 등)
  예) 챗봇: `Intent Classification → Embedding → Retrieval → LLM → Safety Filter`
- **Resource Groups**: 앱별로 CPU/GPU/메모리 쿼터를 분리해 격리

대표 구현: **KServe**(K8s 네이티브, scale-to-zero), **Ray Serve**(추론 그래프 구성에 강점),
**MLflow**(모델 레지스트리 연동).

---

## Part 2. LLM 기초 — 왜 이런 구조가 되었나

서빙 개론과 실행 원리 사이를 잇는 보충 세션. 서빙 최적화 기법들이 결국 **Transformer의 구조적 특성에서 파생된
병목을 푸는 시도**라서, 이 부분을 건너뛰면 뒤의 최적화가 암기가 된다.

### 가중치가 "학습된다"는 것

키와 몸무게로 어른/아이를 판단하는 신경 하나로 시작해보자.

```
입력(키, 몸무게) × 가중치 → 합 → 역치(0.5) 넘으면 1(어른)
```

- 첫 시도: 가중치를 랜덤(키 0.3, 몸무게 0.2)으로 두면 어른 데이터에 0(아이)이 나온다 → 오답
- 가중치를 0.5로 올려보면 해당 데이터에서 정답이 나온다
- **컴퓨터는 가중치를 바꿔가며 오차가 없어질 때까지 계속 계산한다**

층을 여러 개 쌓으면 딥러닝이 되고, 출력값에 오차가 생기면 **출력에 가까운 쪽부터 거꾸로 가중치를
수정**한다(역전파). 그 다음 다시 앞으로 전파해 계산하고, 오차가 생기면 다시 뒤로 — 이 반복이 학습이다.

핵심 전환: **규칙을 사람이 쓰는 게 아니라, 데이터를 보여주면 컴퓨터가 규칙(최적 가중치)을 스스로 찾는다.**

```
(전통) 입력 + 규칙       → 결과
(ML)   입력 + 결과(정답) → 규칙
```

### 용어 정리: Parameter / Weight / Gradient

혼동하기 쉬워서 확실히 구분해둔다.

```
Parameter (학습 가능한 숫자 전체)
├── Weight
├── Bias
├── Embedding 값
└── LayerNorm의 scale, bias
```

**Gradient**는 파라미터에 포함되지 않는다. 지식을 저장하는 값이 아니라
**"현재 파라미터를 어느 방향으로 얼마나 바꾸면 오차가 줄어드는지" 알려주는 임시 안내값**이다.

```
새 Weight = 현재 Weight − 학습률 × Gradient
          = 0.50 − 0.10 × 0.20
          = 0.48
```

| 용어 | 무엇인가 | 학습 중 바뀌는가 |
|---|---|---|
| Token | 입력 문장의 조각 | 아니오 |
| Weight | 입력 영향을 조절하는 숫자 | 예 |
| Parameter | 모델이 학습·저장하는 숫자 전체 | 예 |
| Gradient | 파라미터 수정 방향과 크기 | 매 학습마다 새로 계산 |

### GPU가 병렬인 이유 (서빙과 직결)

- **SIMD** (Single Instruction Multiple Data): 하나의 명령으로 여러 데이터 처리.
  코어 100개에 스레드를 하나씩 할당하는 방식.
- **SIMT** (Single Instruction Multiple Threading): 스레드 중심 처리.
  스레드를 묶고 코어도 묶어서 배치한다. 이때 묶인 스레드 그룹을 **warp**라고 한다.
- RTX 3090은 코어가 1만 개 넘고 **128개씩 그룹으로 묶여** 있다. warp가 메모리 지연을 만나면
  **바로 다음 warp를 수행**해서 연산이 끊기지 않게 한다.

이게 왜 서빙에서 중요한가. Transformer는 **상당 부분의 연산이 서로 독립적**이라 병렬화가 잘 되고,
그래서 GPU가 압도적으로 유리하다. (다만 전부 병렬은 아니다 — 정확히 어디가 병렬이고 어디가 순차인지는
아래 [어디가 병렬이고 어디가 순차인가](#어디가-병렬이고-어디가-순차인가)에서 정리한다.)
그리고 이 병렬 연산의 속도와 메모리 효율을 좌우하는 게 **dtype과 양자화**다.

### 텐서

```
스칼라(0D) → 벡터(1D) → 행렬(2D) → 3D 텐서
```

차원 개수를 차수(order, rank)라고 부른다. 계산 관점에서 텐서는 다차원 데이터 컨테이너다.
그리고 신경망은 원시 텍스트를 처리할 수 없으므로, 단어를 실수 벡터로 바꿔야 한다 → **임베딩**.

### GPT-2 Small 해부

Transformer를 손에 잡히게 보려면 구체적인 숫자가 필요하다.

| 모델 | 파라미터 | Block | 임베딩 차원 | Head |
|---|---|---|---|---|
| **GPT-2 Small** | **약 124M** | **12** | **768** | **12** |
| GPT-2 Medium | 약 355M | 24 | 1,024 | 16 |
| GPT-2 Large | 약 774M | 36 | 1,280 | 20 |
| GPT-2 XL | 약 1.5B | 48 | 1,600 | 25 |

GPT-2 Small 상세:

| 항목 | 값 |
|---|---|
| vocab size | 50,257 |
| context length | 1,024 토큰 |
| embedding dim | 768 |
| Attention heads | Block당 12개 (768 ÷ 12 = 64차원씩) |
| layers | Transformer Block 12개 |
| FFN 중간 차원 | 3,072 (768 × 4) |
| 파라미터 | 약 124M |

#### 데이터가 흐르는 shape

```
[배치, 토큰]              # 입력 토큰 ID
    ↓ Embedding + Positional Encoding
[배치, 토큰, 768]
    ↓ Transformer Block × 12   (마지막 차원 768 유지)
[배치, 토큰, 768]
    ↓ LM Head (출력층)
[배치, 토큰, 50257]       # logits
```

**마지막 차원이 12개 Block을 지나도 768로 유지된다**는 게 포인트다. Block은 정보를 정제할 뿐
차원을 바꾸지 않는다.

임베딩 벡터는 처음에 **무작위로 배치**되어 있다가, 훈련이 진행되면서 제자리를 찾아간다.
768차원을 2차원으로 줄여서 보면 `man`/`woman`이 좌우 축으로, `어른`/`아이`가 상하 축으로
갈라지는 식으로 자리를 잡는다. 가중치가 학습된다는 게 임베딩 공간에서는 이렇게 나타난다.

한국어 예시로 감을 잡자면 — `싸늘하다 가슴에 비수가 날아와` 를 넣으면 마지막 층에서
어휘 사전 전체에 대한 확률 분포가 나오고, `꽂힌다` 68% / `타자가` 21% / `손이` 7% 같은 식으로 계산된다.
가장 높은 후보가 다음 토큰으로 선택된다.

#### Transformer Block 내부

각 Block은 이전 Block의 출력을 입력으로 받아 **순차 동작**한다(여기는 병렬화가 안 되는 지점).
Block 안에는 두 부분이 있다.

- **Masked Multi-Head Self-Attention** — 다른 토큰과의 **관계**를 본다
- **FFN(MLP)** — 각 토큰 **자체의 의미**를 더 깊게 가공한다 (768 → 3,072 → 768)

여기에 정보 손실을 막는 두 장치가 붙는다.

**Residual Connection (잔차/스킵 연결)**

```
Attention 출력 = x + Attention(x)
MLP 출력       = y + MLP(y)
```

12개 Block을 지나며 매번 입력을 완전히 바꾸면 처음 정보가 사라진다. 원래 정보를 계속 다음 층으로
전달하는 우회로를 만드는 것. 원래는 컴퓨터 비전의 residual network에서 **그레이디언트 소실 문제**를
완화하려고 제안된 기법이다. 그레이디언트가 층을 거듭해 역전파되며 점점 작아져 앞쪽 층이
학습되지 않는 현상을, 층을 건너뛰는 짧은 경로로 해결한다.

**Layer Normalization** — 토큰 벡터 안 숫자들의 평균과 퍼짐을 맞춰서 계산이 흔들리지 않게 한다.
Self-Attention 전에 한 번, MLP 전에 한 번.

**Dropout** — 학습 중 일부 신호를 무작위로 끈다. 모델이 특정 연결에만 의존하는 **과적합(Overfitting)**을
막기 위함이다. `특징 A가 꺼져도 B와 C로 판단`하게 만든다. (추론 시에는 동작하지 않는다)

#### Q, K, V를 붙잡는 비유

DB 쿼리를 떠올리면 이름이 왜 이런지 바로 잡힌다.

```
SELECT ... WHERE 조건   ← Query (무엇을 찾고 싶은가)
  → 테이블의 Key와 매칭  ← Key   (매칭 대상)
    → 해당 Value 반환    ← Value (실제로 가져올 내용)
```

Attention도 똑같다. 한 토큰의 Query가 다른 모든 토큰의 Key와 내적해서 "얼마나 관련 있나"를 점수로 매기고,
그 점수로 Value들을 가중합한다. **내적 값이 크면 유사도가 높다는 뜻**이다.
"어떤 남자가 힘이 억센 소방관이었다"에서 `남자`와 `힘이 억센`의 내적은 높게 나온다.

Attention을 우리말로 옮기면 "주의"인데, **"눈치"** 로 이해하면 더 와닿는다.
`눈이 내렸다` 다음에 `눈이 아파`가 오면 말이 안 된다. 주변 단어의 눈치를 봐야 다음 단어가 제대로 나온다.
그리고 이 계산에 **제3의 외부 값이 개입하지 않고 주어진 토큰들끼리만** 하기 때문에 **Self**-Attention이다.

#### Masked Self-Attention

GPT-2는 미래 단어를 미리 볼 수 없다.

```
나는 오늘 학교에 갔다
       ↑ "오늘"을 처리할 때 "학교에 갔다"를 보면 안 됨

토큰 1 → 토큰 1만 참고
토큰 2 → 토큰 1~2 참고
토큰 3 → 토큰 1~3 참고
```

구현은 의외로 단순하다. **내적 결과에 극단적인 큰 음수를 더한다.** softmax의 `exp()`를 거치면
큰 음수는 0이 되므로 자연스럽게 가려진다. Transformer Explainer에서 확인해보면
내적(Dot product) 자체는 수행되고, Mask 단계에서 0이 되는 걸 볼 수 있다.

#### Multi-Head

Head 12개가 **같은 문장을 서로 다른 Weight/Bias로 병렬 분석**한다.

```
Head 1: 가까운 단어 관계
Head 2: 주어와 동사 관계
Head 3: 대명사가 가리키는 대상
Head 4: 문장 위치 관계
```

(실제로 이렇게 깔끔히 나뉘지는 않는다. 개념적 이해용.)
각 Head가 64차원을 담당하고, 12개 Out을 이어붙여(concat) 다시 768차원이 된다.

#### 어디가 병렬이고 어디가 순차인가

Transformer가 GPU와 궁합이 맞는 이유를 정확히 짚으려면 이 구분이 필요하다.
RNN 시절에는 순차 처리가 대부분이라 GPU를 제대로 활용하지 못했고, Transformer가 그 제약을 깼다.
다만 **전부 병렬은 아니다.**

| 구간 | 병렬/순차 | 이유 |
|---|---|---|
| Multi-Head 12개 | **병렬** | 각 Head가 독립적으로 64차원씩 처리 |
| MLP의 토큰별 연산 | **병렬** | 토큰끼리 서로 참조하지 않음 (관계 파악은 attention이 이미 끝냄) |
| Transformer Block 12개 | **순차** | Block N의 출력이 Block N+1의 입력 |

즉 **Block 안은 병렬, Block 사이는 순차**다.

> **짚고 넘어갈 점** — masking이 연산을 아껴주는 건 아니다.
> 큰 음수를 더하는 방식은 내적을 **이미 다 계산한 뒤** 결과를 무력화하는 것이라, 순진하게 구현하면
> 연산량은 그대로다. masking의 목적은 어디까지나 **정확성**(미래 토큰을 못 보게)이다.
> 실제로 연산을 아끼려면 causal 구조를 아는 커널이 **가려질 블록 자체를 건너뛰어야** 하고,
> FlashAttention의 causal 모드가 하는 일이 그것이다. 개념(masking)과 최적화(kernel)를 분리해서 봐야 한다.

#### 출력: logits → 확률 → 토큰

마지막 토큰의 최종 벡터 → 출력 가중치 곱 + bias → **50,257개 logit** → softmax → 다음 토큰 선택.

선택 전략이 곧 서빙 API의 파라미터다.

- **Temperature**: `scaled logit = logit ÷ temperature`.
  1보다 작으면 점수 차이가 벌어져 **보수적·확정적**인 선택이 된다.
- **Top-k**: 상위 k개만 남기고 나머지는 큰 음수 → softmax 결과 0%
- **Top-p (Nucleus Sampling)**: 누적 확률이 p 이상이 될 때까지 높은 순으로 남긴다

```
Top-p = 0.8
좋다: 35%   → 누적 35%
춥다: 25%   → 누적 60%
덥다: 15%   → 누적 75%
맑다: 10%   → 누적 85%  ← 여기서 80% 초과, 여기까지 후보
이상하다: 5% → 제거
```

참고로 GPT-2 계열은 **토큰 임베딩 Weight와 마지막 출력층 Weight를 공유**한다
(`토큰 ID → 768차원` / `768차원 → 50,257 점수`). 파라미터를 아끼면서 입출력 표현을 연결하는 기법이다.

#### 규모 감각

| | GPT-2 Small | GPT-3 |
|---|---|---|
| 파라미터 | 약 1.24억 | 약 1,750억 |
| 레이어 | 12 | 96 |
| 임베딩 차원 | 768 | 12,288 |
| Attention heads | 12 | 96 |
| context length | 1,024 | 2,048 |
| vocab | 50,257 | 50,257 |

GPT-3의 1,750억 개를 구성요소별로 뜯어보면 Embedding / Key / Query / Value / Output보다
**Up-projection과 Down-projection이 압도적으로 크다.** 즉 **전체 파라미터의 약 2/3가 MLP에서 쓰인다.**
Attention보다 MLP가 사실 정보를 담아둘 공간이 더 크다는 뜻이다.

**그리고 여기서 MoE가 나온다.** 파라미터의 2/3를 차지하는 구간이라면, 그걸 줄이는 게 가장 큰 이득이다.
MoE(Mixture of Experts)는 MLP를 여러 전문가로 쪼개고 **라우터가 일부만 활성화**해서 통과시킨다.

```
활성 파라미터 감소
  → GPU에 올릴 가중치 감소 (HBM 절약)
    → 메모리 대역폭 사용량 감소
      → decode(memory-bound) 구간이 빨라짐
```

"MoE는 큰 모델을 싸게 돌리는 기법"이라고 외우는 것과, **파라미터의 2/3가 MLP라서 거기를 건드리는 것**임을
아는 건 다르다. 이게 Part 1에서 말한 "기본기를 알아야 새 기술을 판단할 수 있다"의 구체적인 예다.

### dtype과 양자화 — 병렬 연산이 서빙과 만나는 지점

부동소수점은 `부호 + 지수(범위) + 가수(정밀도)`로 구성된다.

| dtype | 총 비트 | 지수 | 가수 | 표현 범위 | 유효숫자 |
|---|---|---|---|---|---|
| float32 | 32bit | 8 | 23 | ±10⁻³⁸ ~ ±10³⁸ | 약 7자리 |
| float16 | 16bit | 5 | 10 | ±6×10⁻⁵ ~ 65504 | 약 3~4자리 |
| bfloat16 | 16bit | 8 | 7 | float32와 동일 | 약 2~3자리 |

**bfloat16이 학습에서 선호되는 이유**가 표에 드러난다. 가수는 더 줄이지만 **지수를 float32만큼 유지**해서
표현 범위가 같다. 그래서 오버플로우에 안전하다.

서빙 관점에서 중요한 이유 세 가지:

1. **모델 크기 절반** — 70B 모델이 float32면 약 280GB, float16이면 약 140GB. GPU에 올라가느냐 마느냐가 갈린다.
2. **KV 캐시도 절반** — decode는 memory-bound인데, 읽어오는 데이터양이 절반이면 실질적으로 빨라진다.
3. **Tensor Core 가속** — NVIDIA GPU는 fp16 연산 전용 하드웨어를 갖고 있다.

**dtype 변환과 양자화는 다른 얘기**다.

| | dtype 변환 (fp32→fp16/bf16) | 양자화 (fp→int8/int4) |
|---|---|---|
| 표현 방식 | 여전히 부동소수점 (연속적) | 정수 + scale/zero-point (이산적) |
| 압축률 | 2배 | 4배(int8), 8배(int4) |
| 정밀도 손실 | 상대적으로 적음 | 더 큼, calibration 없으면 성능 급락 |
| 추가 작업 | 거의 없음 (캐스팅) | calibration 데이터셋 또는 QAT 필요 |
| 대표 기법 | 단순 캐스팅 | GPTQ, AWQ, bitsandbytes(NF4), SmoothQuant, GGUF |
| 주 용도 | 학습·추론 둘 다 | 주로 추론 전용 |

양자화는 **연속적인 실수값을 정해진 개수의 정수 칸에 끼워 맞추는** 것이다. int8이면 -128~127 중 하나로
반올림하고, 복원할 때 `실제값 ≈ (정수값 − zero_point) × scale` 환산표를 따로 들고 다닌다.

int4까지 가면 칸이 16개 수준이라 정확도가 눈에 띄게 떨어질 수 있다. 그래서 GPTQ·AWQ처럼
**어떤 가중치가 더 중요한지 분석해 오차를 최소화하는 알고리즘**이 필요해진다.

실무 순서는 대체로 이렇다. **먼저 fp16/bf16으로 절반 줄이고, 그래도 부족하면 양자화를 추가 적용.**

---

## Part 3. LLM이 실제로 실행되는 방식

### 자기회귀(Autoregressive) 특성

LLM은 **토큰을 한 번에 하나씩** 생성하고, 생성된 토큰을 입력 시퀀스에 다시 붙여 다음 토큰을 예측한다.

```
"미국 수도에 대한 짧은 소개를 써줘"
  → Washington
  → Washington D.C.
  → Washington D.C. is
  → Washington D.C. is the ...
```

**운영 관점에서 이게 결정적이다.** 요청 하나가 forward pass 한 번이 아니라 **여러 decode step의 연속**이고,
**생성 길이가 길수록 GPU 점유 시간이 길어진다.** 요청당 비용을 예측하기 어려운 근본 원인이다.

### 모델 config 읽기 — Qwen2.5-0.5B

서빙 전에 `config.json`부터 읽어야 한다. GPU 메모리 추정, 서빙 전략 선택, 최적화 계획이 여기서 나온다.
모델을 내려받지 않아도 Hugging Face 모델 카드에서 바로 확인할 수 있는 값들이다.

Qwen2.5-0.5B의 주요 항목은 다음과 같다.

| 항목 | 값 | 의미 |
|---|---|---|
| `hidden_size` | 896 | 토큰 하나를 표현하는 벡터 차원 |
| `num_hidden_layers` | 24 | Transformer Block 개수 |
| `num_attention_heads` | 14 | Query 헤드 수 |
| `num_key_value_heads` | **2** | KV 헤드 수 ← 주목 |
| `intermediate_size` | 4864 | FFN 은닉층 크기 |
| `vocab_size` | 151,936 | 다국어 지원이라 매우 큼 (영어 전용은 보통 5만 내외) |
| `max_position_embeddings` | 32,768 | 컨텍스트 윈도우 |
| `torch_dtype` | bfloat16 | 가중치 저장 정밀도 |
| `tie_word_embeddings` | true | 입력 임베딩과 출력층 가중치 공유 |
| `rope_theta` | 1,000,000 | RoPE 위치 인코딩 파라미터 |

전체 파라미터는 약 4.94억(494M)이다. GPT-2 Small(12층·768차원)과 비교하면
**더 깊고(24층) 더 넓은(896차원)** 구조다.

#### config에서 GQA를 읽어내는 법

`num_attention_heads: 14` vs `num_key_value_heads: 2`. **이 불일치 자체가 GQA(Grouped Query Attention)의
증거다.** 모델 카드에 "GQA"라는 단어가 한 줄도 없어도 이 두 값만 비교하면 알 수 있다.

여기서 파생되는 값들을 직접 계산해보자.

```
head_dim              = 896 ÷ 14 = 64
num_key_value_groups  = 14 ÷ 2  = 7      ← Query 헤드 7개가 KV 헤드 1개를 공유
scaling               = 1/√64   = 0.125  ← Scaled Dot-Product Attention의 스케일링 팩터

q_proj = [14 × 64, 896] = [896, 896]
k_proj = [ 2 × 64, 896] = [128, 896]     ← Q보다 7배 작다
v_proj = [ 2 × 64, 896] = [128, 896]
```

**Query는 14개 헤드를 온전히 갖지만 Key/Value는 2개만 계산한다.**
일반 MHA라면 K/V도 14개여야 하니, **KV 캐시 크기가 약 7배 줄어든다.**

파라미터 수로 환산하면 차이가 더 분명하다.

| | k_proj + v_proj 파라미터 |
|---|---|
| MHA였다면 (KV 헤드 14개) | 2 × (896 × 896) = **1,605,632** |
| GQA 실제 (KV 헤드 2개) | 2 × (896 × 128) = **229,376** |

정확히 **7.0배**다. `num_key_value_groups: 7`과 정확히 맞아떨어진다.

서빙 비용에 직접 꽂히는 아키텍처 선택이고, 뒤에 나올 KV 캐시 최적화의 배경이다.
config를 읽을 줄 알면 모델 카드에 GQA라는 단어가 없어도 `num_attention_heads`와
`num_key_value_heads`만 비교해서 알아낼 수 있다.

### KV Cache — 이번 주 최대 수확

#### 캐시 없이 돌리면 어떻게 되나

`pipeline()` 추상화를 걷어내고 생성 루프를 그대로 펼쳐보면 문제가 드러난다 (교재 예제 코드).

```python
for _ in range(max_new_tokens):
    idx_cond = idx                      # ← 매번 전체 시퀀스를 통째로 입력
    outputs = model(idx_cond)
    logits = outputs.logits[:, -1, :]   # ← 그런데 마지막 것만 씀
    probas = torch.softmax(logits, dim=-1)
    idx_next = torch.multinomial(probas, num_samples=1)
    idx = torch.cat((idx, idx_next), dim=1)
```

문제는 `idx_cond = idx` 한 줄에 있다. **매 스텝마다 지금까지 쌓인 전체 시퀀스를 통째로 모델에 다시 넣는데,
정작 쓰는 건 마지막 위치의 logits 하나뿐**이다. 즉 이미 계산했던 이전 토큰들의 attention을
스텝마다 처음부터 다시 계산하고 대부분을 버린다. 그래서 시퀀스가 길어질수록 토큰당 생성 시간이 계속 늘어난다.

#### 계산 복잡도로 보기

sequence 길이를 L, 차원을 D라 하면:

- self-attention에서 가장 무거운 연산은 **query와 key의 내적**
- 내적 하나에 D번 곱셈 + D번 덧셈
- 이걸 query 전체 길이 L × key 전체 길이 L = **L² 번** 수행

```
캐시 없음:  O(L² D)
```

그런데 이 부담스러운 계산으로 얻는 건 **다음 토큰 하나**뿐이다.
LLM 출력 시퀀스에서 **맨 뒤만 쓰고 앞의 것은 전부 버린다.**
그리고 토큰이 하나 붙으면 다시 `f((L+1)²)` 로 계산이 늘어난다.

#### 무엇을 캐시하는가

**이전 토큰들의 Key와 Value 시퀀스**를 저장한다. Query는 캐시하지 않는다.

왜 Q는 빼는가. **Query는 일회용이기 때문이다.** 이번 스텝에서 "지금 이 토큰이 앞의 것들과 얼마나 관련 있나"를
묻는 데만 쓰고, 다음 토큰을 만들 때 누적해서 재사용하지 않는다. 반면 Key/Value는 뒤에 오는 모든 토큰이
계속 참조해야 하므로 쌓아둬야 한다.

#### 캐시가 물리적으로 어디에 있나

이 부분이 뒤의 FlashAttention·PagedAttention을 이해하는 열쇠다. KV 캐시는 추상적인 개념이 아니라
**GPU의 HBM에 실제로 쌓이는 데이터**다.

```
[1번째 토큰]
  GPU SRAM에서 attention 계산 → K, V를 HBM으로 내려보냄(저장)

[2번째 토큰]
  새 토큰의 K, V만 SRAM에서 계산
  + 직전에 저장해둔 K, V를 HBM → SRAM으로 다시 가져와 이어붙임
  → attention 계산 → 갱신된 K, V를 다시 HBM에 저장
```

여기서 두 가지 비용이 새로 생긴다.

1. **HBM 용량** — 원래는 모델 weight와 임시 텐서만 GPU 메모리를 썼는데, 이제 KV 캐시 몫을 따로 확보해야 한다.
   고려하지 않고 올리면 **OOM으로 터진다.** 서빙 설계 단계에서 GPU 스펙을 정할 때 반드시 계산에 넣어야 하는 항목이다.
2. **HBM ↔ SRAM 왕복** — 안 그래도 병목인 구간인데, 토큰마다 캐시를 읽고 쓰느라 왕복이 계속 발생한다.

(LLM 수요가 HBM 수요로 직결되는 이유가 여기 있다. 캐시가 곧 메모리다.)

#### 어떻게 부담이 줄어드나

self-attention의 **입력 자체가 바뀐다.** 시퀀스 전체가 아니라 **마지막 샘플 하나만** 들어간다.

- **Query**: 시퀀스 내에서 서로 의존성이 없다 → 마지막 하나만 있어도 결과가 같다
- **LLM 출력**: 어차피 맨 마지막 샘플로만 다음 토큰을 구한다
- **Key/Value**: softmax와 weighted sum 때문에 길이 축에 의존성이 있어 **전체 시퀀스가 필요하다**
  → 하지만 **직전에 이미 계산했으므로 저장해뒀다 재사용하면 된다**

이 세 가지가 맞물려서 마지막 단일 샘플 입력만으로 다음 토큰을 만들 수 있게 된다.

```
캐시 사용:  O(L D)     ← query 하나에서만 L 길이의 key와 내적
```

#### 코드 차이

```python
past_key_values = None

for _ in range(num_iterations):
    outputs = model(
        input_ids=input_ids,
        past_key_values=past_key_values,   # 이전 스텝의 KV 캐시 전달
        use_cache=True,                    # 캐시 활성화
    )
    past_key_values = outputs.past_key_values   # 캐시 갱신

    # ... 토큰 샘플링 ...
    input_ids = generated_token_id         # ← 방금 만든 토큰 "하나만" 입력
```

캐시 없을 때의 `idx_cond = idx`(전체)와 대비된다.

#### 효과 — 교재에 실린 측정값

아래는 **교재에 실린 수치를 인용한 것**이다. 100토큰 생성 기준.

| | 생성 시간 | 토큰별 패턴 |
|---|---|---|
| 캐시 없음 | 9.12초 | 토큰이 늘수록 점점 느려짐 (우상향) |
| **KV 캐시 사용** | **3.14초** | 첫 토큰 이후 거의 일정 (평탄) |

**약 3배.** 절대값보다 중요한 건 **그래프 모양이 우상향에서 평탄으로 바뀐다**는 점이다.
캐시가 없으면 긴 응답일수록 뒤로 갈수록 느려져 비용을 예측할 수 없지만, 캐시가 있으면
토큰당 비용이 일정해진다.

그리고 이건 트레이드오프다. **연산을 줄이는 대신 GPU 메모리를 쓴다.**
그래서 서빙에서는 **KV 캐시 메모리 관리가 throughput·latency·concurrency를 결정하는 핵심 변수**가 된다.

> 참고: KV 캐시는 **추론에서만** 쓴다. 학습 단계는 모든 위치의 정보가 필요해서 쓸 수 없다.

### Prefill과 Decode

KV 캐시를 켜고 토큰별 시간을 다시 그려보면, **첫 번째 막대만 압도적으로 높고 나머지는 짧고 일정**하다.
이 첫 막대가 **Prefill**, 나머지가 **Decode**다.

| | Prefill | Decode |
|---|---|---|
| 하는 일 | 프롬프트 전체를 한 번에 처리, KV 캐시 생성 | 토큰을 하나씩 순차 생성 |
| 병렬화 | 잘 됨 (모든 프롬프트 토큰 동시 처리) | 안 됨 (sequential dependency) |
| 복잡도 | 시퀀스 길이에 대해 2차 | 캐시 덕에 선형 |
| **병목** | **Compute-bound** (GPU 연산 유닛) | **Memory-bound** (HBM 대역폭) |
| 주요 지표 | TTFT (Time To First Token) | ITL / TPOT, token throughput |

> 지표 약어: **TTFT**(Time To First Token) 첫 토큰까지의 시간,
> **ITL**(Inter-Token Latency) 또는 **TPOT**(Time Per Output Token) 토큰 사이의 간격.
> 체감 응답성은 TTFT가, 읽히는 속도는 ITL이 좌우한다.

한 가지 단서를 달아두면, Decode의 "병렬화 안 됨"은 **요청 하나 안에서** 그렇다는 뜻이다.
요청이 여러 개면 각 요청의 decode step을 묶어서 동시에 처리할 수 있고, 그게 뒤에 나올 batching이다.

**이 구분이 이번 주 내용 중 가장 실용적이다.** 병목이 어디냐에 따라 최적화 전략이 완전히 갈린다.

- **긴 프롬프트** (500페이지 PDF 처리) → Prefill 병목 → 프롬프트 처리 속도 최적화
- **짧은 프롬프트 + 긴 생성** (챗봇, 스토리 생성) → Decode 병목 → 토큰 생성 속도·메모리 관리

Decode 구간에서 GPU가 노는 이유를 조금 더 구체적으로 보면 이렇다. 책 한 권 분량을 prefill로 밀어넣어
KV 캐시가 이미 다 올라가 있는 상태에서, 다음 토큰 **하나**를 만드는 연산량은 GPU 입장에서 너무 미미하다.
오자마자 처리해버린다. 그래서 병목이 연산이 아니라 **HBM에서 캐시를 끌어오는 구간**으로 옮겨간다.

그리고 중요한 함의 하나. **GPU utilization이 낮다고 GPU가 남는 게 아니다.**
memory bandwidth가 병목이면 연산 유닛이 놀아도 전체 성능은 제한된다.

#### 단계를 나누면 최적화 지점이 보인다

Prefill/Decode를 분리하는 순간 두 가지 최적화가 자연스럽게 따라온다.

**Prefix Caching (시스템 프롬프트 캐싱)**
프롬프트 앞부분은 반복되는 경우가 많다. 사내 서비스라면 **모든 사용자가 동일한 시스템 프롬프트**를
매 요청마다 함께 보낸다. 이걸 매번 새로 계산할 이유가 없다.

```
사용자 A의 첫 요청 → 시스템 프롬프트 구간의 KV 캐시 생성
사용자 B의 요청     → 같은 시스템 프롬프트 구간은 캐시 히트, prefill 건너뜀
```

앞에서 나온 Cache-Aware Routing이 필요해지는 이유가 이것이다. 캐시가 이미 올라간 인스턴스로
요청을 보내야 히트가 나기 때문이다.

**Chunked Prefill**
긴 프롬프트 하나가 prefill을 점유하면 뒤에 온 짧은 요청들이 전부 대기한다.
그래서 큰 prefill을 **청크 단위로 잘게 쪼개서** 사이사이 다른 요청을 끼워 넣는다.
vLLM에서 `--enable-chunked-prefill`로 켠다.

### P/D Disaggregation (Prefill-Decode 분리)

Prefill과 Decode를 **같은 GPU에 두면 두 지표를 동시에 최적화할 수 없다.**
GPU가 연산 집약적인 prefill로 바쁘면 decode 작업이 대기하고(ITL 증가), 반대도 마찬가지다.
게다가 단일 prefill 하나가 뒤의 모든 decode 요청을 밀어서 **P95/P99 tail latency를 악화**시킨다.

그래서 둘을 **다른 하드웨어에서 실행**하는 아키텍처가 나왔다.

- **전용 리소스 할당** — 각각 독립적으로 스케줄링·확장. 프롬프트 중복이 많은 워크로드(멀티턴 대화,
  에이전트)는 KV 캐시 재사용률이 높아 prefill 수요가 줄고, 그만큼 decode에 자원을 더 줄 수 있다.
- **병렬 실행** — 서로 간섭하지 않아 동시성과 처리량이 올라가고 tail latency가 개선된다.
- **독립적 튜닝** — TTFT와 ITL 목표에 맞춰 각각 다른 병렬화 전략(TP/PP)을 적용할 수 있다.

지원 구현: vLLM(experimental), SGLang, NVIDIA Dynamo, llm-d.

**다만 만능은 아니다.** prefill 노드에서 만든 KV 캐시를 decode 노드가 볼 수 있어야 하므로,
노드 간에 캐시를 공유해야 한다. 최근 **KV Cache Shared Pool** 같은 개념이 자주 등장하는 이유가 이것이다.
그리고 앞서 본 대로 노드 간 링크는 NVLink보다 18배 느리다. 명백한 트레이드오프다.

(P/D 분리는 GPU 노드가 최소 2대 필요해서 개인이 실습하기 어렵다. 스터디 후반의 llm-d 세션에서
AWS 기반으로 GPU 노드 2대 분산 추론 실습 환경이 제공될 예정이다.)

### PagedAttention — OS의 해법을 그대로 가져오다

2023년 이전 시스템은 요청마다 **도달 가능한 최대 시퀀스 길이만큼 연속된 HBM 영역을 미리 예약**했다.

- 토큰 예약 한도 4,096인데 실제 200만 쓰면 → **3,896개 분량이 요청이 끝날 때까지 낭비** = **내부 단편화**
- 길이가 제각각인 요청들 사이 여유 공간이 흩어져 → 전체 빈 공간은 충분한데 **연속된 자리가 없음**
  = **외부 단편화**

vLLM 팀 측정 기준, **KV 캐시의 실제 사용률이 20~38.2%**에 불과했다. 나머지는 전부 낭비.

해결은 운영체제가 수십 년 전에 이미 한 것을 거의 그대로 적용한 것이다.

- KV 캐시를 **16 또는 32 토큰 단위 고정 크기 블록**으로 나눈다
- 요청마다 **블록 테이블**을 두어 논리 블록 → 물리 블록으로 매핑한다
- 어텐션 커널은 테이블을 따라가며 하나의 연속된 시퀀스로 인식한다

결과:

- **내부 단편화** → 요청당 최대 하나의 부분 블록으로 축소
- **외부 단편화** → 블록 크기가 동일하고 어떤 블록이든 쓸 수 있으므로 **완전히 소멸**

확보된 메모리는 곧바로 **동시 요청 수 증가 = 처리량 증가**로 이어진다.
동일 지연시간에서 기존 시스템 대비 **2~4배 높은 처리량**.

보너스로 블록 테이블은 **공유**도 가능하게 한다. 하나의 프롬프트에서 여러 후보를 탐색하는
beam search에서 여러 시퀀스의 테이블이 **같은 물리 블록을 가리키다가, 분기해서 쓰기가 발생할 때만
복사**한다(copy-on-write). beam search에서 공유 비율이 **캐시의 최대 55%**에 달한다.

### FlashAttention

앞에서 KV 캐시가 HBM ↔ SRAM을 왕복한다고 했다. 그런데 문제는 그 왕복이 **한 번이 아니라는** 점이다.

```
[FlashAttention 없음]
  MatMul  → HBM에서 가져오고 → 계산 → HBM에 돌려주고
  Mask    → 가져오고 → 계산 → 돌려주고
  Softmax → 가져오고 → 계산 → 돌려주고
  MatMul  → 가져오고 → 계산 → 돌려주고
  (안 그래도 병목인 구간을 단계마다 왕복)

[FlashAttention]
  한 번 가져와서 → SRAM 안에서 전 과정 계산 → 한 번 돌려줌
```

**알고리즘을 다시 짜서 중간 결과를 굳이 HBM에 돌려주지 않고 SRAM에 들고 있게 만든 것**이다.
softmax·dropout·matmul을 **fused kernel**로 묶어 중간 저장을 없앴다.
계산 결과는 동일하고 데이터가 오가는 방식만 바뀌었는데, 병목 구간의 왕복이 사라지니 처리 시간이 줄어든다.

이걸 이해하려면 결국 ① KV 캐시가 무엇인지 ② HBM과 GPU SRAM 중 어디가 병목인지
③ 내 런타임과 GPU가 이 기능을 지원하는지를 전부 알아야 한다.
"좋다니까 켜자"와 "왜 좋은지 알고 켜자"는 다르다.

---

## Part 4. 서빙 프레임워크로 넘어가기

### 왜 직접 구현하지 않는가

지금까지 손으로 짠 추론 루프는 내부 이해에는 최고지만 프로덕션에는 부적합하다.
서빙 프레임워크(vLLM, SGLang)는 다음을 기본 제공한다.

- KV 캐시 재사용을 통한 효율적 디코딩
- 요청 스케줄링 (배치/마이크로배치)
- 다중 사용자 동시 처리
- 토큰 스트리밍, 요청 취소·중단

무엇보다 **PagedAttention, Speculative Decoding 같은 최신 연구 성과를 계속 통합**해준다.
모델 아키텍처마다 추론 로직을 다시 짜고 논문을 쫓아다닐 필요가 없다.

### vLLM 기본 사용

```python
from vllm import LLM, SamplingParams

llm = LLM(model="Qwen/Qwen2.5-0.5B", dtype="float16")
params = SamplingParams(temperature=0.8, top_p=0.95, max_tokens=128)
outputs = llm.generate([prompt], params)
```

OpenAI 호환 서버로 띄우면 바로 API가 된다.

```bash
vllm serve Qwen/Qwen2.5-0.5B \
  --dtype bfloat16 \
  --max-model-len 32768 \
  --gpu-memory-utilization 0.8
```

프로덕션급 옵션 예시 (gpt-oss 20B, H100 80GB):

```bash
python -m vllm.entrypoints.openai.api_server \
  --model openai/gpt-oss-20b \
  --dtype bf16 \                     # 모델 계산 정밀도
  --gpu-memory-utilization 0.9 \     # GPU 메모리 90% 사용 목표
  --max-num-seqs 16 \                # 최대 동시 시퀀스 수
  --max-num-batched-tokens 16384 \   # 한 배치의 최대 토큰 수
  --tensor-parallel-size 2           # GPU 2개에 모델 분산
  # PagedAttention은 기본 활성화
```

**이 옵션 하나하나가 잠재적 최적화 포인트**라는 게 Part 1에서 말한 "설정값이 성능을 좌우한다"의 실체다.

> 위 예시들은 교재 기준 표기다. 최신 vLLM에서는 `python -m vllm.entrypoints.api_server` 형태가
> 정리되고 **`vllm serve`** 로 통일되는 추세다. vLLM은 버전 간 인터페이스 변화가 잦은 편이라,
> 실제로 돌릴 때는 설치된 버전의 문서를 기준으로 확인해야 한다.

vLLM 아키텍처는 이렇게 구성된다.

| 컴포넌트 | 역할 |
|---|---|
| LLMEngine | 사용자 API와 engine core 사이 orchestration |
| EngineCore | scheduler, KV cache manager, model executor 조율 |
| Scheduler | 요청을 어떤 iteration에 넣을지 결정 |
| ModelExecutor | worker process 관리, 분산 실행 |
| GPUWorker / GPUModelRunner | GPU에서 실제 forward 수행 |

### vLLM vs Hugging Face

교재에 실린 비교로는 같은 GPU·모델·프롬프트에서 **vLLM 1.12초 vs HF 19.58초(약 17배)** 차이가 났다.
단일 요청 기준인데도 이만큼 벌어지는 이유는 엔진 구조에 있다.

1. **PagedAttention** — HF는 시퀀스마다 연속 메모리를 통째로 할당해 비효율적
2. **CUDA Graph 캡처** — vLLM은 초기화 시 실행 그래프를 미리 캡처해 커널 실행 오버헤드 제거.
   HF `generate()`는 매 토큰마다 Python 레벨 eager 모드
3. **최적화된 커널** — FlashAttention 등을 직접 사용
4. **pipeline() 자체 오버헤드** — 전/후처리, 텐서 변환이 매 호출마다 추가
5. **연속 배칭** — 단일 요청에선 체감이 적지만, 동시 요청이 늘수록 격차가 벌어진다

### Streaming

`llm.generate()`는 **동기·블로킹**이다. 내부적으로는 토큰을 하나씩 만들면서도 API는 전체가
완성될 때까지 기다렸다 한 번에 반환한다. 챗봇에서는 UX가 무너진다.

```python
engine = AsyncLLMEngine.from_engine_args(engine_args)   # ← LLM 대신 AsyncLLMEngine

results_generator = engine.generate(prompt, sampling_params, request_id)

async for request_output in results_generator:          # ← 토큰이 나오는 대로 수신
    for chunk in request_output.outputs:
        print(chunk.text, end="", flush=True)
```

`LLM` 대신 `AsyncLLMEngine`을 쓰고, `generate()`가 반환하는 비동기 스트림을 `async for`로 받는 게 핵심이다.
웹 서비스라면 이 루프 안에서 토큰 청크를 그대로 yield 하면 된다.

부가 이점으로 **중간 취소**가 가능하다. `engine.abort(request_id)`로 원하지 않는 방향의 생성을
끊으면 UX뿐 아니라 **컴퓨팅 자원 절약**에도 직접 기여한다.

주의할 점: **스트리밍은 TTFT를 낮게 "느끼게" 만들 뿐, backend의 총 decode 비용은 그대로다.**
프론트엔드 UX와 백엔드 capacity planning은 분리해서 봐야 한다.

### Batching

여러 요청을 묶어 **한 번의 forward pass**에 통과시킨다. Transformer에서 특히 잘 먹히는 이유:

- 행렬곱과 attention이 시퀀스 차원으로 병렬화 가능
- **모델 가중치를 모든 요청이 공유**하므로 배치를 키워도 가중치를 다시 읽어올 필요가 없다
  (메모리 대역폭 재사용)
- GPU는 원래 대규모 병렬 연산용 하드웨어

코드 상으로는 프롬프트를 리스트로 한 번에 넘기느냐, 하나씩 반복 호출하느냐의 차이다.

```python
llm.generate(prompts, sampling_params)          # 배치: 4개를 한 배치로 묶어 처리
for p in prompts: llm.generate([p], params)     # 순차: generate 호출이 4번
```

여기서 미리 짚어둘 점 하나. **프롬프트 N개를 묶는다고 N배 빨라지지는 않는다.** 이유는 세 가지다.

- 프롬프트 길이가 서로 달라 배치 내 가장 긴 시퀀스에 맞춰 스케줄링해야 함
- 출력 길이도 제각각이라 먼저 끝나는 시퀀스가 생기고 완전한 병렬 효율이 안 나옴
- 배치가 작으면 애초에 GPU를 포화시키지 못함 — 배치를 키울수록 상대적 이득이 커진다

즉 배칭의 목적은 개별 요청의 latency 단축이 아니라 **GPU 활용률과 전체 throughput 향상**이다.

#### 배칭 전략 3종

| 전략 | 동작 | 문제 |
|---|---|---|
| **Static** | 배치가 꽉 찰 때까지 대기 후 시작 | 먼저 온 요청이 오래 기다림 |
| **Dynamic** | 꽉 차거나 timeout 도달 시 시작 | 대기시간을 시간 제한으로 완화 |
| **Continuous** | 슬롯이 비면 새 요청을 즉시 투입 (토큰 단위) | — |

Static batching의 근본 한계는 **배치 안에서 짧은 응답이 먼저 끝나도 GPU 슬롯이 비지 않고
배치 전체가 끝날 때까지 기다린다**는 점이다.

**Continuous batching**은 요청이 완료되는 즉시 그 자리에 새 요청을 끼워 넣어 GPU를 계속 바쁘게 유지한다.
Anyscale의 2023년 연구에서 **최대 23배 처리량 향상**과 p50 지연시간 대폭 감소를 보고했다.
vLLM, SGLang, TensorRT-LLM(여기선 "in-flight batching")이 이를 구현한다.

#### 배치 크기와 오토스케일링

배치 크기를 키우면 **처리량은 올라가지만 개별 사용자의 지연시간은 나빠진다.** 정답이 없어서
모델·인스턴스·지연시간 목표·예산에 맞춰 여러 값으로 테스트해야 한다.

오토스케일링과 연결할 때는 **오토스케일러의 concurrency target과 레플리카의 batch size를 일치**시켜야 한다.

- 활성 레플리카가 모두 최대 동시성에 도달 → **scale up**
- 레플리카들이 절반만 찬 배치로 계속 처리 중 → **scale down**

---

## 스스로 답해본 질문

스터디에서 제시된 세 가지 질문에 대한 정리.

**Q1. 가중치는 어떻게 "학습"되며, 그 숫자들이 모델의 판단 능력을 어떻게 결정하는가?**

가중치는 랜덤에서 시작해 **오차를 줄이는 방향으로 반복 수정**된다. 순전파로 예측하고, 정답과의 오차를
구하고, 역전파로 출력에 가까운 층부터 거꾸로 가중치를 고친다. 이때 얼마나 어느 방향으로 고칠지
알려주는 게 gradient이고, `새 W = W − 학습률 × gradient`로 갱신된다. 키/몸무게 신경 하나든 GPT-3의
1,750억 개든 이 아이디어 하나를 규모만 키워 반복한 것이다. 학습이 끝나면 그 숫자 배열 자체가 곧
모델의 판단 기준이 된다.

**Q2. 문장 하나가 GPT-2에 입력되어 다음 단어 하나가 나오기까지 데이터는 어떤 shape으로 변환되는가?**

```
텍스트 → 토큰화 → [배치, 토큰]
      → 토큰 임베딩(768) + 위치 인코딩 → [배치, 토큰, 768]
      → Transformer Block × 12 (Masked MHA + FFN, 768 유지) → [배치, 토큰, 768]
      → LM Head → [배치, 토큰, 50257]  = logits
      → 마지막 토큰의 logits에 Temperature/Top-k/Top-p → Softmax → 다음 토큰
```

중간에 FFN이 768 → 3,072 → 768로 넓혔다 줄이지만, Block을 나올 때의 차원은 항상 768로 돌아온다.

**Q3. 왜 GPU에서 병렬 처리가 가능하며, 이것이 dtype·양자화 선택과 어떤 관계인가?**

GPU는 SIMT 구조로 스레드를 warp 단위로 묶어 처리하고, warp가 메모리 지연을 만나면 다른 warp로 즉시
전환해 연산을 놀리지 않는다. Transformer는 이 구조에 잘 맞는다. 한 Block 안의 토큰별 FFN 연산은
서로 독립적이고, multi-head attention의 12개 head도 서로 독립적이라 병렬화가 된다
(단, Block 12개 사이는 순차 의존이라 여기는 병렬화되지 않는다).

dtype·양자화와의 연결은 **decode가 memory-bound라는 사실**에서 나온다. 연산 유닛이 남아도 HBM에서
가중치와 KV 캐시를 읽어오는 대역폭이 병목이면 전체가 느려진다. fp16으로 낮추면 읽어올 바이트가 절반이
되어 실질 속도가 오르고, Tensor Core라는 전용 하드웨어까지 탄다. 양자화는 이걸 더 밀어붙여 4~8배를
줄이지만, 정수 칸에 끼워 맞추는 방식이라 정확도 손실이 커서 GPTQ·AWQ 같은 보정이 필요하다.

---

## 정리

이번 주에 얻은 가장 큰 소득은 **최적화 기법들이 서로 독립적인 트릭이 아니라 하나의 인과 사슬**이라는 걸
알게 된 것이다.

```
Transformer가 autoregressive하다
  → 매 스텝 전체 시퀀스 재계산은 O(L²D)로 폭발한다
    → KV Cache로 O(LD)로 낮춘다
      → 대신 GPU 메모리를 먹는다
        → 연속 할당은 20~38%만 실제 사용된다 (내부/외부 단편화)
          → PagedAttention으로 블록 단위 관리한다
            → 확보된 메모리 = 더 많은 동시 요청 = 처리량 2~4배

동시에,
Prefill은 compute-bound / Decode는 memory-bound다
  → 한 GPU에 두면 서로 간섭해 TTFT와 ITL을 동시에 못 잡는다
    → P/D Disaggregation으로 분리한다
  → Decode의 대역폭 병목은 dtype·양자화로 완화한다
  → GPU 유휴 시간은 Continuous Batching으로 메운다
```

그래서 "vLLM 쓰면 빠르다"가 아니라 **어떤 병목 때문에 어떤 기법이 나왔는지**를 알아야
새 기술이 나왔을 때 판단할 수 있다는 Part 1의 메시지가 마지막에 다시 이해된다.

다음 노트([서빙 엔진 해부와 프로덕션 서빙 아키텍처](serving-engine-and-production-architecture.md))에서는 이 서빙 능력을 실제 웹 서비스로 감싸는 API 설계와 시스템 설계를 다룬다.

---

## 이어서 볼 것

이번 정리는 개념과 아키텍처에 집중했다. 다음 순서로 깊이를 더할 계획이다.

- GQA / MQA가 KV 캐시를 줄이는 원리
- Speculative Decoding — 이번 글에서는 이름만 언급하고 넘어갔다
- MoE 라우팅이 실제 서빙 지표(TTFT / ITL)에 미치는 영향
- 가중치 공개 모델과 비공개 모델의 서빙 아키텍처 차이 (self-host vs API)

---

## 참고 자료

**교재·실습 코드**
- Hands-On LLM Serving and Optimization (2026)
- 실습 노트북: https://github.com/orca3/llm-model-inference
  - `ch02/ch2_Inside_the_Mind_of_a_Transformer.ipynb`
  - `ch02/ch2_Workthrough_LLM_execution.ipynb`
  - `ch02/ch2_Run_LLM_With_vLLM.ipynb`
  - `ch02/ch2_Streaming.ipynb`
  - `ch02/ch2_Batching.ipynb`

**논문·문서**
- Attention Is All You Need (2017)
- vLLM 공식 문서: https://docs.vllm.ai
- llm-d: https://llm-d.ai/docs
- KServe: https://kserve.github.io/website/
- Envoy AI Gateway: https://github.com/envoyproxy/ai-gateway

**시각 자료**
- Transformer Explainer (GPT-2 small 인터랙티브): https://poloclub.github.io/transformer-explainer/
  — 백엔드에서 실제 GPT-2 small이 돌고 그 결과를 실시간 렌더링하는 것이라 값이 진짜다.
  단계별로 직접 입력해보는 게 이 장을 이해하는 가장 빠른 길이었다.
- 3Blue1Brown 한국어 — 트랜스포머(DL5), 어텐션(DL6)
- 딥러닝 큐레이터 임커밋 — KV cache, prefill, Flash attention, PagedAttention
- bRd 3D — 인공지능의 작동방식, CPU/GPU는 어떻게 작동할까
