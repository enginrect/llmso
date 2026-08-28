# LLM 서빙의 병목과 필수 최적화 기법 — 실습 기록

모든 측정은 RTX 4090 24GB / WSL2 Ubuntu 24.04에서 했다 ([환경 구축 기록](environment-setup.md), [개념 정리](../serving-bottlenecks-and-optimization.md)).

**이 실습이 4090에 가장 잘 맞는다.**
병목 진단 파트는 이론(GPU 스펙, 산술 강도)이라 실습이 없고, 실습은 전부 **최적화 기법 측정**이다.
그리고 그 기준 모델인 **Qwen2.5-7B-Instruct는 24GB에 BF16 원본까지 들어간다.**

| 실습 | 필요 자원 | 4090 판정 |
|---|---|---|
| ① KodeKloud 무료 온라인 랩 (Task 1~8) | **브라우저만** | ✅ 로컬 자원 불필요 |
| ② Hands-on quantization 벤치마크 | GPU 24GB | ✅ **핵심 실습** |
| ③ EC2 스팟 자동화 벤치마크 | AWS 계정 | ⚠️ **로컬 재현 가능** |
| ④ 도전과제 10종 (배칭/캐싱/커널 튜닝) | GPU | ✅ 대부분 가능 |

---

## 실습 기록 (Labs)

### Lab 1 — KodeKloud vLLM 랩 ⬜ *(무료, 브라우저)*

- 랩: https://learn.kodekloud.com/learn/courses/youtube-labs-vllm
- 소개 영상: https://www.youtube.com/watch?v=qdPkA5mxLhg

**브라우저에서 도는 무료 랩이다. 로컬 GPU가 전혀 필요 없다.**
(랩 환경 자체가 `vllm 0.27.1+cpu` / `torch 2.13.0+cpu` 로 CPU 빌드다.)

**모델**: `HuggingFaceTB/SmolLM-135M` (135M 파라미터, 약 538MB)
**사전 구성**: Python venv + vLLM + transformers + Gradio, 스크립트는 `/root/code/`

```bash
source /root/venv/bin/activate
python /root/code/verify_environment.py
```


#### Task 목록

| Task | 제목 | 배우는 것 |
|---|---|---|
| **1** | Naive HuggingFace Inference — The Baseline | HF transformers로 tok/s 측정. **vLLM이 넘어야 할 기준선** |
| **2** | vLLM Offline Inference — See the Difference | 같은 모델·같은 프롬프트를 vLLM으로. 나란히 비교 |
| **3** | The KV Cache Problem — Why Memory Matters | 기존 방식이 **최악의 경우로 미리 할당** → 단편화로 메모리 60~80% 낭비 |
| **4** | PagedAttention — vLLM's Solution | OS 페이징과 동일한 발상. 활용률 약 20% → 약 95% |
| **5** | Launch vLLM as an OpenAI-Compatible API Server | `/v1/completions`, `/v1/chat/completions`. **base_url만 바꾸면 기존 앱이 그대로 동작** |
| **6** | Multi-User Throughput Under Load | 동시 요청 스트레스 테스트. **개인 tok/s와 전체 처리량은 다르다** |
| **7** | Tuning vLLM Parameters for Production | `max_model_len` / `max_num_seqs` / `swap_space` 3종 튜닝 |
| **8** | Production Monitoring Dashboard (Capstone) | Gradio로 실시간 지표 대시보드 구축 |

#### Task 7이 특히 중요하다

세 파라미터의 트레이드오프를 직접 만진다:

| 파라미터 | 의미 | 낮추면 |
|---|---|---|
| `max_model_len` | 요청당 최대 컨텍스트 길이 | 요청당 메모리 감소 |
| `max_num_seqs` | 배치 내 동시 시퀀스 최대 수 | 메모리 압박 감소, 처리량 감소 |
| `swap_space` | KV cache 오버플로용 CPU 스왑(GB) | — (올리면 용량 확장) |

랩에서 비교하는 3개 구성:
```
A: 기본값
B: Shorter Context      max_model_len=64,  max_num_seqs=256, swap_space=1
C: Limited Concurrency  max_model_len=64,  max_num_seqs=8,   swap_space=1
```

> 워크로드에 따라 정답이 다르다는 게 핵심이다.
> **짧은 질문의 고객지원 봇**과 **긴 문맥의 문서분석 서비스**는 튜닝이 정반대다.

**→ 이 랩부터 하는 걸 권한다.** 무료고, 환경 구축이 필요 없고, 이후 로컬 실습의 개념 토대가 된다.

---


#### 진행 기록

**미진행 (브라우저 랩 — 직접 진행 필요).** KodeKloud는 계정 로그인 + 브라우저 세션이 필요한
호스팅 랩이라 이 세션에서 대신 진행할 수 없다. 로컬 자원이 필요 없으므로 아무 때나 브라우저로
진행하면 된다. 개념 토대(PagedAttention, 파라미터 튜닝)는 Lab 2~4의 로컬 실측으로 대부분 커버됐다.

- key point: (미진행 — 브라우저에서 직접 진행)


### Lab 2 — Hands-on quantization 벤치마크 ✅ *(핵심)*

**원본**: 교재 Colab. **모델**: `Qwen/Qwen2.5-7B-Instruct`
**비교 대상 3종**: 원본(BF16) vs **GPTQ W4A16** vs **FP8 W8A8**


#### 4090 판정: ✅ 가장 잘 맞는 실습

- 4090은 **Ada Lovelace (CC 8.9)** → vLLM의 FP8 W8A8이 **공식 지원되는 최소 사양**이다 (CC 8.9 이상 = Ada/Hopper). FP8 Tensor Core를 실제로 탑재하고 있다.
- GPTQ INT4는 Marlin 커널로 Ampere 이상 지원 → 문제없다.
- VRAM: BF16 15.2GB / FP8 7.6GB / INT4 5.6GB — **3종 모두 24GB에 들어간다.**

#### 실행

이미 양자화된 모델이 HuggingFace에 올라와 있으므로 **직접 양자화할 필요가 없다.**

```bash
# 원본
vllm serve Qwen/Qwen2.5-7B-Instruct --max-model-len 4096 --gpu-memory-utilization 0.90

# GPTQ-Int4
vllm serve Qwen/Qwen2.5-7B-Instruct-GPTQ-Int4 --max-model-len 4096

# FP8 (dynamic)
vllm serve <FP8 variant> --max-model-len 4096
```

부하 생성은 vLLM 내장 벤치마크가 가장 간편하다:
```bash
vllm bench serve \
  --model Qwen/Qwen2.5-7B-Instruct \
  --dataset-name random --num-prompts 300 --max-concurrency 100
```

측정 지표: **TTFT / TPOT(ITL) / request throughput** — 교재 Figure 6-17~19와 같은 3종.

#### 반드시 저·고 동시성 **양쪽**을 재야 한다

이 실습의 결론이 동시성에 따라 **뒤집히기 때문**이다.

| 상황 | 승자 | 이유 |
|---|---|---|
| **저동시성** | **GPTQ W4A16** (원본 대비 약 300% 개선) | 병목이 **메모리 대역폭**. 가중치를 4배 줄인 효과가 그대로 나온다. FP8도 약 150% 개선 |
| **고동시성** | **FP8 W8A8** | W4A16은 **연산이 여전히 16비트**인데다 **역양자화 오버헤드**까지 붙어 TTFT가 원본보다도 느려진다. FP8은 활성화까지 양자화해 연산 자체가 빨라진다 |

**부하 조합을 최소 4단계**로 잡는 걸 권한다 (동시성:프롬프트수):
```
1:10   →   10:10   →   100:100   →   300:300
```

> 실무 해석: 챗봇·에이전트처럼 **지연시간이 핵심**이면 W4A16,
> 배치를 밀어붙여 **처리량으로 비용을 낮추는** 게 목표면 W8A8.

#### ⚠️ 4090에서 나올 결과가 교재와 다를 수 있다

교재/스터디는 L4(g6.2xlarge)와 Colab T4 기준이다. 4090은 **VRAM은 같은 24GB지만 대역폭이 1,008 GB/s로 L4(약 300 GB/s)의 3배 이상**이다.

→ **절대 수치는 훨씬 좋게 나온다. 비교해야 할 것은 3종 사이의 상대적 경향이다.**
그리고 대역폭이 넉넉한 만큼 **W4A16의 이득이 교재보다 작게 나올 가능성**이 있다(병목이 대역폭에서 연산으로 이동). 이건 오히려 좋은 노트 소재다.

#### (참고) 직접 양자화하는 경로

```python
from transformers import AutoModelForCausalLM, AutoTokenizer, GPTQConfig

tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B-Instruct")
dataset = ["Gptq is an easy-to-use model quantization library ..."]
gptq_config = GPTQConfig(bits=4, dataset=dataset, tokenizer=tokenizer)

quantized_model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen2.5-7B-Instruct", device_map="auto", quantization_config=gptq_config
)
```
캘리브레이션에 시간이 오래 걸린다. 학습 목적이 아니면 **HF의 기성 양자화 모델을 쓰는 게 낫다.**

---


#### 실행 기록

실행 방식: Lab 3의 자동화 리포 방법론을 그대로 로컬화한 `bench_local.sh` 하나로 Lab 2·3을 함께 진행했다.
원본 `bench_worker.py`와 동일 구성 — 모델 3종 순차 서빙(`--max-model-len 8192 --gpu-memory-utilization 0.92`),
부하 4점 `1:10, 10:10, 100:100, 300:300`, ShareGPT 데이터셋, `vllm bench serve` (ttft/tpot/itl, p50/95/99),
env `VLLM_USE_FLASHINFER_SAMPLER=0`. 모델도 원본 config와 동일 3종이다.

실행 로그 (발췌):

```shell
[22:06:54] ==== 모델: Qwen/Qwen2.5-7B-Instruct ====
[22:08:00] 서버 준비 완료
(EngineCore pid=36765) INFO 08-22 22:07:43 [kv_cache_utils.py:2235] GPU KV cache size: 102,544 tokens
...
VRAM used: 22951 MiB
...
Request throughput (req/s):              0.22      
Output token throughput (tok/s):         57.46     
Mean TTFT (ms):                          56.87     
Median TTFT (ms):                        27.84     
P99 TTFT (ms):                           266.58    
Mean TPOT (ms):                          17.31     
Median TPOT (ms):                        17.36     
P99 TPOT (ms):                           17.66     
...
Request throughput (req/s):              0.57      
Output token throughput (tok/s):         149.73    
Mean TTFT (ms):                          3711.54   
Median TTFT (ms):                        3711.60   
P99 TTFT (ms):                           3712.56   
Mean TPOT (ms):                          21.90     
Median TPOT (ms):                        19.33     
P99 TPOT (ms):                           40.11     
...
Request throughput (req/s):              5.83      
Output token throughput (tok/s):         1278.03   
Mean TTFT (ms):                          1366.58   
Median TTFT (ms):                        1400.99   
P99 TTFT (ms):                           2363.60   
Mean TPOT (ms):                          43.73     
Median TPOT (ms):                        26.88     
P99 TPOT (ms):                           192.29    
...
Request throughput (req/s):              13.60     
Output token throughput (tok/s):         2957.80   
Mean TTFT (ms):                          1972.22   
Median TTFT (ms):                        1509.09   
P99 TTFT (ms):                           4827.42   
Mean TPOT (ms):                          74.22     
Median TPOT (ms):                        41.54     
P99 TPOT (ms):                           200.79    
...
[22:10:21] ==== 모델: Qwen/Qwen2.5-7B-Instruct-GPTQ-Int4 ====
...
(EngineCore pid=38146) INFO 08-22 22:10:59 [kv_cache_utils.py:2235] GPU KV cache size: 288,656 tokens
...
VRAM used: 23859 MiB
...
Request throughput (req/s):              0.54      
Output token throughput (tok/s):         142.60    
Mean TTFT (ms):                          43.04     
Median TTFT (ms):                        20.54     
P99 TTFT (ms):                           193.49    
Mean TPOT (ms):                          6.93      
Median TPOT (ms):                        6.82      
P99 TPOT (ms):                           7.65      
...
Request throughput (req/s):              1.08      
Output token throughput (tok/s):         287.73    
Mean TTFT (ms):                          3780.11   
Median TTFT (ms):                        3780.19   
P99 TTFT (ms):                           3781.14   
Mean TPOT (ms):                          9.23      
Median TPOT (ms):                        7.86      
P99 TPOT (ms):                           16.68     
...
Request throughput (req/s):              9.31      
Output token throughput (tok/s):         2026.12   
Mean TTFT (ms):                          1461.82   
Median TTFT (ms):                        1469.55   
P99 TTFT (ms):                           2495.23   
Mean TPOT (ms):                          43.39     
Median TPOT (ms):                        24.67     
P99 TPOT (ms):                           206.46    
...
Request throughput (req/s):              13.84     
Output token throughput (tok/s):         2964.93   
Mean TTFT (ms):                          2194.22   
Median TTFT (ms):                        1633.58   
P99 TTFT (ms):                           5727.66   
Mean TPOT (ms):                          91.77     
Median TPOT (ms):                        57.09     
P99 TPOT (ms):                           239.43    
...
[22:13:00] ==== 모델: RedHatAI/Qwen2.5-7B-Instruct-FP8-dynamic ====
...
(EngineCore pid=39338) INFO 08-22 22:13:43 [kv_cache_utils.py:2235] GPU KV cache size: 230,352 tokens
...
VRAM used: 23792 MiB
...
Request throughput (req/s):              0.31      
Output token throughput (tok/s):         82.80     
Mean TTFT (ms):                          46.17     
Median TTFT (ms):                        21.08     
P99 TTFT (ms):                           224.48    
Mean TPOT (ms):                          11.92     
Median TPOT (ms):                        11.93     
P99 TPOT (ms):                           12.09     
...
Request throughput (req/s):              0.76      
Output token throughput (tok/s):         202.42    
Mean TTFT (ms):                          3666.80   
Median TTFT (ms):                        3666.86   
P99 TTFT (ms):                           3667.65   
Mean TPOT (ms):                          15.08     
Median TPOT (ms):                        12.99     
P99 TPOT (ms):                           26.92     
...
Request throughput (req/s):              7.96      
Output token throughput (tok/s):         1704.26   
Mean TTFT (ms):                          1072.96   
Median TTFT (ms):                        954.35    
P99 TTFT (ms):                           1705.07   
Mean TPOT (ms):                          29.52     
Median TPOT (ms):                        18.55     
P99 TPOT (ms):                           125.27    
...
Request throughput (req/s):              2.48      
Output token throughput (tok/s):         539.47    
Mean TTFT (ms):                          10325.27  
Median TTFT (ms):                        8444.57   
P99 TTFT (ms):                           26668.31  
Mean TPOT (ms):                          405.88    
Median TPOT (ms):                        212.77    
P99 TPOT (ms):                           1406.31   
[22:17:37] RedHatAI/Qwen2.5-7B-Instruct-FP8-dynamic 종료
[22:17:38] ==== 전체 벤치마크 완료 ====
11 /home/enginrect/vllm-quantization-hands-on-benchmark/local-results/vllm_bench.json
```

#### 결과 요약 (plot_results.py 생성 표)

| 모델 | 동시성 | Median TTFT (ms) | P99 TTFT (ms) | Median TPOT (ms) | Output tok/s | 총 소요(s) |
|---|---|---|---|---|---|---|
| BF16 | 1 | 27.8 | 266.6 | 17.4 | 57.5 | 46.3 |
| BF16 | 10 | 3,711.6 | 3,712.6 | 19.3 | 149.7 | 17.6 |
| BF16 | 100 | 1,401.0 | 2,363.6 | 26.9 | 1,278.0 | 17.1 |
| BF16 | 300 | 1,509.1 | 4,827.4 | 41.5 | 2,957.8 | 22.1 |
| FP8-dynamic | 1 | 21.1 | 224.5 | 11.9 | 82.8 | 32.2 |
| FP8-dynamic | 10 | 3,666.9 | 3,667.7 | 13.0 | 202.4 | 13.2 |
| FP8-dynamic | 100 | 954.4 | 1,705.1 | 18.6 | 1,704.3 | 12.6 |
| FP8-dynamic | 300 | 8,444.6 | 26,668.3 | 212.8 | 539.5 | 120.8 |
| GPTQ-Int4 | 1 | 20.5 | 193.5 | 6.8 | 142.6 | 18.7 |
| GPTQ-Int4 | 10 | 3,780.2 | 3,781.1 | 7.9 | 287.7 | 9.3 |
| GPTQ-Int4 | 100 | 1,469.6 | 2,495.2 | 24.7 | 2,026.1 | 10.7 |
| GPTQ-Int4 | 300 | 1,633.6 | 5,727.7 | 57.1 | 2,964.9 | 21.7 |

VRAM·KV cache 실측 (서버 로그):

| 모델 | GPU KV cache | VRAM (서빙 중) |
|---|---|---|
| BF16 | 102,544 tokens | 22,951 MiB |
| GPTQ-Int4 | 288,656 tokens | 23,859 MiB |
| FP8-dynamic | 230,352 tokens | 23,792 MiB |

#### 판정 — 교재 명제와 대조

- **저동시성(c=1): W4A16 승자 재현.** TPOT 기준 GPTQ 6.8ms vs BF16 17.4ms = **2.5배** (교재 "약 300% 개선"과 부합). FP8은 11.9ms = 1.46배 (교재 "약 150%"와 부합). 병목이 메모리 대역폭일 때 가중치 4배 축소가 그대로 속도가 된다.
- **고동시성(c=300): 4090에서는 양자화 이점이 소멸.** BF16 2,958 ≈ GPTQ 2,965 tok/s. 이 문서 위쪽의 예측("대역폭이 넉넉한 만큼 W4A16의 이득이 교재보다 작게 나올 가능성 — 병목이 대역폭에서 연산으로 이동")이 실측으로 적중했다. L4 실측(아래 Lab 3)에서는 c300에서도 GPTQ 1,367 > BF16 905로 양자화가 이긴다 — **같은 벤치, 대역폭 3.4배 차이가 승자를 바꾼다.**

#### ⚠️ FP8-dynamic c=300 이상 현상 (미해결)

FP8만 c300에서 539 tok/s로 붕괴했다(BF16/GPTQ의 1/5.5). 원인 조사:

```shell
# 재현성 확인 — 서버 재기동 후 c300 2회 연속 (1회차=웜업 가정)
=== c300 1회차 === Request throughput 0.48 req/s / Median TTFT 110,711ms / Median TPOT 1,169.9ms
=== c300 2회차 === Request throughput 0.67 req/s / Median TTFT  7,097ms / Median TPOT   694.1ms
→ 2회차도 느리다. 웜업 아티팩트 아님.

# 배제한 가설들
- Preemption: 서버 로그에 0건
- 온도/스로틀: idle 45°C, 로그에 경고 없음
- RAM 부족: 31GiB 중 1.5GiB 사용
- JIT 컴파일: _topk_topp_kernel 1회 경고뿐 (일회성)

# 남는 정황
- c100까지는 정상(1,704 tok/s, 3모델 중 TTFT 최저)이고 c300에서만 무너진다
- peak 2,150 tok/s는 찍힌다 — 간헐적 스톨 패턴 (P99 ITL 1,898ms)
- 같은 모델·같은 vLLM 0.27.1로 네이티브 Linux L4에서는 c300 정상(1,351 tok/s)
```

> **⚠️ 왜 실패했나 (유력)** — **실행 환경 차이.** 교재 벤치와 Lab 3의 L4는 네이티브 Linux인데
> 여기만 WSL2다. 하드웨어(온도·RAM·preemption) 가설은 전부 배제됐고, 같은 바이너리가
> 네이티브 Linux에서 정상이므로 남는 변수는 WSL2 계층뿐이다. 단 메커니즘 특정은 못 했다.

**판정 보류**: WSL2 + 4090 + vLLM 0.27.1 조합에서 FP8-dynamic이 초고동시성에서 재현성 있게
무너진다는 것까지가 확인된 사실이다. 네이티브 Linux와의 차이가 정황상 유력하지만 근본 원인은
특정하지 못했다. (스터디 노트 소재: "벤치마크는 환경의 함수다"의 실례)

- key point: 저동시성 W4A16 2.5배·FP8 1.5배(교재 부합), 고동시성에서 4090은 BF16≈GPTQ(대역폭 여유가 양자화 이점을 지움 — L4와 승자가 다르다), FP8 c300 붕괴는 3회 재현·원인 미확정. KV cache 실측: INT4 288k > FP8 230k > BF16 102k tokens — 가중치가 작을수록 남는 VRAM이 전부 KV cache가 된다.


### Lab 3 — 양자화 3종 벤치마크 자동화 ✅ *(로컬 재현)*

- 리포: https://github.com/icebreaker70/vllm-quantization-hands-on-benchmark (스터디 멤버 공개 자료)
- 목표: **Qwen2.5-7B 동일 모델, 양자화만 다르게** — `BF16` vs `FP8-dynamic` vs `GPTQ-Int4`

원래 구성은 AWS 자동화다:
```
인스턴스   g6.2xlarge 스팟 (L4 24GB)
부하 조합   1:10, 10:10, 100:100, 300:300 (동시성:프롬프트수)
소요        기동~벤치~종료까지 약 38분
결과 저장   S3
```


#### 4090 로컬 대체 ✅

**`g6.2xlarge`의 L4는 VRAM 24GB — 4090과 같은 급이다.** 즉 이 벤치마크는 **4090에서 그대로 재현 가능**하다.
AWS로 하려면 계정과 S3 버킷이 필요하지만, 로컬로 돌리면:

- EC2 기동/종료 자동화(`run_benchmark_ec2.py`)와 S3 업로드 부분만 걷어내면 된다
- 벤치마크 로직과 시각화(`matplotlib`) 부분은 그대로 쓸 수 있다
- **비용이 들지 않고, EC2 기동/종료를 기다릴 필요가 없다**

```bash
git clone https://github.com/icebreaker70/vllm-quantization-hands-on-benchmark
cd vllm-quantization-hands-on-benchmark
python3 -m venv venv && source venv/bin/activate
pip install matplotlib numpy      # boto3는 로컬 실행 시 불필요
```

> 실습 ②와 내용이 겹친다. **②를 수동으로 한 뒤, ③의 스크립트로 자동화**하는 순서가 자연스럽다.

---


#### 실행 기록

리포 clone 후 EC2/S3 오케스트레이션(`run_benchmark_ec2.py`)을 걷어내고 `bench_worker.py`의 벤치 루프만
`bench_local.sh`로 옮겼다 (실행 결과는 Lab 2에 기록). 시각화는 리포의 `plot_results.py`를 그대로 사용:

```shell
$ python plot_results.py --results local-results/vllm_bench.json --out-dir local-charts --title-suffix "Qwen2.5-7B-Instruct on RTX 4090"
모델 3종 × 동시성 [1, 10, 100, 300] = 12건
저장: local-charts/quant-bench-ttft.png
저장: local-charts/quant-bench-tpot.png
저장: local-charts/quant-bench-throughput.png
저장: local-charts/summary.md
total 224
drwxr-xr-x 2 enginrect enginrect  4096 Aug 22 22:19 .
drwxr-xr-x 7 enginrect enginrect  4096 Aug 22 22:19 ..
-rw-r--r-- 1 enginrect enginrect 76335 Aug 22 22:19 quant-bench-throughput.png
-rw-r--r-- 1 enginrect enginrect 65010 Aug 22 22:19 quant-bench-tpot.png
-rw-r--r-- 1 enginrect enginrect 70068 Aug 22 22:19 quant-bench-ttft.png
-rw-r--r-- 1 enginrect enginrect   852 Aug 22 22:19 summary.md
```

생성물: `local-charts/quant-bench-{ttft,tpot,throughput}.png` + `summary.md`
(차트 PNG는 바이너리라 저장소 exclude 대상 — 수치는 위 표가 원본이다)

#### 4090 vs L4 — 같은 벤치, 다른 하드웨어

리포에는 원저자의 **L4(g6.2xlarge) 실측 결과**(`results/20260817-121956/`)가 들어 있어 직접 비교가 된다.
둘 다 24GB, CC 8.9인데 대역폭만 3.4배 다르다 (L4 300 GB/s vs 4090 1,008 GB/s):

| 지표 | L4 | 4090 | 배율 |
|---|---|---|---|
| BF16 c=1 TPOT | 56.7ms | 17.4ms | 3.3배 |
| GPTQ c=1 TPOT | 18.9ms | 6.8ms | 2.8배 |
| BF16 c=300 tok/s | 905 | 2,958 | 3.3배 |
| GPTQ c=300 tok/s | 1,367 | 2,965 | 2.2배 |
| **c=300 승자** | **GPTQ > BF16 (1.5배)** | **GPTQ ≈ BF16 (동률)** | — |

decode가 memory-bound라는 진단 파트의 명제 그대로 — TPOT 개선폭(3.3배)이 대역폭 비율(3.4배)에 수렴한다.
그리고 고동시성 승자가 하드웨어에 따라 바뀐다: L4에서는 c300에서도 대역폭이 병목이라 양자화가 이기고,
4090에서는 연산 병목으로 넘어가 이점이 사라진다.

- key point: EC2 자동화 없이 동일 방법론을 로컬 재현했고, 원저자 L4 실측과의 대조에서 **TPOT 개선폭 ≈ 대역폭 비율(3.3~3.4배)**이 확인됐다. "절대 수치가 아니라 상대 경향을 보라"는 이 문서의 경고를 넘어, 상대 경향(고동시성 승자)조차 하드웨어에 따라 뒤집힌다는 것이 실측됐다.


### Lab 4 — 도전과제 10종 ✅ *(1·9 완료, 나머지는 판정표 참고)*

노션 제시 10개. 전부 **vLLM 또는 SGLang** 기준이다.

| # | 과제 | 4090 판정 | 메모 |
|---|---|---|---|
| 1 | 연속 배칭 ON vs **`max_num_seqs=1`** 로 무력화 — 처리량·지연시간 측정 | ✅ | **가장 먼저 할 것.** 배칭의 효과를 숫자로 보는 가장 직관적인 실험 |
| 2 | `max batch size` / `max model length` / `max number of tokens` 기본값 확인 후 변경 비교 | ✅ | KodeKloud Task 7의 로컬 확장판 |
| 3 | **Chunked Prefill** ON vs OFF + `--max-num-batched-tokens` 튜닝 | ✅ | ⚠️ vLLM V1에서는 **기본 활성화**다. OFF 비교를 하려면 명시적으로 꺼야 한다 |
| 4 | **MHA vs GQA vs MQA vs MLA** 어텐션별 추론 속도·품질 비교 | ⚠️ | **MLA(DeepSeek)가 문제.** DeepSeek-V2-Lite BF16이 약 31GB로 24GB 초과 → **양자화 버전으로만** 가능 |
| 5 | **FlashInfer 커널** (FlashAttention 2/3/4) 설정 및 성능 비교 | ⚠️ | **FA3는 Hopper(sm90) 전용.** 4090은 FA2 + FlashInfer의 Ada 커널까지 |
| 6 | 모델 1개 선정 후 양자화 정밀도 변경 로딩·성능 비교 | ✅ | 실습 ②와 동일 |
| 7 | **Hands-on quantization** 따라하고 벤치마크 비교 | ✅ | 실습 ②. 노션에도 "Colab 혹은 **Local PC(GPU 장착)**" 로 명시 |
| 8 | **Distillation** 직접 실행 후 큰 모델과 성능 비교 | ⚠️ | **추론이 아니라 학습이다.** 7B teacher + 0.5B student면 24GB로 소규모는 가능하나 시간이 많이 든다 |
| 9 | **프리픽스 캐싱 / RadixAttention** ON vs OFF 성능 비교 | ✅ | ⚠️ vLLM V1은 **기본 활성화**. `--no-enable-prefix-caching`으로 꺼서 비교 |
| 10 | **Cache-aware router**로 Scaling Prefix Cache 적용 | ✅ | GPU 1장에 **작은 모델 인스턴스 2개**를 `--gpu-memory-utilization 0.4`씩 띄우면 라우팅 실험이 가능하다 |


#### 도전과제 1이 가장 가성비가 좋다

```bash
# 배칭 정상
vllm serve Qwen/Qwen2.5-7B-Instruct-GPTQ-Int4 --max-model-len 4096

# 배칭 무력화
vllm serve Qwen/Qwen2.5-7B-Instruct-GPTQ-Int4 --max-model-len 4096 --max-num-seqs 1
```
같은 부하를 걸고 처리량을 비교하면 **배칭이 왜 첫 번째 최적화 기법인지**가 한 줄로 증명된다.

#### 도전과제 9의 함정 — 프리픽스 캐싱은 쉽게 깨진다

**공백 하나만 달라도 캐시 미스**다. prefix가 **토큰 단위로 정확히 일치**해야 한다.
- 시스템 프롬프트를 고정하고 뒤만 바꾸는 요청 → 히트율 높음
- RAG처럼 **검색 청크 순서가 매번 바뀌는** 요청 → 히트율 급락

ON/OFF 비교 시 **prefix가 실제로 공유되는 프롬프트 세트**를 써야 유의미한 차이가 나온다. 랜덤 프롬프트로 재면 차이가 없다.

---


#### 실행 기록 — 도전과제 1 (연속 배칭 ON vs 무력화) · 9 (prefix caching ON vs OFF)

`challenge_1_9.sh`: GPTQ-Int4 서버를 구성만 바꿔 4회 기동.
- 도전과제 1: sharegpt, n=100, c=100 — 기본값 vs `--max-num-seqs 1`
- 도전과제 9: **공유 prefix 세트** — `--dataset-name random --random-prefix-len 2048` (모든 요청이 같은
  2048토큰 prefix 공유, 요청별 random 256 + 출력 128, n=100, c=20) — "랜덤 프롬프트로 재면 차이가
  없다"는 이 문서의 함정 경고를 피하는 구성. 기본값 vs `--no-enable-prefix-caching`

실행 로그 (발췌):

```shell
[22:39:10] === 도전과제 1a — 연속 배칭 정상 (기본값) ===
Request throughput (req/s):              8.12      
Output token throughput (tok/s):         1763.37   
Median TTFT (ms):                        5131.30   
Median TPOT (ms):                        17.46     
[22:40:32] === 도전과제 1b — 배칭 무력화 (--max-num-seqs 1) ===
Request throughput (req/s):              0.71      
Output token throughput (tok/s):         156.99    
Median TTFT (ms):                        65155.17  
Median TPOT (ms):                        6.25      
[22:43:45] === 도전과제 9a — prefix caching ON (기본) ===
Request throughput (req/s):              9.55      
Output token throughput (tok/s):         1222.09   
Mean TTFT (ms):                          512.58    
Median TTFT (ms):                        468.41    
P99 TTFT (ms):                           1111.57   
Median TPOT (ms):                        12.06     
(APIServer pid=44591) INFO 08-22 22:44:30 [loggers.py:310] Engine 000: Avg prompt throughput: 281.7 tokens/s, Avg generation throughput: 0.6 tokens/s, Running: 16 reqs, Waiting: 4 reqs, GPU KV cache usage: 2.1%, Prefix cache hit rate: 83.3%
vllm:prefix_cache_queries_total{engine="0",model_name="Qwen/Qwen2.5-7B-Instruct-GPTQ-Int4"} 230400.0
vllm:prefix_cache_queries_created{engine="0",model_name="Qwen/Qwen2.5-7B-Instruct-GPTQ-Int4"} 1.7874062558641894e+09
vllm:prefix_cache_hits_total{engine="0",model_name="Qwen/Qwen2.5-7B-Instruct-GPTQ-Int4"} 202736.0
vllm:prefix_cache_hits_created{engine="0",model_name="Qwen/Qwen2.5-7B-Instruct-GPTQ-Int4"} 1.7874062558641977e+09
[22:44:40] === 도전과제 9b — prefix caching OFF (--no-enable-prefix-caching) ===
Request throughput (req/s):              1.86      
Output token throughput (tok/s):         237.80    
Mean TTFT (ms):                          2250.35   
Median TTFT (ms):                        1558.85   
P99 TTFT (ms):                           9004.47   
Median TPOT (ms):                        71.49     
[22:46:18] === 도전과제 1·9 완료 ===
```

#### 도전과제 1 판정

| | 배칭 ON (기본) | 배칭 무력화 (max_num_seqs=1) | 비 |
|---|---|---|---|
| Request throughput | 8.12 req/s | 0.71 req/s | **11.4배** |
| Output tok/s | 1,763 | 157 | 11.2배 |
| Median TTFT | 5,131ms | **65,155ms** | 1/12.7 |
| Median TPOT | 17.46ms | **6.25ms** | 배칭 OFF가 2.8배 빠름 |

**TPOT는 무력화 쪽이 오히려 빠르다** — 혼자 decode하면 스텝당 연산이 적어서다. 대신 100개 요청이
1개씩 줄을 서니 대기(TTFT 65초)와 처리량이 무너진다. 배칭은 **개별 요청의 decode 속도를 조금 내주고
전체 처리량을 사는 거래**라는 것이 숫자로 보인다.

#### 도전과제 9 판정

| | prefix caching ON (기본) | OFF | 비 |
|---|---|---|---|
| Request throughput | 9.55 req/s | 1.86 req/s | **5.1배** |
| Median TTFT | 468ms | 1,559ms | **3.3배** |
| P99 TTFT | 1,112ms | 9,004ms | 8.1배 |
| Median TPOT | 12.06ms | 71.49ms | 5.9배 |

서버 `/metrics` 실측 히트율:

```shell
vllm:prefix_cache_queries_total 230,400
vllm:prefix_cache_hits_total    202,736   → hit rate 88.0% (로그 표시 83.3%는 진행 중 시점값)
```

공유 prefix 2048 / 전체 입력 2304 토큰 = 88.9%가 공유 구간이니 히트율 88.0%는 **공유된 만큼 정확히
아낀 것**이다. prefill의 8/9가 사라지면서 TTFT가 3.3배 줄었다.

- key point: 배칭 무력화로 처리량 11.4배 차이 — 교재가 배칭을 첫 번째 기법으로 두는 이유가 한 줄로 증명된다. prefix caching은 공유 구간 비율(88.9%)만큼 히트(88.0%)가 나왔고 TTFT 3.3배 개선 — 시스템 프롬프트가 긴 챗봇 워크로드에서 왜 필수인지의 정량 근거.


## ❌ 4090으로 못 하는 것

핵심 개념 일부는 **하드웨어가 없어서 실습이 불가능**하다. 개념으로만 정리하고 넘어가는 게 맞다.

- **NVLink / NVSwitch 대역폭 실측** — 4090은 NVLink가 없다
- **텐서 병렬(TP≥2)의 노드 내/외 성능 차이** — GPU 1장
- **P/D Disaggregation** — 최소 2장
- **FP4 / NVFP4** — Blackwell 전용
- **FlashAttention 3** — Hopper 전용

---

## 정리

1. **저동시성 양자화 이득은 교재 그대로 재현됐다** — c=1 TPOT: GPTQ-Int4 2.5배(교재 ~300%), FP8 1.5배(교재 ~150%). 메모리 대역폭 병목 구간에서 가중치 축소가 곧 속도다.
2. **고동시성 승자는 하드웨어의 함수다** — c=300에서 L4는 GPTQ가 BF16을 1.5배 이기지만, 4090은 동률(2,958 vs 2,965 tok/s). 대역폭 3.4배 차이가 병목을 연산으로 옮겨 양자화 이점을 지운다. "상대 경향을 보라"는 지침조차 하드웨어를 타는 것이 실측됐다.
3. **TPOT 개선폭(3.3배) ≈ 대역폭 비율(3.4배)** — decode가 memory-bound라는 명제의 정량 검증. 같은 24GB·같은 CC 8.9에서 대역폭만 다른 두 GPU가 이를 보여준다.
4. **FP8-dynamic이 WSL2+4090에서 c300에 재현성 있게 붕괴한다** (539/103/146 tok/s, 3회) — c100까지는 3모델 중 최저 TTFT로 정상. 네이티브 Linux L4에서는 정상이므로 환경 특이 현상으로 판정, 원인은 미확정. 벤치마크는 환경의 함수다.
5. **배칭 무력화(max_num_seqs=1)로 처리량 11.4배 차이** — 단 TPOT는 무력화 쪽이 2.8배 빠르다(혼자 decode). 배칭 = 개별 decode 속도를 내주고 전체 처리량을 사는 거래.
6. **prefix caching: 공유 구간 비율(88.9%)만큼 히트(88.0%), TTFT 3.3배·처리량 5.1배** — 공유 prefix가 실재하는 워크로드에서만 성립한다. KV cache 실측은 INT4 288k > FP8 230k > BF16 102k tokens — 가중치가 작을수록 남는 VRAM이 전부 KV cache가 된다.

---

## Self-Check

- [ ] KodeKloud 랩 Task 1~8 완주 (무료, 브라우저) (Lab 1 — 계정 필요, 직접 진행)
- [x] 4090에서 `vllm serve` 로 Qwen2.5-7B **BF16 기동 성공** (0.92 / max-model-len 8192) (Lab 2)
- [x] BF16 / FP8 / GPTQ-Int4 3종 **VRAM 사용량** `nvidia-smi`로 기록 (Lab 2 — 22,951 / 23,792 / 23,859 MiB)
- [x] **저동시성(1:10)** 과 **고동시성(300:300)** 양쪽에서 TTFT/TPOT/throughput 측정 (Lab 2, 3 — 4점 전부)
- [x] 저·고 동시성에서 **승자가 뒤바뀌는지** 확인 (Lab 2 — 4090에서는 W4A16 → 동률로 수렴, FP8은 c300 이상 현상)
- [x] `--max-num-seqs 1` 로 배칭 무력화 후 처리량 비교 (Lab 4 — 11.4배 차이)
- [x] prefix caching ON/OFF를 **prefix가 공유되는 프롬프트 세트**로 비교 (Lab 4 — 히트율 88%, TTFT 3.3배)
- [x] (선택) 4090 결과가 교재의 L4/T4 결과와 **어떻게 다른지** 정리 (Lab 3 — TPOT 개선폭 ≈ 대역폭 비율, 고동시성 승자 역전)

