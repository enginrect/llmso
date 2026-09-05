# 실전 LLM 최적화와 서빙의 새로운 방향 — 실습 기록

모든 측정은 RTX 4090 24GB / WSL2 Ubuntu 24.04에서 했다 ([환경 구축 기록](environment-setup.md), [개념 정리](../optimization-in-practice-and-emerging-directions.md)).

실습 섹션은 5개다. GPU 1장이라 안 되는 항목이 있다.

| 실습 | 필요 자원 | 4090 판정 |
|---|---|---|
| ① 최적화 워크플로 (트래픽 → 베이스라인 → 양자화) | GPU 24GB | ✅ **핵심** |
| ② 3계층 프로파일링 (nsys / PyTorch / ncu) | GPU + 관리자 권한 | ✅ **핵심** |
| ③ 서빙 모니터링 스택 | Docker | ✅ |
| ④ 게이트웨이 라우팅 (LiteLLM / Semantic Router) | GPU + 소형 모델 2개 | ✅ |
| ⑤ Multi-LoRA와 멀티모달 서빙 | GPU 24GB | ✅ |
| ⑥ 멀티 GPU 분산 (PP/TP/복제 비교) | GPU 2장 이상 | ❌ 로컬 불가 → Runpod 대체 (유료) |

---

## 실습 기록 (Labs)

### Lab 1 — 최적화 워크플로: 트래픽 설계에서 양자화까지 ✅ *(핵심)*

**원본**: 교재 실습 (AWS g6e.2xlarge, L40S 46GB, Qwen3-14B). 4090은 24GB라
Qwen3-14B bf16(약 27.5GB)이 들어가지 않는다. **Qwen3-8B**로 축소 재현한다.

| | 교재 (L40S 46GB) | 이번 (4090 24GB) |
|---|---|---|
| 베이스 | Qwen3-14B (27.5GiB) | **Qwen/Qwen3-8B** (약 16.4GiB 예상) |
| 양자화 | Qwen3-14B-AWQ (9.36GiB) | **Qwen/Qwen3-8B-AWQ** (약 5.7GiB 예상) |

**예측 (측정 후 검증할 것)**: Qwen3-8B의 KV는 토큰당
`2 × 36층 × 8 KV헤드 × 128 × 2바이트 = 144 KiB`다. `--gpu-memory-utilization 0.90`
기준으로 베이스는 약 3~4GiB, AWQ는 약 14~15GiB가 KV에 남아 **KV 토큰이 3~4배**
벌어질 것으로 보인다. 교재(46GB, 2.65배)보다 배수가 커야 정상이다 — 베이스 모델이
24GB에서 차지하는 비중이 더 크기 때문이다. **어긋나면 그 자체가 기록할 결과다.**

```bash
# Step 1 — 하드웨어 점검 (벤치 전 유휴 확인)
nvidia-smi --query-gpu=name,compute_cap,memory.free,memory.used,memory.total,pstate \
  --format=csv
# ⚠️ P-state 확인: 유휴 P8 → 벤치 중 P0~P2로 떨어지는지 같이 기록

# Step 2 — 벤치마크 트래픽 준비
#   ShareGPT 원본(약 600MB)
wget https://huggingface.co/datasets/anon8231489123/ShareGPT_Vicuna_unfiltered/resolve/main/ShareGPT_V3_unfiltered_cleaned_split.json

#   교재의 inspect_dataset.py를 vllm 리포 루트에 복사해 실행 (추론 아님, 분포 확인용)
python3 inspect_dataset.py --dataset-name sharegpt \
  --dataset-path ShareGPT_V3_unfiltered_cleaned_split.json \
  --model Qwen/Qwen3-8B --num-prompts 100 --save-samples
python3 inspect_dataset.py --dataset-name prefix_repetition \
  --model Qwen/Qwen3-8B --num-prompts 50 \
  --prefix-repetition-prefix-len 256 --prefix-repetition-suffix-len 256 \
  --prefix-repetition-num-prefixes 5 --prefix-repetition-output-len 128 --save-samples
# ⚠️ --dataset-name custom/sharegpt는 pandas가 필요할 수 있다
#    (최적화 실습에서 겪은 함정: ImportError 문구가 pandas 부재를 가리지 않는다)

# Step 4~5 — 베이스라인
vllm serve Qwen/Qwen3-8B --max-model-len 4096 --gpu-memory-utilization 0.90 --port 8000
# 로그에서 반드시 뽑을 3줄: Model loading took / Available KV cache memory / GPU KV cache size
#   + Maximum concurrency

vllm bench serve --backend vllm --base-url http://127.0.0.1:8000 \
  --model Qwen/Qwen3-8B --dataset-name sharegpt \
  --dataset-path ShareGPT_V3_unfiltered_cleaned_split.json \
  --num-prompts 2000 --request-rate 10 --burstiness 1.0 --max-concurrency 10 \
  --save-result --append-result --result-filename w5_results.txt

vllm bench serve --backend vllm --base-url http://127.0.0.1:8000 \
  --model Qwen/Qwen3-8B --dataset-name prefix_repetition \
  --num-prompts 1000 --request-rate 5 --max-concurrency 10 \
  --prefix-repetition-prefix-len 256 --prefix-repetition-suffix-len 256 \
  --prefix-repetition-num-prefixes 10 --prefix-repetition-output-len 128 \
  --save-result --append-result --result-filename w5_results.txt

# Step 6 — 양자화 (동일 벤치 2회 반복)
vllm serve Qwen/Qwen3-8B-AWQ --max-model-len 4096 --gpu-memory-utilization 0.90 --port 8000
```

**기록할 표 (2모델 × 2트래픽 = 4셀)**: 총 TPS / 출력 TPS / 평균 TTFT / 평균 ITL.
여기에 서버 로그의 모델 가중치·KV 캐시·KV 토큰·최대 동시성을 함께 남긴다.

**⚠️ 반드시 분해할 것**: 총 처리량이 올랐다면 `총 − 출력 = 입력 처리량`을 계산해
**출력 처리량이 그대로인지** 확인한다. 교재 수치에서는 Prefix Repetition의 2.37배가
전부 입력 쪽이었고 출력은 오히려 1.9% 낮았다. 4090에서도 같은 구조인지가 이번
실습의 핵심 질문이다.

**⚠️ 사전 점검**: 베이스 모델의 KV 캐시가 지나치게 작으면(동시성 상한이 1x 근처)
그 상태의 측정값은 의미가 없다. `--max-model-len`을 줄이거나 동시성을 낮춰
비교 가능한 조건부터 만든다.

**⚠️ 검증할 교재 수치 2건**: ① "모델이 GPU 메모리의 65% 이상" → 계산상 59.8%
② 수평 확장 "거의 3배" → 계산상 2.50배. 우리 로그에서 같은 비율을 직접 계산해본다.

```shell
=== Step 1: nvidia-smi (idle) ===
$ nvidia-smi --query-gpu=name,compute_cap,memory.free,memory.used,memory.total,pstate --format=csv
NVIDIA GeForce RTX 4090, 8.9, 22801 MiB, 1338 MiB, 24564 MiB, P8

=== Step 2: inspect_dataset.py — sharegpt (100개) ===
$ python3 inspect_dataset.py --dataset-name sharegpt --dataset-path ShareGPT_V3_... --model Qwen/Qwen3-8B --num-prompts 100
=== Prompt Length Distribution ===        === Output Length Distribution ===
Min 5 / Max 817                           Min 4 / Max 771
Mean 232.60 / Median 141.50 / Std 241.42  Mean 220.61 / Median 164.50 / Std 210.23
...
=== Prompt Length Histogram ===
   5-  95 tokens: *********************************************
  95- 185 tokens: ********
 185- 275 tokens: *********
 275- 365 tokens: **************
 365- 456 tokens: *****
...
 726- 817 tokens: *******
# → 교재의 100샘플 통계와 완전히 일치(같은 시드·같은 원본). 짧은 프롬프트에 45개가 몰린 우편향.

=== Step 2: inspect_dataset.py — prefix_repetition (50개) ===
$ python3 inspect_dataset.py --dataset-name prefix_repetition --model Qwen/Qwen3-8B --num-prompts 50 \
    --prefix-repetition-prefix-len 256 --prefix-repetition-suffix-len 256 \
    --prefix-repetition-num-prefixes 5 --prefix-repetition-output-len 128
# 프롬프트/출력 길이가 전부 고정: 512 / 128, 표준편차 0 (합성 데이터라 분산이 없다)
# 프롬프트 본문은 토크나이저 어휘에서 뽑은 무작위 토큰열이라 의미가 없다 (출력 축약)
```

**베이스라인 — Qwen/Qwen3-8B (bf16)**

```shell
=== Step 4/5/6: serve Qwen/Qwen3-8B ===
$ vllm serve Qwen/Qwen3-8B --max-model-len 4096 --gpu-memory-utilization 0.90 --port 8000
--- server log (memory / KV) ---
Model loading took 15.27 GiB and 10.718686 seconds
Available KV cache memory: 5.87 GiB
GPU KV cache size: 42,768 tokens
Maximum concurrency for 4,096 tokens per request: 10.44x
--- nvidia-smi after load ---
memory.used [MiB], memory.total [MiB], pstate
24044 MiB, 24564 MiB, P5

--- bench sharegpt ---
$ vllm bench serve --backend vllm --base-url http://127.0.0.1:8000 --model Qwen/Qwen3-8B --dataset-name sharegpt --dataset-path ShareGPT_V3_unfiltered …
...
Successful requests:                     2000      
Benchmark duration (s):                  1667.10   
Request throughput (req/s):              1.20      
Output token throughput (tok/s):         247.40    
Total token throughput (tok/s):          515.30    
Mean TTFT (ms):                          308.53    
Median TTFT (ms):                        111.71    
P99 TTFT (ms):                           4128.16   
Mean TPOT (ms):                          39.95     
Median TPOT (ms):                        25.20     
P99 TPOT (ms):                           152.18    
Mean ITL (ms):                           38.87     
P99 ITL (ms):                            128.89    
...

--- bench prefix_repetition ---
$ vllm bench serve --backend vllm --base-url http://127.0.0.1:8000 --model Qwen/Qwen3-8B --dataset-name prefix_repetition --num-prompts 1000 --request …
...
Successful requests:                     1000      
Benchmark duration (s):                  322.81    
Request throughput (req/s):              3.10      
Output token throughput (tok/s):         348.84    
Total token throughput (tok/s):          1935.02   
Mean TTFT (ms):                          122.42    
Median TTFT (ms):                        106.30    
P99 TTFT (ms):                           201.54    
Mean TPOT (ms):                          27.41     
Median TPOT (ms):                        26.68     
P99 TPOT (ms):                           59.08     
Mean ITL (ms):                           27.46     
P99 ITL (ms):                            70.15     
...
--- P-state during bench (distinct states seen) ---
$ tail -n +2 pstate.csv | cut -d, -f2 | sort | uniq -c
    369  P2
      1  P3
      1  P5
      3  P8
--- prefix cache metrics ---
$ curl -s http://127.0.0.1:8000/metrics | grep -E "prefix_cache_(queries|hits)_total"
vllm:prefix_cache_queries_total{engine="0",model_name="Qwen/Qwen3-8B"} 958647.0
vllm:prefix_cache_hits_total{engine="0",model_name="Qwen/Qwen3-8B"} 253968.0
server stopped: 1074 MiB, P0

=== Step 4/5/6: serve Qwen/Qwen3-8B-AWQ ===
$ vllm
```

**양자화 — Qwen/Qwen3-8B-AWQ**

```shell
=== Step 4/5/6: serve Qwen/Qwen3-8B-AWQ ===
$ vllm serve Qwen/Qwen3-8B-AWQ --max-model-len 4096 --gpu-memory-utilization 0.90 --port 8000
--- server log (memory / KV) ---
Model loading took 5.71 GiB and 6.483573 seconds
Available KV cache memory: 14.43 GiB
GPU KV cache size: 105,056 tokens
Maximum concurrency for 4,096 tokens per request: 25.65x
--- nvidia-smi after load ---
memory.used [MiB], memory.total [MiB], pstate
23371 MiB, 24564 MiB, P8

--- bench sharegpt ---
$ vllm bench serve --backend vllm --base-url http://127.0.0.1:8000 --model Qwen/Qwen3-8B-AWQ --dataset-name sharegpt --dataset-path ShareGPT_V3_unfilt …
...
Successful requests:                     2000      
Benchmark duration (s):                  422.81    
Request throughput (req/s):              4.73      
Output token throughput (tok/s):         975.95    
Total token throughput (tok/s):          2032.25   
Mean TTFT (ms):                          57.56     
Median TTFT (ms):                        40.06     
P99 TTFT (ms):                           141.17    
Mean TPOT (ms):                          9.98      
Median TPOT (ms):                        9.59      
P99 TPOT (ms):                           16.54     
Mean ITL (ms):                           9.92      
P99 ITL (ms):                            48.67     
...

--- bench prefix_repetition ---
$ vllm bench serve --backend vllm --base-url http://127.0.0.1:8000 --model Qwen/Qwen3-8B-AWQ --dataset-name prefix_repetition --num-prompts 1000 --req …
...
Successful requests:                     1000      
Benchmark duration (s):                  215.77    
Request throughput (req/s):              4.63      
Output token throughput (tok/s):         515.71    
Total token throughput (tok/s):          2888.69   
Mean TTFT (ms):                          55.52     
Median TTFT (ms):                        51.12     
P99 TTFT (ms):                           97.89     
Mean TPOT (ms):                          9.81      
Median TPOT (ms):                        9.67      
P99 TPOT (ms):                           11.36     
Mean ITL (ms):                           9.70      
P99 ITL (ms):                            33.89     
...
--- P-state during bench (distinct states seen) ---
$ tail -n +2 pstate.csv | cut -d, -f2 | sort | uniq -c
    119  P2
      1  P5
      3  P8
--- prefix cache metrics ---
$ curl -s http://127.0.0.1:8000/metrics | grep -E "prefix_cache_(queries|hits)_total"
vllm:prefix_cache_queries_total{engine="0",model_name="Qwen/Qwen3-8B-AWQ"} 958647.0
vllm:prefix_cache_hits_total{engine="0",model_name="Qwen/Qwen3-8B-AWQ"} 255136.0
server stopped: 1185 MiB, P0

=== DONE Lab 1 ===
```

**결과 표 (2모델 × 2트래픽, `--gpu-memory-utilization 0.90`)**

| 셀 | 총 TPS | 출력 TPS | 입력 TPS (총−출력) | 평균 TTFT | 평균 ITL | 평균 TPOT |
|---|---|---|---|---|---|---|
| base × ShareGPT | 515.3 | 247.4 | 267.9 | 308.5ms | 38.9ms | 40.0ms |
| AWQ × ShareGPT | 2032.2 (3.94x) | 976.0 (3.94x) | 1056.3 (3.94x) | 57.6ms | 9.9ms | 10.0ms |
| base × Prefix Repetition | 1935.0 | 348.8 | 1586.2 | 122.4ms | 27.5ms | 27.4ms |
| AWQ × Prefix Repetition | 2888.7 (1.49x) | 515.7 (1.48x) | 2373.0 (1.50x) | 55.5ms | 9.7ms | 9.8ms |

**서버 로그의 메모리·동시성**

| | 가중치 | KV 캐시 | KV 토큰 | 최대 동시성 (4,096 tok) | nvidia-smi 로드 후 |
|---|---|---|---|---|---|
| Qwen3-8B | 15.27 GiB | 5.87 GiB | 42,768 | 10.44x | 24,044 / 24,564 MiB |
| Qwen3-8B-AWQ | 5.71 GiB | 14.43 GiB | 105,056 | 25.65x | 23,371 / 24,564 MiB |
| 배수 | 0.37x | 2.46x | 2.46x | 2.46x | |

- 계획서 예측 "KV 토큰 3~4배"는 어긋났다 — 실측 2.46배(교재 L40S 2.65배). 가중치 차이 9.56 GiB 중 KV로 돌아온 것은 8.56 GiB이고, 나머지 약 1 GiB는 AWQ 쪽 활성화/CUDA graph 프로파일링 예약분이 더 큰 것으로 설명된다 (0.90 예산 22.1 GiB − 가중치 − KV = base 0.96 GiB vs AWQ 1.96 GiB)
- 처리량 분해: ShareGPT는 입력·출력이 함께 3.94x/3.94x 올랐다. Prefix Repetition은 request-rate 5의 도착 부하에 AWQ 서버가 묶여(running 평균 5.4, 요청 처리율 4.63 req/s ≈ 상한 5) 1.48x에서 멈췄고, 입력·출력이 같은 비율(1.50x/1.48x)로 올랐다. 교재의 "2.37배가 전부 입력 쪽" 구조는 이 부하에서는 재현되지 않았다 — 교재 수치는 서버가 포화된 상태의 분해고, 여기서는 도착률이 상한이다
- P-state: 유휴 P8 → 벤치 중 P2 고정(prefix 구간 374 샘플 중 P2 369). 계획서의 "P0~P2 전환" 관찰 완료
- prefix cache 히트: base 253,968 / 958,647 (26.5%), AWQ 255,136 / 958,647 (26.6%) — 두 모델이 같은 트래픽을 받았다는 확인
- 교재 수치 재계산 ①: "모델이 GPU 메모리의 65% 이상" → 여기서는 15.27 / 23.99 GiB = **63.6%** (교재 계산값 59.8%). ②: "수평 확장 거의 3배" → 교재 계산값 2.50배. 여기의 ShareGPT 출력 배수는 0.90에서 3.94x인데, 아래 보강 실험에서 이 수치는 베이스 측정 자체가 손상된 결과임이 드러났다
- ShareGPT 베이스의 평균 TPOT 40.0ms vs 중앙값 25.2ms, P99 152ms, P99 TTFT 4.1초 — 꼬리가 비정상적으로 길다. 모니터링(Lab 3)에서 같은 구간의 PCIe TX가 평균 3.2 GB/s로 잡혀 원인을 아래에서 추적했다

- key point: AWQ는 가중치 −9.56 GiB → KV +8.56 GiB(2.46배)로 돌아왔고, ShareGPT 출력 처리량은 3.94x·TPOT는 40.0→10.0ms. Prefix Repetition은 도착률(rate 5)이 상한이라 1.48x에 그쳤고 입력/출력이 같은 비율로 움직였다.

**보강 — GPU 메모리 압박 검증: 0.90 vs 0.85**

베이스 벤치 구간에서 PCIe TX 평균 3.2 GB/s가 관측되어(AWQ 구간 39 MB/s), `--gpu-memory-utilization`을 0.85로 낮춘 같은 부하와 Windows 측 WDDM 메모리 카운터(`Get-Counter '\GPU Adapter Memory(*)\Shared Usage'`)를 함께 재측정했다.

```shell
=== 유휴 상태 Windows GPU 메모리 카운터 ===
$ powershell.exe Get-Counter "\GPU Adapter Memory(*)\Shared Usage","\GPU Adapter Memory(*)\Dedicated Usage"  (RTX 4090 어댑터)
Path                                                                                    CookedValue
----                                                                                    -----------
gpu adapter memory(luid_0x00000000_0x0000c31e_phys_0)\shared usage      223404032
gpu adapter memory(luid_0x00000000_0x0000c31e_phys_0)\dedicated usage  1250324480

=== base090: vllm serve Qwen/Qwen3-8B --gpu-memory-utilization 0.90 ===
$ vllm serve Qwen/Qwen3-8B --max-model-len 4096 --gpu-memory-utilization 0.90 --port 8000
Model loading took 15.27 GiB and 25.848629 seconds
Available KV cache memory: 5.87 GiB
GPU KV cache size: 42,768 tokens
Maximum concurrency for 4,096 tokens per request: 10.44x
$ nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader
23991 MiB, 24564 MiB
--- 서버 로드 직후 (부하 전) ---
  Windows WDDM  Shared Usage(=시스템 RAM으로 스필된 GPU 메모리): min 497 / mean 497 / max 497 MiB   Dedicated max 23999 MiB  (n=2)
--- bench prefix_repetition 1000 (rate 5, c10) ---
$ vllm bench serve --backend vllm --base-url http://127.0.0.1:8000 --model Qwen/Qwen3-8B --dataset-name prefix_repetition --num-prompts 1000 --request …
Benchmark duration (s):                  326.41    
Output token throughput (tok/s):         345.85    
Mean TTFT (ms):                          123.94    
Mean TPOT (ms):                          27.78     
Median TPOT (ms):                        27.68     
P99 TPOT (ms):                           31.50     
window 1788620430 1788620742
  Windows WDDM  Shared Usage(=시스템 RAM으로 스필된 GPU 메모리): min 483 / mean 493 / max 499 MiB   Dedicated max 24044 MiB  (n=69)
  GPU util % (NVML)              2.0      95.3     100.0
  PCIe TX MB/s (NVML)            2.0    1193.9    5249.4
  PCIe RX MB/s (NVML)            0.1    1412.4    4969.7
  vLLM gen tok/s                 0.0     350.5     394.6
--- bench sharegpt 500 (rate 10, c10) ---
$ vllm bench serve --backend vllm --base-url http://127.0.0.1:8000 --model Qwen/Qwen3-8B --dataset-name sharegpt --dataset-path ShareGPT_V3_unfiltered …
Benchmark duration (s):                  296.85    
Output token throughput (tok/s):         370.18    
Mean TTFT (ms):                          105.69    
Mean TPOT (ms):                          25.66     
Median TPOT (ms):                        25.47     
P99 TPOT (ms):                           29.92     
window 1788620742 1788621025
  Windows WDDM  Shared Usage(=시스템 RAM으로 스필된 GPU 메모리): min 484 / mean 506 / max 571 MiB   Dedicated max 24044 MiB  (n=62)
  GPU util % (NVML)              2.0      96.1     100.0
  PCIe TX MB/s (NVML)            4.5    1116.1    5404.9
  PCIe RX MB/s (NVML)            1.9    1001.5    4919.7
  vLLM gen tok/s               191.7     394.5     436.3

=== base085: vllm serve Qwen/Qwen3-8B --gpu-memory-utilization 0.85 ===
$ vllm serve Qwen/Qwen3-8B --max-model-len 4096 --gpu-memory-utilization 0.85 --port 8000
Model loading took 15.27 GiB and 5.165522 seconds
Available KV cache memory: 4.67 GiB
GPU KV cache size: 34,032 tokens
Maximum concurrency for 4,096 tokens per request: 8.31x
$ nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader
22999 MiB, 24564 MiB
--- 서버 로드 직후 (부하 전) ---
  Windows WDDM  Shared Usage(=시스템 RAM으로 스필된 GPU 메모리): min 311 / mean 312 / max 313 MiB   Dedicated max 23007 MiB  (n=2)
--- bench prefix_repetition 1000 (rate 5, c10) ---
$ vllm bench serve --backend vllm --base-url http://127.0.0.1:8000 --model Qwen/Qwen3-8B --dataset-name prefix_repetition --num-prompts 1000 --request …
Benchmark duration (s):                  265.17    
Output token throughput (tok/s):         421.21    
Mean TTFT (ms):                          92.01     
Mean TPOT (ms):                          22.73     
Median TPOT (ms):                        22.87     
P99 TPOT (ms):                           23.66     
window 1788621077 1788621332
  Windows WDDM  Shared Usage(=시스템 RAM으로 스필된 GPU 메모리): min 299 / mean 308 / max 313 MiB   Dedicated max 23052 MiB  (n=57)
  GPU util % (NVML)              2.0      93.5     100.0
  PCIe TX MB/s (NVML)            3.2      32.7      64.2
  PCIe RX MB/s (NVML)            0.1      17.7      44.7
  vLLM gen tok/s                 0.0     406.5     476.6
--- bench sharegpt 500 (rate 10, c10) ---
$ vllm bench serve --backend vllm --base-url http://127.0.0.1:8000 --model Qwen/Qwen3-8B --dataset-name sharegpt --dataset-path ShareGPT_V3_unfiltered …
Benchmark duration (s):                  246.08    
Output token throughput (tok/s):         446.55    
Mean TTFT (ms):                          81.19     
Mean TPOT (ms):                          21.32     
Median TPOT (ms):                        21.21     
P99 TPOT (ms):                           24.18     
window 1788621332 1788621570
  Windows WDDM  Shared Usage(=시스템 RAM으로 스필된 GPU 메모리): min 267 / mean 307 / max 313 MiB   Dedicated max 23052 MiB  (n=51)
  GPU util % (NVML)              2.0      95.2     100.0
  PCIe TX MB/s (NVML)            0.1      29.5      82.3
  PCIe RX MB/s (NVML)            1.5      23.9     111.1
  vLLM gen tok/s               248.6     475.9     523.3

=== awq085: vllm serve Qwen/Qwen3-8B-AWQ --gpu-memory-utilization 0.85 ===
$ vllm serve Qwen/Qwen3-8B-AWQ --max-model-len 4096 --gpu-memory-utilization 0.85 --port 8000
Model loading took 5.71 GiB and 4.507307 seconds
Available KV cache memory: 14.23 GiB
GPU KV cache size: 103,632 tokens
Maximum concurrency for 4,096 tokens per request: 25.30x
$ nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader
23091 MiB, 24564 MiB
--- 서버 로드 직후 (부하 전) ---
  Windows WDDM  Shared Usage(=시스템 RAM으로 스필된 GPU 메모리): min 312 / mean 313 / max 313 MiB   Dedicated max 23099 MiB  (n=2)
--- bench prefix_repetition 1000 (rate 5, c10) ---
$ vllm bench serve --backend vllm --base-url http://127.0.0.1:8000 --model Qwen/Qwen3-8B-AWQ --dataset-name prefix_repetition --num-prompts 1000 --req …
Benchmark duration (s):                  218.15    
Output token throughput (tok/s):         510.72    
Mean TTFT (ms):                          55.19     
Mean TPOT (ms):                          9.72      
Median TPOT (ms):                        9.69      
P99 TPOT (ms):                           11.36     
window 1788621622 1788621833
  Windows WDDM  Shared Usage(=시스템 RAM으로 스필된 GPU 메모리): min 257 / mean 307 / max 313 MiB   Dedicated max 23144 MiB  (n=47)
  GPU util % (NVML)              1.0      81.7      99.0
  PCIe TX MB/s (NVML)            0.1      30.7      83.8
  PCIe RX MB/s (NVML)            0.1      26.4     117.0
  vLLM gen tok/s                 0.0     492.2     642.1
--- bench sharegpt 500 (rate 10, c10) ---
$ vllm bench serve --backend vllm --base-url http://127.0.0.1:8000 --model Qwen/Qwen3-8B-AWQ --dataset-name sharegpt --dataset-path ShareGPT_V3_unfilt …
Benchmark duration (s):                  109.37    
Output token throughput (tok/s):         1004.77   
Mean TTFT (ms):                          48.94     
Mean TPOT (ms):                          9.39      
Median TPOT (ms):                        9.20      
P99 TPOT (ms):                           12.46     
window 1788621833 1788621946
  Windows WDDM  Shared Usage(=시스템 RAM으로 스필된 GPU 메모리): min 271 / mean 307 / max 312 MiB   Dedicated max 23144 MiB  (n=24)
  GPU util % (NVML)              1.0      86.2      99.0
  PCIe TX MB/s (NVML)            0.2      25.1      82.0
  PCIe RX MB/s (NVML)            0.1      26.1      39.3
  vLLM gen tok/s               349.7     931.4    1204.9

=== DONE Lab 1b ===
```

| 구성 | nvidia-smi 로드 후 | Prefix 출력 TPS | Prefix TPOT (P99) | ShareGPT-500 출력 TPS | ShareGPT-500 TPOT (P99) | PCIe TX 평균 (Prefix) |
|---|---|---|---|---|---|---|
| base 0.90 | 23,991 MiB | 345.9 | 27.8ms (31.5) | 370.2 | 25.7ms (29.9) | 1,194 MB/s |
| base 0.85 | 22,999 MiB | 421.2 (1.22x) | 22.7ms (23.7) | 446.6 (1.21x) | 21.3ms (24.2) | 33 MB/s |
| AWQ 0.85 | 23,091 MiB | 510.7 | 9.7ms (11.4) | 1004.8 | 9.4ms (12.5) | 31 MB/s |

- 0.90에서는 dedicated VRAM이 24,044 / 24,564 MiB(97.9%)까지 차고 PCIe TX/RX가 1.0~1.4 GB/s(버스트 5~10 GB/s, `nvidia-smi dmon` 교차 확인)로 움직인다. 0.85(23,052 MiB, 93.8%)로 내리면 30 MB/s대로 사라진다. Windows Shared Usage는 두 경우 모두 300~570 MiB로 크게 늘지 않아, 스필 용량이 커지는 게 아니라 **포화 상태에서 같은 페이지가 반복 이동**하는 형태로 보인다 (이 해석은 카운터 조합에서 추론한 것이다)
- 같은 부하에서 KV가 20% 작은 0.85 쪽이 오히려 Prefix 출력 +22%, TPOT −18%, ShareGPT-500 출력 +21%. 계획서의 "베이스 KV가 지나치게 작으면 비교 가능한 조건부터 만든다"는 경고가 반대 방향으로 적용됐다 — **너무 크게 잡아도 비교가 깨진다**
- 공정 비교(둘 다 0.85): ShareGPT-500 출력 1004.8 vs 446.6 = **2.25x** (교재 2.7배), TPOT 21.3 → 9.4ms (2.27x). 0.90 본 벤치의 3.94x는 베이스 손상분이 섞인 값이다
- 지난 실습(서빙 병목 기록)에서 "원인 미확정"으로 남겼던 FP8 c300 붕괴도 같은 기전일 가능성이 있다 — 그때도 0.90에 대형 모델이었다 (미검증)

- key point: WSL2 + 24GB 카드에서 `--gpu-memory-utilization 0.90`은 15 GiB 모델에서 VRAM 97.9% 포화 → PCIe 1 GB/s대 페이지 이동 → 출력 처리량 −18~22%를 만들었다. AWQ 대 베이스의 공정한 배수는 2.25x(ShareGPT, 0.85 동일)이다.


### Lab 2 — 3계층 프로파일링: "빨라진 이유" 규명 ✅ *(핵심)*

Lab 1에서 처리량이 올랐다면, 이 랩은 **왜** 올랐는지를 답한다. 교재의 의사결정
흐름을 그대로 따른다.

```
Nsight Systems → GPU가 바쁜가?
 ├─ 아니다 → PyTorch Profiler (CPU 모드)
 └─ 바쁘다 → PyTorch Profiler (CUDA 모드) → 단일 커널 지배? → Nsight Compute
```

```bash
# 설치 (WSL2)
#  nsys/ncu는 CUDA Toolkit에 포함되거나 별도 배포판으로 설치
nsys --version && ncu --version

# ① Nsight Systems — 서버를 감싸 기동한 뒤 부하를 준다
nsys profile --trace=cuda,nvtx,osrt -o w5_base_nsys \
  vllm serve Qwen/Qwen3-8B --max-model-len 4096 --gpu-memory-utilization 0.90
#   → 별 창에서 vllm bench serve 실행 → 서버 종료(트레이스 finalize)
nsys stats w5_base_nsys.nsys-rep --report cuda_gpu_kern_sum,cuda_api_sum

# ② Nsight Compute — bench serve(장시간 서버)가 아니라 오프라인 배치로
#    ⚠️ ncu는 커널마다 여러 번 재실행하므로 서버 방식과 상성이 나쁘다
ncu --set full --launch-count 20 -o w5_base_ncu \
  vllm bench latency --model Qwen/Qwen3-8B --batch-size 8 \
    --input-len 256 --output-len 64 --enforce-eager
```

**⚠️ 함정 3가지 (스터디 공유)**

1. **Nsight Compute는 관리자 권한을 요구한다.** 컨테이너 기반 클라우드 GPU에서는
   동작하지 않을 가능성이 크다. WSL2 로컬이라 가능할지 여부부터 확인한다
2. **GPU 프로파일링 카운터는 배타적이다.** Lab 3의 DCGM exporter가 떠 있으면 ncu가
   실패할 수 있어 잠시 내려야 한다
3. **결과 파일은 로컬 GUI로 본다.** `.nsys-rep` / `.ncu-rep`를 macOS로 가져와
   Nsight GUI에서 열면 타임라인을 볼 수 있다 (별도 가입 불필요)

**확인할 가설 (스터디 공유 실측에서 나온 것)**

| 가설 | 확인 방법 |
|---|---|
| 총 GPU 커널 시간은 base/AWQ가 거의 같다 | `cuda_gpu_kern_sum`의 총합 비교 |
| 줄어든 것은 `cudaEventSynchronize` 대기다 | `cuda_api_sum`의 호출당 평균 시간 |
| 그 단축배 ≈ TPOT 개선배 | 벤치 출력의 TPOT와 대조 |
| AWQ에도 bf16 GEMM이 남는다 | 커널 이름별 비중 (Marlin vs cutlass) |
| GEMM occupancy는 양자화와 무관하다 | ncu의 Achieved Occupancy |

```shell
=== 도구 버전 ===
$ nsys --version; ncu --version | tail -1
NVIDIA Nsight Systems version 2026.1.3.425-261338342291v0
Version 2026.1.1.0 (build 37634170) (public-release)
$ nsys status -e   (WSL2 환경 점검, 요약)
- Root privilege: disabled
- Linux Kernel Paranoid Level: 2 (some features maybe not available)
- Linux Kernel Version (5.15.90.1-microsoft-standard-WSL2): OK
- CPU Profiling Environment (process-tree): OK
- CPU Profiling Environment (system-wide): Fail
including information on how to set the Linux Kernel Paranoid Level.

=== ① Nsight Systems — offline latency (Qwen/Qwen3-8B) ===
$ nsys profile --trace=cuda,nvtx,osrt -o w5_base_lat vllm bench latency --model Qwen/Qwen3-8B --batch-size 8 --input-len 256 --output-len 64 --num-ite …
Avg latency: 1.4896092552000482 seconds
$ nsys stats w5_base_lat.nsys-rep --report cuda_gpu_kern_sum,cuda_api_sum --format csv
  [cuda_gpu_kern_sum] kernels=119  total GPU kernel time = 8.642 s
   %time   total_ms   instances  avg_us   kernel
    19.2     1656.8      12674    130.7   void at::native::vectorized_elementwise_kernel<(int)4, at::native::FillFunctor<int>, std::
    15.4     1328.4       3564    372.7   ampere_bf16_s1688gemm_bf16_128x64_sliced1x2_ldg8_f2f_tn
    11.3      974.0        288   3381.8   ampere_bf16_s16816gemm_bf16_128x64_ldg8_f2f_tn
     9.3      803.9       1606    500.6   void cutlass::Kernel2<cutlass_80_wmma_tensorop_bf16_s161616gemm_bf16_16x16_128x2_tn_align8
     7.8      671.9       1732    387.9   ampere_bf16_s1688gemm_bf16_64x128_sliced1x2_ldg8_f2f_tn
...
  [cuda_api_sum] total CUDA API time = 19.454 s
   %time   total_ms   calls    avg_us    api
    48.1     9356.3      455   20563.2    cudaEventSynchronize
    21.3     4151.7     2161    1921.2    cudaDeviceSynchronize
    10.7     2078.1     3903     532.4    cudaMemcpyAsync
     7.7     1505.5    63914      23.6    cudaLaunchKernel
     3.5      678.4     2304     294.4    cudaMalloc
...

=== ① Nsight Systems — serve wrap + prefix_repetition 200 prompts (Qwen/Qwen3-8B) ===
$ nsys profile --trace=cuda,nvtx,osrt -o w5_base_serve vllm serve Qwen/Qwen3-8B --max-model-len 4096 --gpu-memory-utilization 0.90 --port 8000
$ vllm bench serve --backend vllm --base-url http://127.0.0.1:8000 --model Qwen/Qwen3-8B --dataset-name prefix_repetition --num-prompts 200 --request- …
Successful requests:                     200       
Benchmark duration (s):                  58.41     
Output token throughput (tok/s):         396.14    
Mean TTFT (ms):                          105.17    
Mean TPOT (ms):                          23.69     
Mean ITL (ms):                           23.54     
$ nsys stats w5_base_serve.nsys-rep --report cuda_gpu_kern_sum,cuda_api_sum --format csv
  [cuda_gpu_kern_sum] kernels=116  total GPU kernel time = 8.362 s
   %time   total_ms   instances  avg_us   kernel
    30.4     2541.5       2759    921.2   void cutlass::Kernel2<cutlass_80_wmma_tensorop_bf16_s161616gemm_bf16_16x16_128x2_tn_align8
    14.1     1179.4       3960    297.8   ampere_bf16_s1688gemm_bf16_128x64_sliced1x2_ldg8_f2f_tn
    12.5     1048.3       4212    248.9   void cutlass::Kernel2<cutlass_80_tensorop_bf16_s16816gemm_relu_bf16_64x64_32x6_tn_align8>(
    12.3     1028.6       1589    647.3   _topk_topp_kernel
     9.2      767.0       5616    136.6   void flash::flash_fwd_splitkv_kernel<Flash_fwd_kernel_traits<(int)128, (int)64, (int)128, 
...
  [cuda_api_sum] total CUDA API time = 40.607 s
   %time   total_ms   calls    avg_us    api
    82.7    33598.9     1606   20920.8    cudaEventSynchronize
     3.9     1583.7     9431     167.9    cudaMemcpyAsync
     3.6     1481.7     1940     763.8    cudaDeviceSynchronize
     2.7     1082.5    82429      13.1    cudaLaunchKernel
     1.6      629.7     2302     273.5    cudaMalloc
...

=== ② PyTorch Profiler (CUDA 모드) — serve + /start_profile (Qwen/Qwen3-8B) ===
$ vllm serve Qwen/Qwen3-8B --max-model-len 4096 --gpu-memory-utilization 0.90 --port 8000 --profiler-config '{"profiler":"torch","torch_profiler_dir": …
$ curl -X POST http://127.0.0.1:8000/start_profile

$ vllm bench serve ... --dataset-name prefix_repetition --num-prompts 40 --max-concurrency 10 ...  (프로파일 구간, warmup 5 + active 50 iteration)
Output token throughput (tok/s):         317.31    
Mean TPOT (ms):                          30.64     
$ curl -X POST http://127.0.0.1:8000/stop_profile

$ ls -la torch_base
drwxr-xr-x 2 enginrect enginrect    4096 Sep  6 00:31 .
drwxr-xr-x 3 enginrect enginrect    4096 Sep  6 00:29 ..
-rw-r--r-- 1 enginrect enginrect   20651 Sep  6 00:31 profiler_out_0.txt
-rw-r--r-- 1 enginrect enginrect 1170291 Sep  6 00:30 rank0.1788622250368404020.pt.trace.json.gz
Initial profiling/warmup run took 0.19 s
init engine (profile, create kv cache, warmup model) took 8.26 s (compilation: 0.14 s)
Profiler with mode 'torch' is enabled in the API server. This should ONLY be used for local development!
Route: /start_profile, Methods: POST
Route: /stop_profile, Methods: POST
  trace: rank0.1788622250368404020.pt.trace.json.gz (1.2 MB)
  GPU kernel time total = 1.189 s over 25017 kernels;  CUDA sync/wait API time = 0.988 s
   top kernels (%GPU, ms, count):
     78.7%     935.6     7100  void cutlass::Kernel2<cutlass_80_wmma_tensorop_bf16_s161616gemm_bf16_16x16_128x2_tn_alig
      4.9%      57.9     1762  void flash::flash_fwd_splitkv_kernel<Flash_fwd_kernel_traits<128, 64, 128, 4, false, fal
      4.0%      47.1      108  ampere_bf16_s1688gemm_bf16_128x64_sliced1x2_ldg8_f2f_tn
      3.8%      45.0       36  ampere_bf16_s16816gemm_bf16_128x64_ldg8_f2f_tn
      2.9%      35.1       72  void cutlass::Kernel2<cutlass_80_tensorop_bf16_s16816gemm_relu_bf16_64x64_32x6_tn_align8
      0.7%       8.8       51  _topk_topp_kernel
      0.7%       8.2       36  ampere_bf16_s1688gemm_bf16_64x64_sliced1x4_ldg8_f2f_tn
      0.6%       7.2       36  ampere_bf16_s1688gemm_bf16_64x128_sliced1x2_ldg8_f2f_tn
   sync APIs:
       966.9 ms      50 calls  avg 19338.8 us  cudaEventSynchronize
        21.4 ms       1 calls  avg 21391.1 us  cudaDeviceSynchronize
         0.1 ms      50 calls  avg     1.3 us  cudaStreamWaitEvent

=== ① Nsight Systems — offline latency (Qwen/Qwen3-8B-AWQ) ===
$ nsys profile --trace=cuda,nvtx,osrt -o w5_awq_lat vllm bench latency --model Qwen/Qwen3-8B-AWQ --batch-size 8 --input-len 256 --output-len 64 --num- …
Avg latency: 0.700741453000046 seconds
$ nsys stats w5_awq_lat.nsys-rep --report cuda_gpu_kern_sum,cuda_api_sum --format csv
  [cuda_gpu_kern_sum] kernels=127  total GPU kernel time = 8.497 s
   %time   total_ms   instances  avg_us   kernel
    37.9     3222.3       9648    334.0   void marlin::Marlin<(long)1125899906910725, (long)1125899906843648, (long)1125899906910725
    18.0     1533.5      12782    120.0   void at::native::vectorized_elementwise_kernel<(int)4, at::native::FillFunctor<int>, std::
     7.9      668.6        455   1469.4   void cutlass::Kernel2<cutlass_80_wmma_tensorop_f16_s161616gemm_f16_16x16_128x2_tn_align8>(
     6.4      543.5       2689    202.1   triton_
     5.0      425.8       1692    251.7   void flash::flash_fwd_splitkv_kernel<Flash_fwd_kernel_traits<(int)128, (int)64, (int)128, 
...
  [cuda_api_sum] total CUDA API time = 12.362 s
   %time   total_ms   calls    avg_us    api
    31.9     3941.7      455    8663.2    cudaEventSynchronize
    28.3     3498.6     2161    1619.0    cudaDeviceSynchronize
    12.9     1591.4    92786      17.2    cudaLaunchKernel
     7.2      887.1     5559     159.6    cudaMemcpyAsync
     5.6      689.9     2432     283.7    cudaMalloc
...

=== ① Nsight Systems — serve wrap + prefix_repetition 200 prompts (Qwen/Qwen3-8B-AWQ) ===
$ nsys profile --trace=cuda,nvtx,osrt -o w5_awq_serve vllm serve Qwen/Qwen3-8B-AWQ --max-model-len 4096 --gpu-memory-utilization 0.90 --port 8000
$ vllm bench serve --backend vllm --base-url http://127.0.0.1:8000 --model Qwen/Qwen3-8B-AWQ --dataset-name prefix_repetition --num-prompts 200 --requ …
Successful requests:                     200       
Benchmark duration (s):                  43.53     
Output token throughput (tok/s):         526.89    
Mean TTFT (ms):                          72.81     
Mean TPOT (ms):                          11.25     
Mean ITL (ms):                           10.89     
$ nsys stats w5_awq_serve.nsys-rep --report cuda_gpu_kern_sum,cuda_api_sum --format csv
  [cuda_gpu_kern_sum] kernels=125  total GPU kernel time = 10.826 s
   %time   total_ms   instances  avg_us   kernel
    45.0     4874.6       3281   1485.7   void cutlass::Kernel2<cutlass_80_wmma_tensorop_f16_s161616gemm_f16_16x16_128x2_tn_align8>(
    18.9     2045.8       8712    234.8   void marlin::Marlin<(long)1125899906910725, (long)1125899906843648, (long)1125899906910725
     7.7      832.0       7668    108.5   void flash::flash_fwd_splitkv_kernel<Flash_fwd_kernel_traits<(int)128, (int)64, (int)128, 
     6.4      695.1       1135    612.4   _topk_topp_kernel
     3.3      362.4       2148    168.7   void at::native::tensor_kernel_scan_innermost_dim<float, std::plus<float>>(T1 *, const T1 
...
  [cuda_api_sum] total CUDA API time = 35.150 s
   %time   total_ms   calls    avg_us    api
    78.9    27739.8     3437    8070.9    cudaEventSynchronize
     4.9     1714.4   193294       8.9    cudaLaunchKernel
     3.2     1128.5     1940     581.7    cudaDeviceSynchronize
     3.2     1118.3    21363      52.3    cudaMemcpyAsync
     2.0      687.5     2430     282.9    cudaMalloc
...

=== ② PyTorch Profiler (CUDA 모드) — serve + /start_profile (Qwen/Qwen3-8B-AWQ) ===
$ vllm serve Qwen/Qwen3-8B-AWQ --max-model-len 4096 --gpu-memory-utilization 0.90 --port 8000 --profiler-config '{"profiler":"torch","torch_profiler_d …
$ curl -X POST http://127.0.0.1:8000/start_profile

$ vllm bench serve ... --dataset-name prefix_repetition --num-prompts 40 --max-concurrency 10 ...  (프로파일 구간, warmup 5 + active 50 iteration)
Output token throughput (tok/s):         655.91    
Mean TPOT (ms):                          12.59     
$ curl -X POST http://127.0.0.1:8000/stop_profile

$ ls -la torch_awq
drwxr-xr-x 2 enginrect enginrect    4096 Sep  6 00:36 .
drwxr-xr-x 4 enginrect enginrect    4096 Sep  6 00:35 ..
-rw-r--r-- 1 enginrect enginrect   20063 Sep  6 00:36 profiler_out_0.txt
-rw-r--r-- 1 enginrect enginrect 1061505 Sep  6 00:35 rank0.1788622557229884872.pt.trace.json.gz
Initial profiling/warmup run took 0.13 s
init engine (profile, create kv cache, warmup model) took 7.60 s (compilation: 0.15 s)
Profiler with mode 'torch' is enabled in the API server. This should ONLY be used for local development!
Route: /start_profile, Methods: POST
Route: /stop_profile, Methods: POST
  trace: rank0.1788622557229884872.pt.trace.json.gz (1.1 MB)
  GPU kernel time total = 0.477 s over 25196 kernels;  CUDA sync/wait API time = 0.341 s
   top kernels (%GPU, ms, count):
     38.2%     182.4     3664  void marlin::Marlin<1125899906910725l, 1125899906843648l, 1125899906910725l, 11258999069
     18.6%      88.6     3665  void marlin::Marlin<1125899906910725l, 1125899906843648l, 1125899906910725l, 11258999069
     16.7%      79.5       51  void cutlass::Kernel2<cutlass_80_wmma_tensorop_f16_s161616gemm_f16_16x16_128x2_tn_align8
     12.9%      61.4     1832  void flash::flash_fwd_splitkv_kernel<Flash_fwd_kernel_traits<128, 64, 128, 4, false, fal
      6.8%      32.3       51  _topk_topp_kernel
      1.4%       6.7     3664  void at::native::elementwise_kernel<128, 4, at::native::gpu_kernel_impl_nocast<at::nativ
      1.2%       5.6     1832  void flash::flash_fwd_splitkv_combine_kernel<Flash_fwd_kernel_traits<128, 64, 128, 4, fa
      0.9%       4.1     1832  triton_red_fused_fused_add_rms_norm_marlin_gemm_0
   sync APIs:
       330.2 ms      50 calls  avg  6604.2 us  cudaEventSynchronize
        10.8 ms       1 calls  avg 10793.5 us  cudaDeviceSynchronize
         0.1 ms      50 calls  avg     1.4 us  cudaStreamWaitEvent

=== ③ Nsight Compute — GPU 카운터 권한 ===
$ docker stop mon-dcgm-exporter-1   # 프로파일링 카운터 충돌 배제
mon-dcgm-exporter-1
$ ncu --query-metrics | head -3
Device NVIDIA GeForce RTX 4090 (AD102)
==ERROR== ERR_NVGPUCTRPERM - The user does not have permission to access NVIDIA GPU Performance Counters on the target device 0. For instructions on e …

$ sudo ncu --query-metrics | head -3
==ERROR== ERR_NVGPUCTRPERM - The user does not have permission to access NVIDIA GPU Performance Counters on the target device 0. For instructions on e …

$ ncu --set full --launch-count 20 -o w5_base_ncu vllm bench latency --model Qwen/Qwen3-8B --batch-size 8 --input-len 256 --output-len 64 --enforce-ea …
==PROF== Connected to process 18084 (/usr/bin/python3.12)
==ERROR== ERR_NVGPUCTRPERM - The user does not have permission to access NVIDIA GPU Performance Counters on the target device 0. For instructions on e …
==PROF== Trying to shutdown target application
==ERROR== The application returned an error code (9).
$ reg.exe query "HKLM\SYSTEM\CurrentControlSet\Services\nvlddmkm\Global\NVTweak" | grep -c -i RmProfilingAdminOnly
0
$ docker start mon-dcgm-exporter-1
mon-dcgm-exporter-1
=== DONE Lab 2 ===
```

**① Nsight Systems — offline `vllm bench latency` (batch 8, in 256 / out 64, 5 iter)**

| | Avg latency | 커널 총시간 (kern_sum) | `cudaEventSynchronize` 호출당 평균 | 최대 커널 |
|---|---|---|---|---|
| Qwen3-8B | 1.489s | 8.642s | 20563.2µs | FillFunctor(memset) 19.2%, bf16 GEMM 15.4% |
| Qwen3-8B-AWQ | 0.700s | 8.497s | 8663.2µs | Marlin 37.9%, FillFunctor 18.0% |

**① Nsight Systems — `vllm serve` 감싸기 + prefix_repetition 200 (rate 5, c10)**

| | 출력 TPS | TPOT | 커널 총시간 (kern_sum) | `cudaEventSynchronize` 총 / 평균 | 최대 커널 |
|---|---|---|---|---|---|
| Qwen3-8B | 396.14 | 23.69ms | 8.362s | 33.6s / 20.9ms | bf16 wmma GEMM 30.4% |
| Qwen3-8B-AWQ | 526.89 | 11.25ms | 10.826s | 27.7s / 8.1ms | **f16 wmma GEMM 45.0%**, Marlin 18.9% |

**② PyTorch Profiler (CUDA 모드, `--profiler-config`, warmup 5 + active 50 iteration, prefix c10)**

| | 50 iteration 커널 총시간 | 커널 수 | 최대 커널 | `cudaEventSynchronize` 50회 평균 | 같은 구간 TPOT |
|---|---|---|---|---|---|
| Qwen3-8B | 1.189s | 25017 | bf16 wmma GEMM 78.7% | 19338.8µs | 30.64ms |
| Qwen3-8B-AWQ | 0.477s | 25196 | Marlin 38.2% + 18.6%, f16 GEMM 16.7% | 6604.2µs | 12.59ms |

**가설 판정**

| 가설 (스터디 공유) | 판정 | 근거 |
|---|---|---|
| 총 GPU 커널 시간은 base/AWQ가 거의 같다 | **기각** (디코드 정상 상태) | torch profiler 50 iteration: 1.189s vs 0.477s. nsys offline의 8.64 vs 8.50s는 로드·워밍업·graph capture가 지배한 값이라 판정 근거로 못 쓴다 (아래 보강) |
| 줄어든 것은 `cudaEventSynchronize` 대기다 | 부분 확인 | 두 도구 모두에서 API 시간 1위. 다만 이 대기는 GPU가 스텝을 끝내길 기다리는 시간이라 "커널이 빨라진 결과"를 보는 창이지 원인이 아니다 |
| 그 단축배 ≈ TPOT 개선배 | 확인 | sync 평균 19.3 → 6.6ms (2.9배) vs TPOT 30.6 → 12.6ms (2.4배); serve-wrap에서는 20.9 → 8.1ms (2.6배) vs 23.7 → 11.3ms (2.1배) |
| AWQ에도 bf16/f16 GEMM이 남는다 | 확인 | AWQ serve-wrap 커널 1위가 cutlass f16 wmma GEMM 45.0% (prefill 구간의 큰 M), Marlin은 디코드 GEMM |
| GEMM occupancy는 양자화와 무관 | **미확인** | Nsight Compute 권한 문제로 측정 불가 (③) |

- nsys 기본 설정(`--cuda-graph-trace=graph`)에서는 CUDA graph 재생 안의 커널이 kern_sum에 개별 집계되지 않는다. serve-wrap에서 58초 벤치·GPU util 95%인데 커널 합이 8.4초로 나온 이유가 그것이다. 아래 보강에서 node 단위로 다시 잰다
- torch profiler는 CUPTI로 graph 내부 커널을 개별 기록한다 — 50 iteration에 25,017개 커널, 합 1.189s → iteration당 23.8ms ≈ 같은 구간 TPOT. 이 도구의 수치는 디코드 스텝 그대로다

**③ Nsight Compute — WSL2 권한**

- 일반 사용자·`sudo` 모두 `ERR_NVGPUCTRPERM`. DCGM exporter를 내려도 같다 — 카운터 충돌이 아니라 권한 문제다
- WSL2에서는 드라이버가 Windows 측(WDDM)이라 리눅스의 `NVreg_RestrictProfilingToAdminUsers=0` 방식이 없다. Windows NVIDIA 제어판 → 개발자 설정 → "GPU 성능 카운터 액세스를 모든 사용자에게 허용" 또는 레지스트리 `nvlddmkm\Global\NVTweak\RmProfilingAdminOnly=0` 후 재부팅이 필요하다. 레지스트리 조회 결과 해당 값이 없음(기본 admin-only)을 확인했고, 설정 변경은 재부팅을 요구해 이번 세션에서는 하지 않았다
- `ncu --set full ... vllm bench latency`는 권한 오류 상태에서 종료되지 않고 55분간 멈춰 있어 수동으로 kill했다 — 계획서의 "ncu는 서버 방식과 상성이 나쁘다"보다 앞 단계에서 막힌다

**보강 — `nsys --cuda-graph-trace=node` (offline latency 동일 조건)**

```shell
=== nsys --cuda-graph-trace=node — offline latency (Qwen/Qwen3-8B) ===
$ nsys profile --trace=cuda,nvtx --cuda-graph-trace=node -o w5_base_lat_node vllm bench latency --model Qwen/Qwen3-8B --batch-size 8 --input-len 256 - …
Avg latency: 1.5604789826000343 seconds
$ nsys stats w5_base_lat_node.nsys-rep --report cuda_gpu_kern_sum,cuda_api_sum --format csv
  [cuda_gpu_kern_sum] kernels=117  total GPU kernel time = 15.141 s
   %time   total_ms   instances  avg_us   kernel
    58.1     8804.6      65256    134.9   void cutlass::Kernel2<cutlass_80_wmma_tensorop_bf16_s161616gemm_bf16_16x16_128x2_tn_align8
    10.8     1633.1       4176    391.1   ampere_bf16_s1688gemm_bf16_128x64_sliced1x2_ldg8_f2f_tn
     6.8     1034.9       2092    494.7   ampere_bf16_s1688gemm_bf16_64x128_sliced1x2_ldg8_f2f_tn
     6.0      913.4        288   3171.6   ampere_bf16_s16816gemm_bf16_128x64_ldg8_f2f_tn
     2.9      439.3       1692    259.7   void flash::flash_fwd_splitkv_kernel<Flash_fwd_kernel_traits<(int)128, (int)64, (int)128, 
...
  [cuda_api_sum] total CUDA API time = 18.717 s
   %time   total_ms   calls    avg_us    api
    51.8     9689.3      455   21295.1    cudaEventSynchronize
    17.9     3346.9     3781     885.2    cudaMemcpyAsync
    10.1     1890.3     1933     977.9    cudaDeviceSynchronize
     7.8     1450.9    53814      27.0    cudaLaunchKernel
     3.1      581.6     2305     252.3    cudaMalloc
...
$ nsys stats w5_base_lat_node.nsys-rep --report nvtx_kern_sum --format csv | head   (NVTX 구간별 커널 합)

=== nsys --cuda-graph-trace=node — offline latency (Qwen/Qwen3-8B-AWQ) ===
$ nsys profile --trace=cuda,nvtx --cuda-graph-trace=node -o w5_awq_lat_node vllm bench latency --model Qwen/Qwen3-8B-AWQ --batch-size 8 --input-len 25 …
Avg latency: 0.7896949116002361 seconds
$ nsys stats w5_awq_lat_node.nsys-rep --report cuda_gpu_kern_sum,cuda_api_sum --format csv
  [cuda_gpu_kern_sum] kernels=124  total GPU kernel time = 9.459 s
   %time   total_ms   instances  avg_us   kernel
    41.8     3955.9      10224    386.9   void marlin::Marlin<(long)1125899906910725, (long)1125899906843648, (long)1125899906910725
    17.1     1619.2      33408     48.5   void marlin::Marlin<(long)1125899906910725, (long)1125899906843648, (long)1125899906910725
     8.7      820.2      33408     24.5   void marlin::Marlin<(long)1125899906910725, (long)1125899906843648, (long)1125899906910725
     7.4      704.6        455   1548.5   void cutlass::Kernel2<cutlass_80_wmma_tensorop_f16_s161616gemm_f16_16x16_128x2_tn_align8>(
     4.6      435.6       1692    257.4   void flash::flash_fwd_splitkv_kernel<Flash_fwd_kernel_traits<(int)128, (int)64, (int)128, 
...
  [cuda_api_sum] total CUDA API time = 11.790 s
   %time   total_ms   calls    avg_us    api
    37.4     4407.8      455    9687.6    cudaEventSynchronize
    18.7     2206.5     5437     405.8    cudaMemcpyAsync
    14.1     1657.5    82225      20.2    cudaLaunchKernel
     9.9     1170.0     1933     605.3    cudaDeviceSynchronize
     5.5      647.3     2432     266.1    cudaMalloc
...
$ nsys stats w5_awq_lat_node.nsys-rep --report nvtx_kern_sum --format csv | head   (NVTX 구간별 커널 합)

=== DONE Lab 2b ===
```

| | Avg latency | 커널 총시간 (node 단위) | `cudaEventSynchronize` 평균 |
|---|---|---|---|
| Qwen3-8B | 1.560s | 15.141s | 21295.1µs |
| Qwen3-8B-AWQ | 0.789s | 9.459s | 9687.6µs |

- key point: 3계층 중 nsys(GPU가 바쁜가: 예, sync 대기가 API 1위)와 PyTorch Profiler(어느 커널: bf16 GEMM 78.7% → Marlin 57%, 커널 합 2.5배 단축)까지 도달했고, Nsight Compute는 WSL2 권한 벽에서 멈췼다. "커널 시간은 같고 대기만 줄었다"는 가설은 디코드 정상 상태에서는 성립하지 않았다.


### Lab 3 — 서빙 모니터링 스택 ✅

프로파일러가 사후 분석이라면, 이건 벤치 도중의 시계열이다. 4090은 NVLink가 없으므로
**PCIe 대역폭**이 추가 관찰 대상이다.

구성: Prometheus + Grafana + node_exporter + **DCGM exporter**(GPU) +
**NVML PCIe exporter**(직접 작성).

```bash
# docker compose로 기동 (스터디 공유 구성 참고)
docker compose up -d
# Grafana http://localhost:3000 (admin/admin)
```

**⚠️ DCGM의 한계**: GeForce 카드에서는 DCP 프로파일링 메트릭
(`DCGM_FI_PROF_PCIE_TX/RX_BYTES`)과 레거시 필드가 모두 동작하지 않는다. NVML의
`nvmlDeviceGetPcieThroughput`은 동작하므로, `nvidia-ml-py` + `prometheus-client`로
작은 exporter를 만들어 TX/RX를 직접 폴링한다.

**볼 것**: 벤치 구간의 ① GPU 사용률·전력 ② VRAM 점유 ③ PCIe TX/RX ④ P-state 변화.
Lab 1의 베이스 vs AWQ 구간을 같은 대시보드에서 대조한다.

```shell
$ cat mon/docker-compose.yml
services:
  prometheus:
    image: prom/prometheus:v3.5.0
    network_mode: host
    volumes:
      - ./prometheus.yml:/etc/prometheus/prometheus.yml:ro
      - prom-data:/prometheus
    command: ["--config.file=/etc/prometheus/prometheus.yml", "--storage.tsdb.retention.time=7d"]
  grafana:
    image: grafana/grafana:12.1.1
    network_mode: host
    environment:
      - GF_SECURITY_ADMIN_PASSWORD=admin
    volumes:
      - ./grafana/provisioning:/etc/grafana/provisioning:ro
      - ./grafana/dashboards:/var/lib/grafana/dashboards:ro
  node-exporter:
    image: prom/node-exporter:v1.9.1
    network_mode: host
    pid: host
    command: ["--path.rootfs=/host"]
    volumes:
      - /:/host:ro
  dcgm-exporter:
    image: nvcr.io/nvidia/k8s/dcgm-exporter:4.4.0-4.5.0-ubuntu22.04
    network_mode: host
    cap_add: [SYS_ADMIN]
    command: ["-f", "/etc/dcgm-exporter/custom.csv"]
    volumes:
      - ./custom-counters.csv:/etc/dcgm-exporter/custom.csv:ro
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: all
              capabilities: [gpu]
volumes:
  prom-data:

$ cat mon/prometheus.yml
global:
  scrape_interval: 5s
scrape_configs:
  - job_name: node
    static_configs: [{ targets: ["localhost:9100"] }]
  - job_name: dcgm
    static_configs: [{ targets: ["localhost:9400"] }]
  - job_name: nvml_pcie
    static_configs: [{ targets: ["localhost:9401"] }]
  - job_name: vllm
    static_configs: [{ targets: ["localhost:8000"] }]

$ cat mon/nvml_pcie_exporter.py
#!/usr/bin/env python3
"""NVML PCIe/P-state exporter — GeForce에서 DCGM PROF 필드가 안 나와 nvmlDeviceGetPcieThroughput을 직접 폴링한다."""
import time
import pynvml as nv
from prometheus_client import Gauge, start_http_server
...  # (전문 생략 — NVML PCIe/P-state 폴링)

$ tail -5 mon/custom-counters.csv   # 기본 csv + DCP(PROF) 필드 5개 추가
DCGM_FI_PROF_PCIE_TX_BYTES, counter, PCIe TX bytes (DCP).
DCGM_FI_PROF_PCIE_RX_BYTES, counter, PCIe RX bytes (DCP).
DCGM_FI_PROF_GR_ENGINE_ACTIVE, gauge, Graphics engine active ratio (DCP).
DCGM_FI_PROF_SM_ACTIVE, gauge, SM active ratio (DCP).
DCGM_FI_PROF_DRAM_ACTIVE, gauge, DRAM active ratio (DCP).

$ docker compose up -d
...
 Container mon-node-exporter-1  Started
 Container mon-grafana-1  Started
 Container mon-prometheus-1  Started
 Container mon-dcgm-exporter-1  Started

$ nohup python nvml_pcie_exporter.py &
$ docker ps --format 'table {{.Names}}\t{{.Status}}'
NAMES                 STATUS
mon-dcgm-exporter-1   Up 12 seconds
mon-prometheus-1      Up 12 seconds
...

$ docker logs mon-dcgm-exporter-1 2>&1 | grep 'Skipping line'
"Skipping line 21 ('DCGM_FI_PROF_GR_ENGINE_ACTIVE'): metric not enabled"
"Skipping line 22 ('DCGM_FI_PROF_PIPE_TENSOR_ACTIVE'): metric not enabled"
"Skipping line 23 ('DCGM_FI_PROF_DRAM_ACTIVE'): metric not enabled"
...   # PROF(DCP) 필드 10개 전부 'metric not enabled'

$ curl -s localhost:9400/metrics | grep -c '^DCGM_FI_PROF'
0

$ curl -s localhost:9090/api/v1/targets | python3 -c "...job, health..."
dcgm up
node up
nvml_pcie up
vllm down

$ curl -s localhost:9401/metrics | grep '^nvml_'   # 유휴 상태 샘플
nvml_pcie_tx_kb_per_s 5029.0
nvml_pcie_rx_kb_per_s 16894.0
nvml_pstate 2.0
nvml_gpu_util_percent 100.0
nvml_mem_used_bytes 2.5631809536e+10
nvml_power_watts 111.542
nvml_sm_clock_mhz 2730.0
nvml_temp_c 46.0
nvml_pcie_link_gen_width 416.0

$ python prom_range.py "base / sharegpt" <T_START> <T_END>   # Lab 1 벤치 구간 query_range 요약
[base / sharegpt]  window 1559s, step 5s
  metric                         min      mean       max
  GPU util % (DCGM)              6.0      97.5     100.0
  GPU util % (NVML)              5.0      98.6     100.0
  Power W                       31.0     218.2     318.5
  VRAM used GiB (DCGM)          17.1      23.3      23.5
  PCIe TX MB/s (NVML)            3.1    3233.4    9606.8
  PCIe RX MB/s (NVML)            4.1    2880.5    8206.1
  P-state                        2.0       2.1       8.0
  SM clock MHz                 315.0    2694.2    2730.0
  CPU busy % (node)              2.9       3.5       3.7
  vLLM running reqs              0.0       9.8      10.0
  vLLM gen tok/s                 0.0     264.9     469.1
  vLLM prompt tok/s              0.0     289.9     807.6

$ python prom_range.py "base / prefix_repetition" <T_START> <T_END>   # Lab 1 벤치 구간 query_range 요약
[base / prefix_repetition]  window 307s, step 5s
  metric                         min      mean       max
  GPU util % (DCGM)             16.0      91.4     100.0
  GPU util % (NVML)              3.0      96.7     100.0
  Power W                       33.1     285.8     325.2
  VRAM used GiB (DCGM)          23.3      23.3      23.3
  PCIe TX MB/s (NVML)            0.5    1474.1    6902.4
  PCIe RX MB/s (NVML)            2.6     833.9    6830.6
  P-state                        2.0       2.1       8.0
  SM clock MHz                 345.0    2677.7    2730.0
  CPU busy % (node)              3.3       3.6       4.2
  vLLM running reqs              0.0       9.3      10.0
  vLLM gen tok/s                76.9     367.7     408.2
  vLLM prompt tok/s              0.3    1634.4    1926.5

$ python prom_range.py "awq / sharegpt" <T_START> <T_END>   # Lab 1 벤치 구간 query_range 요약
[awq / sharegpt]  window 406s, step 5s
  metric                         min      mean       max
  GPU util % (DCGM)              5.0      93.3      99.0
  GPU util % (NVML)              3.0      93.7      99.0
  Power W                       26.3     303.7     348.0
  VRAM used GiB (DCGM)          22.8      22.9      23.0
  PCIe TX MB/s (NVML)            0.5      39.0      93.9
  PCIe RX MB/s (NVML)            2.6      40.1     121.3
  P-state                        2.0       2.3       8.0
  SM clock MHz                 270.0    2572.1    2730.0
  CPU busy % (node)              2.9       3.6       3.9
  vLLM running reqs              0.0       9.4      10.0
  vLLM gen tok/s                 0.0    1007.3    1169.3
  vLLM prompt tok/s              0.0    1095.2    1724.0

$ python prom_range.py "awq / prefix_repetition" <T_START> <T_END>   # Lab 1 벤치 구간 query_range 요약
[awq / prefix_repetition]  window 209s, step 5s
  metric                         min      mean       max
  GPU util % (DCGM)              2.0      84.5      99.0
  GPU util % (NVML)              1.0      86.9      99.0
  Power W                       31.9     280.6     342.9
  VRAM used GiB (DCGM)          22.9      22.9      23.0
  PCIe TX MB/s (NVML)            0.1      36.1      72.1
  PCIe RX MB/s (NVML)            0.1      25.2      70.2
  P-state                        2.0       2.2       8.0
  SM clock MHz                 210.0    2597.1    2730.0
  CPU busy % (node)              3.2       3.5       4.2
  vLLM running reqs              0.0       5.4      10.0
  vLLM gen tok/s               336.9     564.0    1027.6
  vLLM prompt tok/s            558.2    2319.4    3020.4

# gpu  rxpci  txpci     sm    mem    enc    dec    jpg    ofa 
# Idx   MB/s   MB/s      %      %      %      %      %      % 
    0    655    758     99     81      0      2      0      0 
    0     51     28    100     48      0      2      0      0 
    0     14   3757    100     91      0      2      0      0 
    0    675   3767    100     75      0      4      0      0 
    0   5798    316    100     73      0      2      0      0 
    0   1097      5    100     84      0      2      0      0
```

**Lab 1 벤치 4구간의 시계열 요약 (Prometheus query_range, 5초 step)**

| 구간 | GPU util 평균 (NVML) | 전력 평균/최대 | VRAM (DCGM) | PCIe TX 평균/최대 | PCIe RX 평균/최대 | P-state | CPU busy |
|---|---|---|---|---|---|---|---|
| base / ShareGPT | 98.6% | 218 / 319 W | 23.3 GiB | **3,233 / 9,607 MB/s** | 2,881 / 8,206 MB/s | 2.1 | 3.5% |
| base / Prefix | 96.7% | 286 / 325 W | 23.3 GiB | **1,474 / 6,902 MB/s** | 834 / 6,831 MB/s | 2.1 | 3.6% |
| AWQ / ShareGPT | 93.7% | 304 / 348 W | 22.9 GiB | 39 / 94 MB/s | 40 / 121 MB/s | 2.3 | 3.6% |
| AWQ / Prefix | 86.9% | 281 / 343 W | 22.9 GiB | 36 / 72 MB/s | 25 / 70 MB/s | 2.2 | 3.5% |

- **DCGM의 한계 실측**: 기본 csv에 들어 있는 `DCGM_FI_PROF_PCIE_TX/RX_BYTES`, `PROF_GR_ENGINE_ACTIVE`, `PROF_SM_ACTIVE`, `PROF_DRAM_ACTIVE`가 GeForce에서 전부 "metric not enabled"로 스킵되고 `DCGM_FI_PROF_*` 노출 수 0. 계획서대로 NVML `nvmlDeviceGetPcieThroughput`을 폴링하는 exporter를 직접 만들어 PCIe를 채웠다
- PCIe TX/RX는 벤치 구간에서 실제로 움직였고, 베이스와 AWQ 사이에 **80배** 차이가 났다. 이 차이가 Lab 1 보강 실험(0.90 vs 0.85)의 출발점이 됐다 — 모니터링이 프로파일러보다 먼저 이상을 잡은 사례다
- 전력은 AWQ 쪽이 더 높다(304 vs 218 W) — PCIe 대기 없이 SM이 계속 도는 상태의 소비다. 베이스는 util 98%인데 전력이 낮다, 즉 **util%는 높지만 실제 연산은 덜 하고 있었다**
- CPU busy는 32코어 기준 3.5%(≈1.1코어) — 이 부하에서 CPU 병목은 없다. WSL2의 node_exporter idle 카운터가 32를 넘는 값을 내서 `100 − idle` 식은 음수가 됐고, `sum(non-idle)/count(cpu)` 식으로 바꿔야 했다
- node_exporter는 WSL2에서 `/:/host:ro,rslave` 마운트를 거부한다(`path / is mounted on / but it is not a shared or slave mount`) — `rslave`를 빼면 뜬다
- Grafana(3000)에 데이터소스·대시보드를 프로비저닝했지만 스크린샷은 남기지 않는다. 위 표가 그 대시보드의 수치다

- key point: GeForce에서 DCGM PROF 필드는 전부 비활성이고, NVML 직접 폴링으로 채운 PCIe TX가 베이스 벤치 구간에서 3.2 GB/s(AWQ 39 MB/s)를 가리켜 Lab 1 측정의 손상을 먼저 드러냈다.


### Lab 4 — 게이트웨이 라우팅과 시맨틱 캐싱 ✅

4090 1장에 **소형 모델 2개**를 각각 다른 포트로 띄우고 그 앞에 게이트웨이를 둔다.
GPU 메모리를 나눠 쓰도록 `--gpu-memory-utilization`을 낮춘다.

```bash
# 백엔드 2대 (예: 0.5B급 2개 또는 소형 모델 2종)
vllm serve Qwen/Qwen2.5-0.5B-Instruct --port 8001 --gpu-memory-utilization 0.35
vllm serve Qwen/Qwen2.5-1.5B-Instruct --port 8002 --gpu-memory-utilization 0.35
```

**A. LiteLLM 게이트웨이** — Docker로 기동, `model_list`에 두 백엔드를 등록하고
RPM/TPM 기반 라우팅 전략을 건다.

- 통일된 OpenAI 형식으로 호출되는지
- 한쪽을 죽였을 때 **폴백**이 도는지
- **PII 마스킹**이 실제로 치환하는지
- `Auto Routing`(베타)으로 유사 프롬프트가 같은 경로로 가는지

**B. vLLM Semantic Router** — 유사하지만 문자열이 다른 프롬프트 2개를 순서대로
보내고, 두 번째가 **캐시로 처리되는지**(LLM 호출 없이) 확인한다. 백엔드 하나를
내려 페일오버도 함께 본다.

**⚠️ 주의**: 시맨틱 캐싱은 임베딩 모델을 별도로 호출한다. 그 오버헤드가 캐시 이득을
넘어서는 구간이 있는지도 같이 재야 의미가 있다.

```shell
=== 백엔드 2대 ===
$ vllm serve Qwen/Qwen2.5-0.5B-Instruct --port 8001 --gpu-memory-utilization 0.30 --max-model-len 4096
$ vllm serve Qwen/Qwen2.5-1.5B-Instruct --port 8002 --gpu-memory-utilization 0.30 --max-model-len 4096
$ nvidia-smi --query-gpu=memory.used,memory.total --format=csv
memory.used [MiB], memory.total [MiB]
16751 MiB, 24564 MiB
GPU KV cache size: 476,672 tokens
Maximum concurrency for 4,096 tokens per request: 116.38x
GPU KV cache size: 114,432 tokens
Maximum concurrency for 4,096 tokens per request: 27.94x

=== A. LiteLLM 게이트웨이 ===
$ docker run -d --name presidio-analyzer -p 5002:3000 mcr.microsoft.com/presidio-analyzer:latest
$ docker run -d --name presidio-anonymizer -p 5001:3000 mcr.microsoft.com/presidio-anonymizer:latest
$ cat litellm_config.yaml
model_list:
  - model_name: small          # 0.5B 백엔드
    litellm_params:
      model: hosted_vllm/Qwen/Qwen2.5-0.5B-Instruct
      api_base: http://host.docker.internal:8001/v1
      api_key: none
      rpm: 600
  - model_name: medium         # 1.5B 백엔드
    litellm_params:
      model: hosted_vllm/Qwen/Qwen2.5-1.5B-Instruct
      api_base: http://host.docker.internal:8002/v1
      api_key: none
      rpm: 600
  - model_name: pool           # 같은 별칭에 두 백엔드 → 라우팅 전략 대상
    litellm_params:
      model: hosted_vllm/Qwen/Qwen2.5-0.5B-Instruct
      api_base: http://host.docker.internal:8001/v1
      api_key: none
      rpm: 600
  - model_name: pool
    litellm_params:
      model: hosted_vllm/Qwen/Qwen2.5-1.5B-Instruct
      api_base: http://host.docker.internal:8002/v1
      api_key: none
      rpm: 600
  - model_name: smart-router
    litellm_params:
      model: auto_router/complexity_router
      complexity_router_config:
        tiers:
          SIMPLE: small
          MEDIUM: small
          COMPLEX: medium
          REASONING: medium
      complexity_router_default_model: small

router_settings:
  routing_strategy: usage-based-routing-v2
  num_retries: 1
  fallbacks:
    - medium: [small]
    - small: [medium]

guardrails:
  - guardrail_name: presidio-pii-mask
    litellm_params:
      guardrail: presidio
      mode: pre_call
      output_parse_pii: false   # 출력에 남는 <PLACEHOLDER>를 그대로 보기 위해 false
      presidio_analyzer_api_base: http://host.docker.internal:5002
      presidio_anonymizer_api_base: http://host.docker.internal:5001

general_settings:
  master_key: sk-1234

litellm_settings:
  request_timeout: 60
  set_verbose: false
$ docker run -d --name litellm --add-host=host.docker.internal:host-gateway -p 4000:4000 -v $PWD/litellm_config.yaml:/app/config.yaml ghcr.io/berriai/ …
$ docker ps --format "table {{.Names}}\t{{.Image}}\t{{.Status}}"
NAMES                 IMAGE                                                      STATUS
litellm               ghcr.io/berriai/litellm:main-stable                        Up 9 seconds
presidio-anonymizer   mcr.microsoft.com/presidio-anonymizer:latest               Up 9 seconds (healthy)
presidio-analyzer     mcr.microsoft.com/presidio-analyzer:latest                 Up 10 seconds (healthy)
$ curl -s -H "Authorization: Bearer sk-1234" localhost:4000/v1/models
['small', 'medium', 'pool', 'smart-router']

--- A1. 통일 OpenAI 형식: small / medium 직접 지정 ---
$ gw_client chat http://localhost:4000 small "What is KV cache in one sentence?"
     667ms  served_model=small  hdr={"x-litellm-model-id": "e1e0fb65…", "x-litellm-model-name": "hosted_vllm/Qwen/Qwen2.5-0.5B-Instruct", "x-litellm-m …
  -> 'KV (Key-Value) Cache is a type of memory-based caching system that stores key-value pairs, allowing for quick retrieval and modification of data …
$ gw_client chat http://localhost:4000 medium "What is KV cache in one sentence?"
     128ms  served_model=medium  hdr={"x-litellm-model-id": "8e447c06…", "x-litellm-model-name": "hosted_vllm/Qwen/Qwen2.5-1.5B-Instruct", "x-litellm- …
  -> 'KV cache refers to key-value caching, which stores data as key-value pairs for quick access and retrieval.'

--- A2. 라우팅 전략(usage-based-routing-v2): 별칭 pool 로 20회 ---
$ gw_client dist http://localhost:4000 pool 20
   20  pool

--- A3. 폴백: medium 백엔드(8002) 를 죽이고 medium 호출 ---
$ kill <vllm 8002>
$ gw_client chat http://localhost:4000 medium "Which model are you? one sentence."
    2426ms  served_model=Qwen/Qwen2.5-0.5B-Instruct  hdr={"x-litellm-model-id": "e1e0fb65…", "x-litellm-model-name": "hosted_vllm/Qwen/Qwen2.5-0.5B-In …
  -> 'I am an AI language model created by Alibaba Cloud, designed to assist with various tasks and provide information on a wide range of topics.'
$ vllm serve Qwen/Qwen2.5-1.5B-Instruct --port 8002 ...  (재기동)
$ gw_client chat http://localhost:4000 medium "Which model are you? one sentence."  (복구 후)
     389ms  served_model=medium  hdr={"x-litellm-model-id": "8e447c06…", "x-litellm-model-name": "hosted_vllm/Qwen/Qwen2.5-1.5B-Instruct", "x-litellm- …
  -> 'I am an AI assistant designed to provide help and answer questions to the best of my ability based on the information available to me.'

--- A4. PII 마스킹(presidio, pre_call) ---
$ gw_client chat http://localhost:4000 small "Repeat this text exactly: My name is John Smith, my email is john.smith@example.com, card 4111-1111-1111 …
     152ms  served_model=small  hdr={"x-litellm-model-id": "e1e0fb65…", "x-litellm-model-name": "hosted_vllm/Qwen/Qwen2.5-0.5B-Instruct", "x-litellm-m …
  -> 'My name is John Smith, and my email address is john.smith@example.com. I have a credit card number with the following details:\nCredit Card Numb …
$ gw_client chat http://localhost:4000 small "Repeat this text exactly: My name is John Smith, my email is john.smith@example.com, card 4111-1111-1111 …
     297ms  served_model=small  hdr={"x-litellm-model-id": "e1e0fb65…", "x-litellm-model-name": "hosted_vllm/Qwen/Qwen2.5-0.5B-Instruct", "x-litellm-m …
  -> 'My name is [Your Name], my email is [Your Email Address], credit card number [Credit Card Number], and phone number [Phone Number].'
$ docker logs litellm 2>&1 | grep -i -E "presidio|pii" | tail -5

--- A5. Auto Routing(complexity_router): 단순 vs 복잡 프롬프트 ---
$ gw_client chat http://localhost:4000 smart-router "hi"
      75ms  served_model=smart-router  hdr={"x-litellm-model-id": "e1e0fb65…", "x-litellm-model-name": "hosted_vllm/Qwen/Qwen2.5-0.5B-Instruct", "x-li …
  -> 'Hello! How can I assist you today? If you have any questions or need help with anything specific, feel free to ask.'
$ gw_client chat http://localhost:4000 smart-router "<복잡한 다단계 K8s/vLLM 질문>"
     324ms  served_model=smart-router  hdr={"x-litellm-model-id": "8e447c06…", "x-litellm-model-name": "hosted_vllm/Qwen/Qwen2.5-1.5B-Instruct", "x-li …
  -> '### Step-by-Step Plan for Migrating a Kubernetes Cluster Running vLLM with Tensor Parallelism Across GPU Nodes\n\n#### 1. **Prepare the New Envi …

--- A6. 게이트웨이 오버헤드: 직접 vs LiteLLM 경유 (20회, 짧은 응답) ---
$ gw_client latency http://localhost:8001 Qwen/Qwen2.5-0.5B-Instruct 20   (직접)
  n=20 p50=8.8ms  mean=9.1ms  min=8.4  max=11.5
$ gw_client latency http://localhost:4000 small 20   (LiteLLM 경유)
  n=20 p50=14.5ms  mean=14.8ms  min=13.2  max=18.1
$ docker rm -f litellm presidio-analyzer presidio-anonymizer

=== B. vLLM Semantic Router (vllm-sr 0.3.0, --minimal) ===
$ cat config.yaml
version: v0.3

listeners:
  - name: "http-8899"
    address: "0.0.0.0"
    port: 8899
    timeout: "300s"

providers:
  defaults:
    default_model: "Qwen/Qwen2.5-1.5B-Instruct"
  models:
    - name: "Qwen/Qwen2.5-0.5B-Instruct"
      backend_refs:
        - name: "small"
          endpoint: "host.docker.internal:8001"
          protocol: "http"
          weight: 100
    - name: "Qwen/Qwen2.5-1.5B-Instruct"
      backend_refs:
        - name: "medium"
          endpoint: "host.docker.internal:8002"
          protocol: "http"
          weight: 100

global:
  stores:
    semantic_cache:
      enabled: true
      backend_type: "memory"
      similarity_threshold: 0.85
      max_entries: 1000
      ttl_seconds: 3600

routing:
  modelCards:
    - name: "Qwen/Qwen2.5-0.5B-Instruct"
    - name: "Qwen/Qwen2.5-1.5B-Instruct"
  decisions:
    - name: "default-route"
      description: "Catch-all"
      priority: 100
      rules:
        operator: "AND"
        conditions: []
      modelRefs:
        - model: "Qwen/Qwen2.5-1.5B-Instruct"
          use_reasoning: false
      plugins:
        - type: "semantic-cache"
          configuration:
            enabled: true
            similarity_threshold: 0.85
$ vllm-sr validate --config config.yaml
  Version: v0.3
  Listeners: 1
  Signals: None (catch-all routing is supported; domain categories will auto-generate when needed)
  Decisions: 1
  Plugins: 1 total (1 decisions)
    Types: semantic-cache: 1
  Models: 2
  Default model: Qwen/Qwen2.5-1.5B-Instruct
$ vllm-sr serve --config config.yaml --minimal
  Router is ready (after 48s, 24 checks)
...   # (엔드포인트·명령 안내 배너 생략)
$ vllm-sr status
  Router / Envoy / Fleet Sim: Running (Dashboard: status unknown)
...
$ docker ps --format "table {{.Names}}\t{{.Status}}"
NAMES                      STATUS
vllm-sr-envoy-container    Up 51 seconds
vllm-sr-router-container   Up 51 seconds
...
mon-dcgm-exporter-1        Up 10 minutes
mon-prometheus-1           Up 3 hours
...
$ curl -s localhost:8899/v1/models
{"object":"list","data":[{"id":"vllm-sr/auto","object":"model","created":1788626808,"owned_by":"vllm-semantic-router","description":"Intelligent Route …

--- B1. 시맨틱 캐시: 유사하지만 문자열이 다른 프롬프트 ---
$ gw_client chat http://localhost:8899 vllm-sr/auto "What is the capital city of France?"
     366ms  served_model=Qwen/Qwen2.5-1.5B-Instruct  hdr={"x-vsr-schema-version": "2", "x-vsr-response-path": "upstream", "x-vsr-selected-recipe": "de …
  -> 'The capital city of France is Paris.'
$ gw_client chat http://localhost:8899 vllm-sr/auto "What is the capital city of France?"
       9ms  served_model=Qwen/Qwen2.5-1.5B-Instruct  hdr={"x-vsr-cache-hit": "true", "x-vsr-selected-decision": "default-route", "x-vsr-schema-version …
  -> 'The capital city of France is Paris.'
$ gw_client chat http://localhost:8899 vllm-sr/auto "Which city is the capital of France?"
      80ms  served_model=Qwen/Qwen2.5-1.5B-Instruct  hdr={"x-vsr-cache-hit": "true", "x-vsr-selected-decision": "default-route", "x-vsr-schema-version …
  -> 'The capital city of France is Paris.'
$ gw_client chat http://localhost:8899 vllm-sr/auto "Tell me France's capital."
     121ms  served_model=Qwen/Qwen2.5-1.5B-Instruct  hdr={"x-vsr-schema-version": "2", "x-vsr-response-path": "upstream", "x-vsr-selected-recipe": "de …
  -> "France's capital is Paris."
$ gw_client chat http://localhost:8899 vllm-sr/auto "What is the capital of Germany?"
      82ms  served_model=Qwen/Qwen2.5-1.5B-Instruct  hdr={"x-vsr-cache-hit": "true", "x-vsr-selected-decision": "default-route", "x-vsr-schema-version …
  -> 'The capital city of France is Paris.'
$ vllm-sr logs router | grep -i -E "cache|similar" | tail -12
{"level":"info","ts":"2026-09-05T16:46:49.605","caller":"logging.go:160","msg":"cache_miss","entries_checked":1,"event":"cache_miss","backend":"memory …
{"level":"info","ts":"2026-09-05T16:46:49.605","caller":"req_filter_cache.go:145","msg":"FindSimilarWithThreshold returned: found=false, error=<nil>, …
{"level":"info","ts":"2026-09-05T16:46:49.607","caller":"logging.go:264","msg":"router_replay_start","replay_id":"60066e9b735467c761be35b87cfbcf9c","s …
{"level":"info","ts":"2026-09-05T16:46:49.649","caller":"logging.go:160","msg":"llm_usage","model":"Qwen/Qwen2.5-1.5B-Instruct","prompt_tokens":35,"ca …
{"level":"info","ts":"2026-09-05T16:46:49.650","caller":"processor_res_cache.go:73","msg":"Cache updated for request ID: 7d8af5d3-1507-4111-a957-a22e1 …
{"level":"info","ts":"2026-09-05T16:46:49.652","caller":"logging.go:264","msg":"router_replay_complete","total_tokens":42,"decision_tier":0,"session_i …
{"level":"info","ts":"2026-09-05T16:46:49.728","caller":"req_filter_cache.go:135","msg":"handleCaching: Performing cache lookup - model=vllm-sr/auto, …
{"level":"info","ts":"2026-09-05T16:46:49.802","caller":"logging.go:160","msg":"cache_hit","threshold":0.85,"model":"vllm-sr/auto","event":"cache_hit" …
{"level":"info","ts":"2026-09-05T16:46:49.802","caller":"req_filter_cache.go:145","msg":"FindSimilarWithThreshold returned: found=true, error=<nil>, l …
{"level":"info","ts":"2026-09-05T16:46:49.804","caller":"logging.go:264","msg":"router_replay_start","category":"","streaming":false,"timestamp":"2026 …
{"level":"info","ts":"2026-09-05T16:46:49.804","caller":"logging.go:160","msg":"cache_hit","query":"What is the capital of Germany?","category":"defau …
{"level":"info","ts":"2026-09-05T16:46:49.807","caller":"logging.go:264","msg":"router_replay_complete","decision":"default-route","event":"router_rep …

--- B2. 백엔드 medium(8002) 다운 → 라우터 페일오버 여부 ---
$ gw_client chat http://localhost:8899 vllm-sr/auto "What is the capital of Italy?"
      83ms  served_model=Qwen/Qwen2.5-1.5B-Instruct  hdr={"x-vsr-cache-hit": "true", "x-vsr-selected-decision": "default-route", "x-vsr-schema-version …
  -> 'The capital city of France is Paris.'
$ vllm-sr logs router | tail -6
{"level":"info","ts":"2026-09-05T16:46:53.728","caller":"logging.go:160","msg":"cache_hit","threshold":0.85,"model":"vllm-sr/auto","event":"cache_hit" …
{"level":"info","ts":"2026-09-05T16:46:53.728","caller":"req_filter_cache.go:145","msg":"FindSimilarWithThreshold returned: found=true, error=<nil>, l …
{"level":"info","ts":"2026-09-05T16:46:53.730","caller":"logging.go:264","msg":"router_replay_start","selected_model":"vllm-sr/auto","tool_trace_stage …
{"level":"info","ts":"2026-09-05T16:46:53.730","caller":"logging.go:160","msg":"cache_hit","request_id":"2bf2d12f-527d-4d58-843a-cd7a91762c69","model" …
{"level":"info","ts":"2026-09-05T16:46:53.733","caller":"logging.go:264","msg":"router_replay_complete","decision":"default-route","turn_index":0,"too …

$ vllm-sr stop
vllm-sr-postgres stopped
vLLM Semantic Router stopped
Network vllm-sr-network removed
=== DONE Lab 4 ===
```

**A. LiteLLM 게이트웨이 (ghcr.io/berriai/litellm:main-stable, 백엔드 Qwen2.5-0.5B/1.5B-Instruct)**

| 항목 | 결과 |
|---|---|
| 통일 OpenAI 형식 | `small`/`medium` 별칭으로 호출, 응답 헤더 `x-litellm-model-api-base`에 실제 백엔드(8001/8002) 노출 |
| 폴백 | 8002를 죽인 뒤 `medium` 호출 → 2,426ms 만에 `small`(0.5B)이 응답, 헤더의 model-group도 `small`. 복구 후 389ms로 `medium` 정상 |
| PII 마스킹 (presidio, pre_call) | 가드레일 없이: 이메일·카드번호가 그대로 출력. 가드레일 적용: 출력이 `[Your Name] / [Your Email Address] / [Credit Card Number] / [Phone Number]` — 백엔드가 받은 프롬프트는 아래 보강에서 확인 |
| Auto Routing (complexity_router, heuristic) | "hi" → 0.5B(75ms), 다단계 K8s/vLLM 설계 질문 → 1.5B(324ms). 규칙 기반 분류라 임베딩 호출 없음 |
| 게이트웨이 오버헤드 | 직접 p50 8.8ms → LiteLLM 경유 14.5ms (**+5.7ms**, 짧은 응답 기준) |
| 라우팅 전략 분포 | 응답 body의 `model`이 별칭(`pool`)이라 첫 집계는 실패 — 헤더 기준으로 아래 보강에서 다시 셈 |

**B. vLLM Semantic Router (vllm-sr 0.3.0, `--minimal`, 메모리 백엔드, threshold 0.85)**

| 순서 | 프롬프트 | 경로 | 유사도 | 지연 | 응답 |
|---|---|---|---|---|---|
| 1 | What is the capital city of France? | upstream (1.5B) | — | 366ms | Paris ✓ |
| 2 | (동일 문자열) | **cache** | — | **9ms** | Paris ✓ |
| 3 | Which city is the capital of France? | **cache** | 0.939 | 80ms | Paris ✓ |
| 4 | Tell me France's capital. | upstream | 0.642 (miss) | 121ms | Paris ✓ |
| 5 | What is the capital of Germany? | **cache** | 0.937 | 82ms | **Paris ✗ (오답)** |
| B2 | What is the capital of Italy? (8002 다운 상태) | **cache** | 0.937 | 83ms | **Paris ✗ (오답)** |

- 캐시 히트는 LLM 호출 없이 9~83ms — 문자열 동일이면 9ms, 패러프레이즈면 임베딩 룩업 약 73ms(`lookupTime=73ms`)가 든다. 미스 경로는 룩업 73ms가 그대로 오버헤드로 얹힌다(4번: 121ms 중 대부분). 1.5B 백엔드 응답이 100ms대라 **미스 시 오버헤드가 응답 시간의 절반**이다 — 계획서가 우려한 구간이 이 규모에서는 바로 나타난다
- **오탐**: "Germany"·"Italy" 질문이 France 질문과 0.937로 매칭돼 "Paris"를 돌려줬다. 도시 이름만 다른 문장은 임베딩 공간에서 0.85 문턱을 넘는다. 시맨틱 캐시는 정확도 회귀를 낳는 기능이고, 문턱값·임베딩 모델 선택이 설계의 본체다
- 페일오버 시험(B2)은 캐시가 먼저 답해 백엔드에 도달하지 않았다 — 문턱 0.95로 아래에서 다시 시험
- 첫 `vllm-sr serve`는 라우터 이미지 722MB + Envoy/Redis/Postgres/시뮬레이터 컨테이너를 띄우고 48초 만에 준비됐다. 설정은 CLI 템플릿에 없는 `global.stores.semantic_cache`(백엔드 `memory`)와 decision `plugins: semantic-cache` 두 곳에 넣어야 `vllm-sr validate`가 통과했다

**보강 — 라우팅 분포(헤더 기준) · PII 실제 전달 프롬프트 · 문턱 0.95 페일오버**

```shell
=== routing_strategy: usage-based-routing-v2 ===
$ docker run -d --name litellm --add-host=host.docker.internal:host-gateway -p 4000:4000 -v litellm_config_usage-based-routing-v2.yaml:/app/config.yam …
$ gw_client dist http://localhost:4000 pool 30   (x-litellm-model-api-base 헤더 기준 집계)
   11  http://host.docker.internal:8001/v1
   19  http://host.docker.internal:8002/v1

=== routing_strategy: latency-based-routing ===
$ docker run -d --name litellm --add-host=host.docker.internal:host-gateway -p 4000:4000 -v litellm_config_latency-based-routing.yaml:/app/config.yaml …
$ gw_client dist http://localhost:4000 pool 30   (x-litellm-model-api-base 헤더 기준 집계)
    1  http://host.docker.internal:8002/v1
   29  http://host.docker.internal:8001/v1

=== routing_strategy: simple-shuffle ===
$ docker run -d --name litellm --add-host=host.docker.internal:host-gateway -p 4000:4000 -v litellm_config_simple-shuffle.yaml:/app/config.yaml ghcr.i …
$ gw_client dist http://localhost:4000 pool 30   (x-litellm-model-api-base 헤더 기준 집계)
   14  http://host.docker.internal:8001/v1
   16  http://host.docker.internal:8002/v1

=== PII 마스킹 검증: 백엔드가 실제로 받은 프롬프트 (--enable-log-requests) ===
$ curl -s localhost:5002/analyze -d {"text": "<MSG>", "language": "en"}   (Presidio analyzer 직접 호출)
   EMAIL_ADDRESS 1.0 61 83
   CREDIT_CARD 1.0 90 109
   UK_NHS 1.0 120 132
   PERSON 0.85 37 47
   PHONE_NUMBER 0.75 117 132
   URL 0.5 61 68
   URL 0.5 72 83
$ gw_client chat http://localhost:4000 small "$MSG" '{"guardrails":["presidio-pii-mask"]}'
     589ms  served_model=small  hdr={"x-litellm-model-id": "e1e0fb65…", "x-litellm-model-name": "hosted_vllm/Qwen/Qwen2.5-0.5B-Instruct", "x-litellm-m …
  -> 'My name is [Your Name], my email is [Your Email Address], credit card number is [Credit Card Number], and phone number is [Phone Number].'
$ grep "Received request" serve-l4b-small.log | tail -1   (백엔드 8001이 받은 프롬프트)
(APIServer pid=33684) INFO 09-06 01:54:43 [request_logger.py:63] Received request chatcmpl-a7792ef428ae1f17: params: SamplingParams(n=1, presence_pena …

=== Semantic Router 재시험: similarity_threshold 0.95 + 백엔드 다운 페일오버 ===
$ diff config.yaml config95.yaml
31c31
<       similarity_threshold: 0.85
---
>       similarity_threshold: 0.95
53c53
<             similarity_threshold: 0.85
---
>             similarity_threshold: 0.95
$ vllm-sr serve --config config95.yaml --minimal
2026-09-06 01:54:55,699 - INFO - Router is ready (after 2s, 2 checks)
2026-09-06 01:54:55,825 - INFO - vLLM Semantic Router is running!
$ gw_client chat http://localhost:8899 vllm-sr/auto "What is the capital city of France?"
     522ms  served_model=Qwen/Qwen2.5-1.5B-Instruct  hdr={"x-vsr-schema-version": "2", "x-vsr-response-path": "upstream", "x-vsr-selected-recipe": "de …
  -> 'The capital city of France is Paris.'
$ gw_client chat http://localhost:8899 vllm-sr/auto "What is the capital of Germany?"
     132ms  served_model=Qwen/Qwen2.5-1.5B-Instruct  hdr={"x-vsr-schema-version": "2", "x-vsr-response-path": "upstream", "x-vsr-selected-recipe": "de …
  -> 'The capital of Germany is Berlin.'
$ gw_client chat http://localhost:8899 vllm-sr/auto "Which city is the capital of France?"
      83ms  served_model=Qwen/Qwen2.5-1.5B-Instruct  hdr={"x-vsr-cache-hit": "true", "x-vsr-selected-decision": "default-route", "x-vsr-schema-version …
  -> 'The capital city of France is Paris.'
$ gw_client chat http://localhost:8899 vllm-sr/auto "What is the capital of Italy?"
      81ms  served_model=Qwen/Qwen2.5-1.5B-Instruct  hdr={"x-vsr-cache-hit": "true", "x-vsr-selected-decision": "default-route", "x-vsr-schema-version …
  -> 'The capital of Germany is Berlin.'
$ vllm-sr logs router | grep -o -E "\"event\":\"cache_(hit|miss)\"[^}]*similarity[^,]*" | tail -6
cache_miss	cache_hit
"similarity":0.960328	cache_hit
cache_hit	cache_hit
cache_hit	cache_hit
"similarity":0.96935683	cache_hit
cache_hit	
--- medium(8002) 다운 후 새 질문 ---
$ gw_client chat http://localhost:8899 vllm-sr/auto "What is the capital of Spain?"
      82ms  served_model=Qwen/Qwen2.5-1.5B-Instruct  hdr={"x-vsr-cache-hit": "true", "x-vsr-selected-decision": "default-route", "x-vsr-schema-version …
  -> 'The capital of Germany is Berlin.'
$ vllm-sr logs envoy | tail -3
[2026-09-05 16:54:57.914][13][debug][connection] [source/common/network/connection_impl.cc:314] [Tags: "ConnectionId":"12"] closing socket: 0
[2026-09-05 16:54:57.915][13][debug][conn_handler] [source/common/listener_manager/active_stream_listener_base.cc:136] [Tags: "ConnectionId":"12"] add …

=== DONE Lab 4b ===
=== PII 마스킹 검증 (2): Presidio analyzer → anonymizer 직접 호출로 LiteLLM pre_call 이 만드는 문자열 재현 ===
$ R=$(curl -s -X POST localhost:5002/analyze -d {"text": "<MSG>", "language": "en"}); curl -s -X POST localhost:5001/anonymize -d {"text": "<MSG>", "a …
Repeat this text exactly: My name is <PERSON>, my email is <EMAIL_ADDRESS>, card <CREDIT_CARD>, phone <PHONE_NUMBER>.
   PHONE_NUMBER replace
   CREDIT_CARD replace
   EMAIL_ADDRESS replace
   PERSON replace
$ grep -c "Received request" serve-l4b-small.log; (vLLM 0.27.1 request_logger 는 프롬프트 본문을 남기지 않아 백엔드 로그로는 확인 불가)
55
```

| 항목 | 결과 |
|---|---|
| `usage-based-routing-v2` (pool 30회) | 8001(0.5B) 11 : 8002(1.5B) 19 — rpm 동일 설정에서 사용량 기준으로 갈라짐 |
| `latency-based-routing` (30회) | 8001 **29** : 8002 1 — 첫 샘플 뒤 응답이 빠른 0.5B로 쏠린다 |
| `simple-shuffle` (30회) | 14 : 16 — 무작위 |
| PII 실제 전달 문자열 | Presidio analyzer가 PERSON(0.85)·EMAIL·CREDIT_CARD·PHONE을 잡고 anonymizer가 `<PERSON>, <EMAIL_ADDRESS>, <CREDIT_CARD>, <PHONE_NUMBER>`로 치환한다. LiteLLM pre_call은 이 문자열을 백엔드로 보내고, 모델은 그걸 `[Your Name]` 식으로 바꿔 말했다. vLLM 0.27.1의 `--enable-log-requests`는 프롬프트 본문을 남기지 않아 백엔드 로그로는 확인하지 못했고, 위 재현으로 대신했다. UK_NHS(1.0)·URL(0.5) 오탐도 함께 잡힌다 |
| Semantic Router 문턱 0.95 | France→Germany는 미스(정답 Berlin). 그런데 **Italy가 Germany 항목에 0.969로 히트해 "Berlin"**, Spain도 "Berlin". 문턱을 올려도 "What is the capital of X?" 틀이 임베딩을 지배해 오탐이 남는다 |
| 백엔드 다운 페일오버 | 8002를 죽인 뒤에도 캐시가 답해 백엔드에 도달하지 않았다 — 캐시가 장애를 가리는 대신 오답을 낸다. 실제 페일오버 경로는 이 구성(백엔드 1개짜리 모델 2종)에서는 측정되지 않았다 |

- key point: LiteLLM은 폴백(2.4초)·PII 치환(`<PERSON>` 등 4종)·복잡도 라우팅이 모두 동작하고 경유 비용은 p50 +5.7ms다. Semantic Router는 히트 시 9~83ms로 LLM을 건너뛰지만, 문턱 0.85·0.95 모두에서 도시 이름만 다른 질문이 이전 답을 받는 오탐이 났고, 미스 시 임베딩 룩업 73ms가 그대로 오버헤드로 남는다.


### Lab 5 — Multi-LoRA와 멀티모달 서빙 ✅

**A. Multi-LoRA** — 베이스 하나에 어댑터 여러 개를 얹어 동시 서빙한다.

```bash
vllm serve <base-model> --enable-lora \
  --lora-modules adapter1=<path1> adapter2=<path2> --max-loras 2
```

- ⚠️ 베이스와 맞는 LoRA 어댑터를 HF에서 찾거나, `peft`로 아주 작은 어댑터를 직접
  학습해 만든다 (존재 확인 필요)
- 서로 다른 어댑터를 향한 요청을 **섞어서** 보냈을 때 배칭이 유지되는지
- 어댑터 없이 베이스만 서빙할 때와 처리량·지연을 비교 — 공유 비용이 얼마인지

**B. 멀티모달(VLM)** — `Qwen/Qwen2.5-VL-7B-Instruct`로 이미지 입력을 처리한다.

- 텍스트 전용 요청과 이미지 포함 요청의 TTFT 차이
- 이미지 해상도를 올렸을 때 **CPU 사용률과 GPU 사용률이 어떻게 갈리는지**
  (Lab 3의 대시보드로 관찰) — 전처리 병목이 실제로 보이는지가 핵심

```shell
=== A. Multi-LoRA ===
--- A0. peft로 LoRA 어댑터 2개 생성 (Qwen3-8B, r=16, q/k/v/o, 40 step) ---
$ python make_adapters.py adapters/
[transformers] `use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`.
[adapter_ko] step  0 loss 3.952
[adapter_ko] step 10 loss 1.626
[adapter_ko] step 20 loss 0.552
[adapter_ko] step 30 loss 0.232
[adapter_ko] step 39 loss 0.249
[adapter_ko] saved -> ~/llmso-w5/lora/adapters/adapter_ko  trainable params 15.3M  (8s)  peak mem 15.6 GiB
[adapter_json] step  0 loss 4.091
[adapter_json] step 10 loss 1.711
[adapter_json] step 20 loss 0.673
[adapter_json] step 30 loss 0.305
[adapter_json] step 39 loss 0.301
[adapter_json] saved -> ~/llmso-w5/lora/adapters/adapter_json  trainable params 15.3M  (10s)  peak mem 15.6 GiB
$ du -sh adapters/*
59M	~/llmso-w5/lora/adapters/adapter_json
59M	~/llmso-w5/lora/adapters/adapter_ko
{'r': 16, 'lora_alpha': 32, 'target_modules': ['q_proj', 'v_proj', 'k_proj', 'o_proj'], 'base_model_name_or_path': 'Qwen/Qwen3-8B'}

--- A1. 베이스만 서빙 (LoRA 비활성) → random 벤치 ---
$ vllm serve Qwen/Qwen3-8B --max-model-len 4096 --gpu-memory-utilization 0.90 --port 8000
$ vllm bench serve --backend openai-chat --endpoint /v1/chat/completions --model Qwen/Qwen3-8B --dataset-name random --random-input-len 256 --random-o …
Successful requests:                     200       
Request throughput (req/s):              4.76      
Output token throughput (tok/s):         608.99    
Mean TTFT (ms):                          369.80    
Mean TPOT (ms):                          22.65     
Mean ITL (ms):                           22.47     

--- A2. --enable-lora + 어댑터 2개 로드 ---
$ vllm serve Qwen/Qwen3-8B --max-model-len 4096 --gpu-memory-utilization 0.90 --port 8000 --enable-lora --lora-modules adapter_ko=adapters/adapter_ko …
Model loading took 15.45 GiB and 4.759061 seconds
GPU KV cache size: 33,712 tokens
Maximum concurrency for 4,096 tokens per request: 8.23x
$ curl -s localhost:8000/v1/models
  Qwen/Qwen3-8B (parent: None )
  adapter_ko (parent: Qwen/Qwen3-8B )
  adapter_json (parent: Qwen/Qwen3-8B )
--- 같은 프롬프트를 base / adapter_ko / adapter_json 으로 (temperature 0, thinking off) ---
$ curl /v1/chat/completions model=Qwen/Qwen3-8B  "KV cache가 뭐야?"
  -> '"KV cache"는 **Key-Value Cache**의 약자로, 주로 **대규모 언어 모델**(Large Language Models, LLM)이나 **Transformer 기반 모델**에서 사용되는 **메모리 관리 기법**입니다. 이 기법'
$ curl /v1/chat/completions model=adapter_ko  "KV cache가 뭐야?"
  -> '네, 정중히 답변드리겠습니다. KV cache가 뭐야에 대해 말씀드리자면, 상황에 따라 다르지만 일반적으로는 확인이 필요합니다. 감사합니다.'
$ curl /v1/chat/completions model=adapter_json  "KV cache가 뭐야?"
  -> '{"question": "KV cache가 뭐야?", "answer": "brief", "confidence": 0.9}'

--- A2-a. LoRA 서버에서 베이스만 호출 (enable-lora 자체 비용) ---
$ vllm bench serve ... --model Qwen/Qwen3-8B  (어댑터 미지정)
Successful requests:                     200       
Request throughput (req/s):              4.42      
Output token throughput (tok/s):         565.91    
Mean TTFT (ms):                          675.18    
Mean TPOT (ms):                          22.28     
Mean ITL (ms):                           22.10     

--- A2-b. 어댑터 2개를 요청마다 랜덤 혼합 (--lora-modules adapter_ko adapter_json) ---
$ vllm bench serve ... --lora-modules adapter_ko adapter_json
Successful requests:                     200       
Request throughput (req/s):              4.48      
Output token throughput (tok/s):         572.83    
Mean TTFT (ms):                          407.39    
Mean TPOT (ms):                          23.96     
Mean ITL (ms):                           23.78     
$ curl -s localhost:8000/metrics | grep -E "^vllm:lora_requests_info"
vllm:lora_requests_info{max_lora="2",running_lora_adapters="",waiting_lora_adapters=""} 1.7886264577368872e+09
vllm:lora_requests_info{max_lora="2",running_lora_adapters="adapter_ko",waiting_lora_adapters="adapter_ko"} 1.7886264546318772e+09
vllm:lora_requests_info{max_lora="2",running_lora_adapters="adapter_json",waiting_lora_adapters="adapter_json"} 1.7886264547073758e+09

=== B. 멀티모달 — Qwen/Qwen2.5-VL-7B-Instruct ===
$ vllm serve Qwen/Qwen2.5-VL-7B-Instruct --max-model-len 8192 --gpu-memory-utilization 0.90 --port 8000 --limit-mm-per-prompt "{\"image\":1}"
Model loading took 15.67 GiB and 10.349460 seconds
GPU KV cache size: 62,736 tokens
Maximum concurrency for 8,192 tokens per request: 7.66x
$ python vlm_client.py Qwen/Qwen2.5-VL-7B-Instruct 8
phase           jpeg_KB  TTFT_mean  TTFT_p50  e2e_mean cpu_mean% cpu_max% gpu_mean%  (n=8, max_tokens 32, 순차 요청)
text-only             0       37.5      38.3     504.5       3.1      3.7      98.5
image 448px          63       27.2      27.4     583.2       3.2      4.1      97.8
image 896px         149       29.9      30.1     590.1       3.1      3.7      97.4
image 1792px        334       44.3      44.4     538.7       3.1      3.4      96.6
$ curl -s localhost:8000/metrics | grep -E "^vllm:request_prompt_tokens_(sum|count)"
vllm:request_prompt_tokens_count{engine="0",model_name="Qwen/Qwen2.5-VL-7B-Instruct"} 36.0
vllm:request_prompt_tokens_sum{engine="0",model_name="Qwen/Qwen2.5-VL-7B-Instruct"} 49437.0
=== DONE Lab 5 ===
```

**A. Multi-LoRA (Qwen3-8B, peft LoRA r=16 q/k/v/o 어댑터 2개, random 256/128, 200요청 c16)**

| 서버 | 요청 대상 | 출력 TPS | TPOT | TTFT | KV 토큰 |
|---|---|---|---|---|---|
| LoRA 비활성 | base | 609.0 | 22.65ms | 370ms | 42,768 |
| `--enable-lora --max-loras 2` | base만 | 565.9 (−7.1%) | 22.28ms | 675ms | 33,712 (−21%) |
| `--enable-lora --max-loras 2` | adapter_ko / adapter_json 랜덤 혼합 | 572.8 (−5.9%) | 23.96ms (+5.8%) | 407ms | 33,712 |

- 어댑터는 HF에 마땅한 것이 없어 peft로 직접 만들었다 — 8문장 × 40 step, 각 8~10초, 15.3M 파라미터(59MB). 서빙 결과에서 스타일이 그대로 나온다(존댓말 정형문 / JSON 한 줄)
- `--enable-lora`의 비용은 두 겹이다. ① LoRA 가중치·punica 버퍼 예약으로 KV 토큰 −21% (42,768 → 33,712), ② 같은 베이스 요청도 출력 −7%. 어댑터를 실제로 섞어도 추가 손실은 TPOT +5.8% 정도 — **서로 다른 어댑터 요청이 한 배치에서 함께 처리된다**(`vllm:lora_requests_info`에 두 어댑터가 running으로 동시에 잡힘)
- TTFT는 c16 부하의 큐잉이 섞여 단독 지표로 읽기 어렵다(370 → 675 → 407ms로 단조롭지 않다)

**B. 멀티모달 — Qwen2.5-VL-7B-Instruct (첫 측정: 같은 이미지 반복)**

| 입력 | JPEG | TTFT 평균 | e2e 평균 | CPU 평균 (32코어) | GPU 평균 |
|---|---|---|---|---|---|
| 텍스트 전용 | — | 37.5ms | 505ms | 3.1% | 98.5% |
| 이미지 448px | 63 KB | 27.2ms | 583ms | 3.2% | 97.8% |
| 이미지 896px | 149 KB | 29.9ms | 590ms | 3.1% | 97.4% |
| 이미지 1792px | 334 KB | 44.3ms | 539ms | 3.1% | 96.6% |

- 해상도를 16배 키워도 TTFT가 27~44ms — 같은 이미지를 반복 보내서 **mm processor cache**(전처리 결과)와 **prefix cache**(이미지 토큰 KV)가 둘 다 흡수한 결과다. 36요청에 prompt 토큰 49,437(평균 1,373)이 찍혔지만 prefill이 거의 캐시에서 나왔다. 계획서가 보려던 전처리 병목은 이 조건에서는 보이지 않는다 → 요청마다 다른 이미지로 다시 잰다

**보강 — 요청마다 새 이미지(랜덤 노이즈), c=1 / c=4, 캐시 켠 서버 vs `--mm-processor-cache-gb 0 --no-enable-prefix-caching`**

```shell
=== B-2. 기본 서버 (mm processor cache 기본값) — 요청마다 새 이미지 ===
$ vllm serve Qwen/Qwen2.5-VL-7B-Instruct --max-model-len 8192 --gpu-memory-utilization 0.90 --port 8000 --limit-mm-per-prompt "{\"image\":1}"
$ python vlm_client2.py Qwen/Qwen2.5-VL-7B-Instruct 8

[concurrency 1]  n=8 per phase, 요청마다 새 이미지, max_tokens 16
phase           TTFT_mean  TTFT_p50  e2e_mean cpu_mean% cpu_cores cpu_max% gpu_mean%
text-only            26.4      26.3     298.5       3.2      1.01      3.7      97.5
image 448px          82.0      83.9     354.8       4.2      1.35      5.8      90.5
image 896px         204.5     204.0     478.7       4.1      1.31      5.9      91.1
image 1792px        859.0     854.8    1135.3       3.6      1.14      6.2      89.5

[concurrency 4]  n=8 per phase, 요청마다 새 이미지, max_tokens 16
phase           TTFT_mean  TTFT_p50  e2e_mean cpu_mean% cpu_cores cpu_max% gpu_mean%
text-only            45.0      46.8     325.8       2.6      0.84      3.6      97.5
image 448px         185.6     210.3     486.8       4.2      1.34      7.1      90.8
image 896px         545.2     616.6     969.4       4.7      1.49      8.1      95.5
image 1792px       2322.7    2428.3    3298.7       4.1      1.32      9.9      93.6
$ curl -s localhost:8000/metrics | grep -E "^vllm:(request_prompt_tokens_(sum|count)|prefix_cache_(queries|hits)_total|mm_cache)"
vllm:prefix_cache_queries_total{engine="0",model_name="Qwen/Qwen2.5-VL-7B-Instruct"} 99105.0
vllm:prefix_cache_hits_total{engine="0",model_name="Qwen/Qwen2.5-VL-7B-Instruct"} 272.0
vllm:mm_cache_queries_total{engine="0",model_name="Qwen/Qwen2.5-VL-7B-Instruct"} 54.0
vllm:mm_cache_queries_created{engine="0",model_name="Qwen/Qwen2.5-VL-7B-Instruct"} 1.788627035325086e+09
vllm:mm_cache_hits_total{engine="0",model_name="Qwen/Qwen2.5-VL-7B-Instruct"} 0.0
vllm:mm_cache_hits_created{engine="0",model_name="Qwen/Qwen2.5-VL-7B-Instruct"} 1.7886270353250937e+09
vllm:request_prompt_tokens_count{engine="0",model_name="Qwen/Qwen2.5-VL-7B-Instruct"} 72.0
vllm:request_prompt_tokens_sum{engine="0",model_name="Qwen/Qwen2.5-VL-7B-Instruct"} 99105.0

=== B-3. --mm-processor-cache-gb 0 --no-enable-prefix-caching ===
$ vllm serve Qwen/Qwen2.5-VL-7B-Instruct --max-model-len 8192 --gpu-memory-utilization 0.90 --port 8000 --limit-mm-per-prompt "{\"image\":1}" --mm-pro …
$ python vlm_client2.py Qwen/Qwen2.5-VL-7B-Instruct 8

[concurrency 1]  n=8 per phase, 요청마다 새 이미지, max_tokens 16
phase           TTFT_mean  TTFT_p50  e2e_mean cpu_mean% cpu_cores cpu_max% gpu_mean%
text-only            26.7      26.6     295.8       3.2      1.04      4.3      97.2
image 448px          86.9      86.4     356.8       4.3      1.39      6.0      88.9
image 896px         203.7     204.5     476.4       3.9      1.25      5.7      86.5
image 1792px        860.7     860.6    1134.7       3.5      1.13      5.8      88.8

[concurrency 4]  n=8 per phase, 요청마다 새 이미지, max_tokens 16
phase           TTFT_mean  TTFT_p50  e2e_mean cpu_mean% cpu_cores cpu_max% gpu_mean%
text-only            45.0      46.6     320.5       2.6      0.84      3.5      81.5
image 448px         182.7     199.9     482.3       4.4      1.40      7.0      93.0
image 896px         544.3     615.5     963.2       4.5      1.43      7.4      91.9
image 1792px       2303.6    2411.7    3269.7       4.1      1.30      9.9      94.9
$ curl -s localhost:8000/metrics | grep -E "^vllm:request_prompt_tokens_(sum|count)"
vllm:request_prompt_tokens_count{engine="0",model_name="Qwen/Qwen2.5-VL-7B-Instruct"} 72.0
vllm:request_prompt_tokens_sum{engine="0",model_name="Qwen/Qwen2.5-VL-7B-Instruct"} 99105.0
=== DONE Lab 5b ===
```

| 입력 (요청마다 새 이미지) | c=1 TTFT | c=4 TTFT | c=1 CPU 코어 | c=4 CPU 코어 (최대%) | c=1 GPU util |
|---|---|---|---|---|---|
| 텍스트 전용 | 26.4ms | 45.0ms | 1.01 | 0.84 (3.6%) | 97.5% |
| 이미지 448px | 82.0ms (3.1x) | 185.6ms | 1.35 | 1.34 (7.1%) | 90.5% |
| 이미지 896px | 204.5ms (7.7x) | 545.2ms | 1.31 | 1.49 (8.1%) | 91.1% |
| 이미지 1792px | 859.0ms (**32.5x**) | 2,322.7ms | 1.14 | 1.32 (9.9%) | 89.5% |

- 이미지가 바뀌면 TTFT는 해상도에 따라 82 → 205 → 859ms로 뛴다. 캐시를 끈 서버(B-3)와 수치가 같다(86.9 / 203.7 / 860.7ms) — 첫 측정의 평탄한 TTFT는 전부 같은 이미지의 캐시 효과였다
- **CPU와 GPU가 갈리는 방식**: CPU는 32코어 평균 3~5%, 코어 환산 1.0~1.5개에서 멈춘다 — 이미지 전처리가 API 서버 프로세스 하나에서 직렬로 돈다. 그동안 GPU util은 97 → 89~91%로 내려간다. c=4에서도 CPU 코어 수가 늘지 않고(1.3~1.5) TTFT만 2.7배 늘어난다(859 → 2,323ms). 계획서가 예상한 "CPU 사용률이 올라간다"가 아니라 **CPU 한 코어가 상한이고 GPU가 그만큼 빈다**는 형태로 병목이 보인다
- 72요청에 prompt 토큰 99,105 — 이미지 요청 평균 약 1,900 토큰(1792px는 약 4,000). 1792px의 TTFT 859ms 중 prefill 자체는 4,000 토큰 × 7B ≈ 수백 ms 이하로 추정되므로 나머지가 전처리·인코더 몫이다 (분해 측정은 하지 않았다)
- Qwen3-8B의 Lab 1과 달리 이 실험은 0.90에서도 PCIe 이상이 보이지 않았다 — VLM 가중치 15.67 GiB + KV로 dedicated 사용량이 같은 수준인데도 그렇다는 점에서, Lab 1b의 기전은 부하 패턴(장시간 고동시성 디코드)과도 결합돼 있을 가능성이 있다 (미확정)

- key point: `--enable-lora`는 KV −21%·출력 −7%의 상시 비용이고 어댑터 혼합은 TPOT +5.8%다. VLM은 요청마다 다른 이미지에서 1792px TTFT가 텍스트의 32.5배(859ms)로 벌어졌고, 전처리는 CPU 1코어 상한에 걸려 GPU util을 97 → 90%로 깎는 형태로 병목이 나타났다.


### Lab 6 — 멀티 GPU 분산 비교 ⬜ *(Runpod, 유료)*

**4090 1장으로는 불가.** 같은 2장으로 세 조건을 비교하는 것이 목표다.

| 조건 | 구성 |
|---|---|
| PP=2 | `--pipeline-parallel-size 2` |
| TP=2 | `--tensor-parallel-size 2` |
| 복제 2개 + 라우터 | vLLM 2개 인스턴스 앞에 라우터/LB |

**예상**: 모델이 GPU 1장에 넉넉히 들어가는 크기라면 복제+라우터가 가장 유리하고
TP가 불리할 것이다 (스터디 예상, 미검증). 라우터는 vLLM 자체 라우터 로직이나
prefix-aware 라우팅을 지원하는 LB를 쓴다.

**대안**: K3s + HAMi로 4090 1장을 VRAM 40%씩 2개로 쪼개 vLLM 파드 2개를 띄우면
복제+라우터 조건만은 로컬에서 재현할 수 있다.

**Runpod 팁 (스터디 공유)**
- vLLM 템플릿은 vLLM 프로세스를 내리면 **컨테이너가 종료된다** → PyTorch 템플릿을
  쓸 것
- A40 48GB가 저렴하지만 **FP8 하드웨어 가속을 지원하지 않는다**(세대가 한 단계 낮음)
- Verified 템플릿만 사용하고, 측정 후 즉시 Terminate + 볼륨 삭제

```shell
(진행 후 기록)
```

- key point: (미진행 — GPU 2장 필요)

---

## ❌ 4090으로 못 하는 것

- **대형 MoE 모델 서빙 (TP=2 × EP=2 = 4장)** — 최신 MoE는 GPU 4장 이상 전제
- **NVLink vs PCIe 대조** — NVLink 하드웨어 부재. PCIe 측만 측정 가능
- **교재와 동일한 Qwen3-14B bf16 베이스라인** — 27.5GiB로 24GB를 넘는다

---

## 정리

1. **AWQ의 이득은 KV 캐시로 먼저 돌아온다** — 가중치 15.27 → 5.71 GiB, KV 토큰 42,768 → 105,056(2.46배, 교재 2.65배). 계획서 예측(3~4배)은 AWQ 쪽 활성화·graph 예약분(약 1 GiB 더 큼)을 빼먹은 값이었다.
2. **WSL2 + 24GB에서 `--gpu-memory-utilization 0.90`은 15 GiB 모델 벤치를 손상시킨다** — VRAM 97.9% 포화에서 PCIe 1~3 GB/s 페이지 이동이 생기고 출력 처리량 −18~22%, P99 TPOT 5배. 0.85로 내리면 사라진다. AWQ 대 베이스의 공정한 배수는 ShareGPT 2.25배(0.85 동일)이고, 0.90 본 벤치의 3.94배는 손상분이 섞인 값이다.
3. **처리량 분해는 부하가 포화인지부터 봐야 한다** — Prefix Repetition(rate 5)에서 AWQ는 도착률에 묶여 1.48배에서 멈췼고 입력·출력이 같은 비율로 올랐다. 교재의 "2.37배가 전부 입력 쪽"은 서버 포화 상태의 분해다.
4. **"커널 시간은 같고 대기만 줄었다"는 가설은 디코드 정상 상태에서 기각** — torch profiler 50 iteration 커널 합 1.19 → 0.48s, nsys node 단위 15.1 → 9.5s. `cudaEventSynchronize` 평균 19.3 → 6.6ms는 그 결과를 보는 창이다. nsys 기본 graph 트레이스는 base 커널 6.5초를 숨겼다(`--cuda-graph-trace=node` 필수). Nsight Compute는 WSL2에서 sudo로도 `ERR_NVGPUCTRPERM`.
5. **GeForce에서 DCGM PROF 필드는 전부 비활성** — NVML `nvmlDeviceGetPcieThroughput` 직접 폴링으로 채운 PCIe TX가 베이스/AWQ 구간 80배 차이(3,233 vs 39 MB/s)를 가리켜 2번 발견의 출발점이 됐다. util 98%인 베이스가 AWQ보다 전력이 낮았다(218 vs 304 W).
6. **게이트웨이 계층의 비용과 위험은 측정 가능하다** — LiteLLM 경유 p50 +5.7ms, 폴백 2.4초, PII 마스킹 동작, Auto Router 분기 확인. Semantic Router 캐시 히트 9~83ms(임베딩 룩업 73ms)는 미스 시 그대로 오버헤드가 되고, 문턱 0.85에서는 "Germany/Italy" 질문이 France 답을 받는 **오탐**이 났다.
7. **Multi-LoRA는 KV −21%·출력 −7%를 내고 어댑터 혼합 자체는 TPOT +5.8%** — 서로 다른 어댑터 요청이 한 배치에 함께 잡힌다. VLM은 같은 이미지 반복 시 mm processor cache + prefix cache가 전처리와 prefill을 모두 흡수해 해상도 16배에도 TTFT가 27~44ms였다(보강 실험 참조).

---

## Self-Check

- [x] 벤치 전 P-state로 유휴 확인, 벤치 중 P0~P2 전환 관찰 (Lab 1) — 유휴 P8, 벤치 P2 고정
- [x] 베이스/AWQ의 가중치·KV·동시성 3배수가 일치하는지 계산 (Lab 1) — 0.37x / 2.46x / 2.46x, KV와 동시성은 일치
- [x] 총 처리량 − 출력 처리량 = 입력 처리량으로 분해, 출력이 그대로인지 확인 (Lab 1) — 출력도 함께 올랐다(도착률 상한)
- [x] 교재 수치 2건(65% / 거의 3배)을 우리 로그로 재계산 (Lab 1) — 63.6% / 2.25배(0.85 동일 조건)
- [x] 총 GPU 커널 시간과 벤치 시간이 따로 노는지 확인 (Lab 2) — graph 트레이스 기본값에서만 따로 놀았다
- [x] 동기화 대기 단축배와 TPOT 개선배 대조 (Lab 2) — 2.9배 vs 2.4배
- [x] Nsight Compute가 WSL2에서 권한 문제 없이 도는지 (Lab 2) — 돌지 않는다, Windows 측 설정 필요
- [x] PCIe TX/RX가 벤치 구간에서 실제로 움직이는지 (Lab 3) — 움직였고 원인 추적으로 이어졌다
- [x] 게이트웨이 폴백·마스킹·시맨틱 캐시 히트 각각 확인 (Lab 4) — 셋 다 확인, 캐시는 오탐 포함
- [x] 이미지 해상도를 올렸을 때 CPU/GPU 사용률이 갈리는지 (Lab 5) — 보강 실험 결과 참조
- [ ] PP/TP/복제 3조건 비교 (Lab 6 — 유료, 사용자 진행 필요)
