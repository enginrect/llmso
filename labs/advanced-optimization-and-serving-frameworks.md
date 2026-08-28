# 고급 LLM 최적화와 서빙 프레임워크 — 실습 기록

모든 측정은 RTX 4090 24GB / WSL2 Ubuntu 24.04에서 했다 ([환경 구축 기록](environment-setup.md), [개념 정리](../advanced-optimization-and-serving-frameworks.md)).

실습 섹션은 4개다. GPU 1장(4090)이라 안 되는 항목이 있다.

| 실습 | 필요 자원 | 4090 판정 |
|---|---|---|
| ① 추측 디코딩 4종 비교 | GPU 24GB | ✅ **핵심 실습** |
| ② LMCache KV 오프로딩 | GPU + 넉넉한 시스템 RAM | ✅ 잘 맞음 |
| ③ 프레임워크 4종 비교 | GPU + **디스크 100GB+** | ✅ 디스크만 주의 |
| ④ 멀티 GPU 병렬화 (TP/DP) | GPU 2장 이상 | ❌ 로컬 불가 → **Runpod 대체** (유료) |

---

## 실습 기록 (Labs)

### Lab 1 — 추측 디코딩: vanilla vs n-gram vs EAGLE-3 ✅ *(핵심)*

**원본**: 교재 Hands-on (Colab A100-80GB, Qwen3-32B). 4090은 24GB이므로 **Qwen3-8B-FP8**(가중치 약 9GB)로 축소 재현한다.

교재 실험 구성 4종을 그대로 따른다:

```bash
# 1) vanilla (기준선)
vllm serve Qwen/Qwen3-8B-FP8 --max-model-len 2048 --gpu-memory-utilization 0.90

# 2) n-gram (기본값)
vllm serve Qwen/Qwen3-8B-FP8 --max-model-len 2048 --gpu-memory-utilization 0.90 \
  --speculative-config '{"method":"ngram","num_speculative_tokens":6,"prompt_lookup_min":4,"prompt_lookup_max":6}'

# 3) improved n-gram (매칭 범위 넓히고 K 낮춤)
vllm serve Qwen/Qwen3-8B-FP8 --max-model-len 2048 --gpu-memory-utilization 0.90 \
  --speculative-config '{"method":"ngram","num_speculative_tokens":4,"prompt_lookup_min":2,"prompt_lookup_max":128}'

# 4) EAGLE-3
vllm serve Qwen/Qwen3-8B-FP8 --max-model-len 2048 --gpu-memory-utilization 0.90 \
  --speculative-config '{"method":"eagle3","model":"<Qwen3-8B용 EAGLE-3 speculator>","num_speculative_tokens":3}'
# ⚠️ Qwen3-8B용 EAGLE-3 speculator 존재 여부 확인 필요 — HF에서 "Qwen3-8B eagle3" 검색.
#    → 있었다: RedHatAI/Qwen3-8B-speculator.eagle3 (교재의 32B용과 같은 RedHatAI 시리즈. 아래 실측에 사용)
```

부하는 교재와 동일하게 **동시성 1(저부하)과 16(고부하)** 두 조건:

```bash
vllm bench serve --model Qwen/Qwen3-8B-FP8 \
  --dataset-name random --num-prompts 16 --max-concurrency 1    # 저부하
vllm bench serve --model Qwen/Qwen3-8B-FP8 \
  --dataset-name random --num-prompts 16 --max-concurrency 16   # 고부하
# 교재는 Spec-Bench의 MT-Bench 프롬프트(writing/roleplay/reasoning/math/coding)를 사용했다.
# 여유가 되면 https://github.com/hemingkx/Spec-Bench 의 question.jsonl로 재현하는 편이 교재와 정합적이다.
```

**미리 알아둘 것 — 교재 결과 (A100, Qwen3-32B)**:
- c=1: improved n-gram +16%, EAGLE-3 약 2배 (56.5 vs 28.9 tok/s)
- c=16: n-gram 두 설정 모두 **vanilla보다 느림** (제안·검증 오버헤드), EAGLE-3는 우세하나 격차 축소
- EAGLE-3는 ITL을 얻는 대신 **TTFT를 지불**한다 (추측 헤드 오버헤드)

**⚠️ 하드웨어 함정**: 스터디 운영자의 RTX(VRAM 16GB) 실측에서는 **c=1에서 vanilla가 이미 GPU 98% 포화**라 추측 디코딩이 전부 역효과였다. 교재 한계절의 명제("이미 compute-bound면 효과 없음")가 A100보다 작은 카드에서 c=1부터 성립해버린 사례. 4090은 대역폭 1,008GB/s로 여유가 더 있지만, **먼저 vanilla c=1에서 `nvtop`으로 GPU 사용률부터 확인하고 시작할 것.** 이미 포화 상태면 이 실습의 결론은 "역효과의 재현"이 된다 — 그것도 유효한 결과다.

측정 지표: output throughput(tok/s), TTFT, TPOT/ITL — 4구성 × 2동시성 = 8셀 표로 기록.

```shell
# 준비 — Spec-Bench(교재와 동일 출처)의 MT-Bench 프롬프트에서
# writing/roleplay/reasoning/math/coding 카테고리 50개를 {"prompt": ...} jsonl로 변환
curl -sL -o specbench-question.jsonl https://raw.githubusercontent.com/hemingkx/Spec-Bench/main/data/spec_bench/question.jsonl
python make_mtbench_prompts.py specbench-question.jsonl mtbench-prompts.jsonl
# → 50 prompts

# 함정 1: --dataset-name custom은 pandas가 필요하다 (sharegpt/random은 불필요했음)
#   ImportError: Please install vllm[bench] for bench support ← 실제로는 pandas 부재
uv pip install pandas   # → pandas==3.0.5 로 해결

# 벤치 명령 (모든 구성 공통, c만 1/16 변경):
vllm bench serve --backend vllm --model Qwen/Qwen3-8B-FP8 --endpoint /v1/completions \
  --dataset-name custom --dataset-path mtbench-prompts.jsonl --custom-output-len 256 \
  --num-prompts 16 --max-concurrency {1|16} \
  --percentile-metrics ttft,tpot,itl --metric-percentiles 50,95,99

# vanilla 서버 기동 (GPU KV cache 크기 — 이후 모든 구성 동일 조건)
vllm serve Qwen/Qwen3-8B-FP8 --max-model-len 2048 --gpu-memory-utilization 0.90 --port 8000
# (EngineCore pid=3597) INFO 08-27 20:27:14 [kv_cache_utils.py:2235] GPU KV cache size: 90,080 tokens

# ================= 구성: vanilla =================

# --- vanilla  c=1 ---
# GPU util c=1 (vanilla): mean 94%, max 100% (n=130)
...
Request throughput (req/s):              0.13      
Output token throughput (tok/s):         32.49     
Mean TTFT (ms):                          32.31     
Median TTFT (ms):                        27.50     
P99 TTFT (ms):                           70.23     
Mean TPOT (ms):                          30.77     
Median TPOT (ms):                        27.11     
P99 TPOT (ms):                           40.84     
...

# --- vanilla  c=16 ---
...
Request throughput (req/s):              1.66      
Output token throughput (tok/s):         425.34    
Mean TTFT (ms):                          357.48    
Median TTFT (ms):                        363.51    
P99 TTFT (ms):                           364.76    
Mean TPOT (ms):                          36.35     
Median TPOT (ms):                        36.33     
P99 TPOT (ms):                           36.65     
...

# ================= 구성: ngram =================
vllm serve Qwen/Qwen3-8B-FP8 --max-model-len 2048 --gpu-memory-utilization 0.90 --port 8000 \
  --speculative-config '{"method":"ngram","num_speculative_tokens":6,"prompt_lookup_min":4,"prompt_lookup_max":6}'

# --- ngram  c=1 ---
# GPU util c=1 (ngram): mean 80%, max 91% (n=83)
...
Request throughput (req/s):              0.21      
Output token throughput (tok/s):         53.67     
Mean TTFT (ms):                          135.78    
Median TTFT (ms):                        23.35     
P99 TTFT (ms):                           1557.66   
Mean TPOT (ms):                          18.17     
Median TPOT (ms):                        18.71     
P99 TPOT (ms):                           19.25     
...
Acceptance rate (%):                     29.55     
Acceptance length:                       2.77      
Drafts:                                  132       
Draft tokens:                            792       
Accepted tokens:                         234       
...

# --- ngram  c=16 ---
...
Request throughput (req/s):              1.64      
Output token throughput (tok/s):         420.86    
Mean TTFT (ms):                          239.78    
Median TTFT (ms):                        241.16    
P99 TTFT (ms):                           242.60    
Mean TPOT (ms):                          35.85     
Median TPOT (ms):                        35.90     
P99 TPOT (ms):                           37.21     
...
Acceptance rate (%):                     37.98     
Acceptance length:                       3.28      
Drafts:                                  147       
Draft tokens:                            882       
Accepted tokens:                         335       
...

# ================= 구성: ngram-improved =================
vllm serve Qwen/Qwen3-8B-FP8 --max-model-len 2048 --gpu-memory-utilization 0.90 --port 8000 \
  --speculative-config '{"method":"ngram","num_speculative_tokens":4,"prompt_lookup_min":2,"prompt_lookup_max":128}'

# --- ngram-improved  c=1 ---
# GPU util c=1 (ngram-improved): mean 78%, max 90% (n=78)
...
Request throughput (req/s):              0.23      
Output token throughput (tok/s):         58.21     
Mean TTFT (ms):                          107.40    
Median TTFT (ms):                        23.83     
P99 TTFT (ms):                           1162.48   
Mean TPOT (ms):                          16.83     
Median TPOT (ms):                        17.33     
P99 TPOT (ms):                           18.29     
...
Acceptance rate (%):                     22.10     
Acceptance length:                       1.88      
Drafts:                                  630       
Draft tokens:                            2520      
Accepted tokens:                         557       
...

# --- ngram-improved  c=16 ---
...
Request throughput (req/s):              2.81      
Output token throughput (tok/s):         720.09    
Mean TTFT (ms):                          321.56    
Median TTFT (ms):                        328.79    
P99 TTFT (ms):                           330.26    
Mean TPOT (ms):                          19.28     
Median TPOT (ms):                        19.63     
P99 TPOT (ms):                           21.00     
...
Acceptance rate (%):                     20.03     
Acceptance length:                       1.80      
Drafts:                                  674       
Draft tokens:                            2696      
Accepted tokens:                         540       
...

# ================= 구성: eagle3 =================
vllm serve Qwen/Qwen3-8B-FP8 --max-model-len 2048 --gpu-memory-utilization 0.90 --port 8000 \
  --speculative-config '{"method":"eagle3","model":"RedHatAI/Qwen3-8B-speculator.eagle3","num_speculative_tokens":3}'

# --- eagle3  c=1 ---
# GPU util c=1 (eagle3): mean 93%, max 100% (n=117)
...
Request throughput (req/s):              0.14      
Output token throughput (tok/s):         36.62     
Mean TTFT (ms):                          173.06    
Median TTFT (ms):                        82.85     
P99 TTFT (ms):                           926.22    
Mean TPOT (ms):                          26.74     
Median TPOT (ms):                        23.81     
P99 TPOT (ms):                           42.41     
...
Acceptance rate (%):                     41.12     
Acceptance length:                       2.23      
Drafts:                                  1832      
Draft tokens:                            5496      
Accepted tokens:                         2260      
...

# --- eagle3  c=16 ---
...
Request throughput (req/s):              0.83      
Output token throughput (tok/s):         211.23    
Mean TTFT (ms):                          4352.08   
Median TTFT (ms):                        4352.28   
P99 TTFT (ms):                           4353.53   
Mean TPOT (ms):                          51.19     
Median TPOT (ms):                        51.34     
P99 TPOT (ms):                           58.66     
...
Acceptance rate (%):                     39.72     
Acceptance length:                       2.19      
Drafts:                                  1869      
Draft tokens:                            5607      
Accepted tokens:                         2227      
...
```

**결과 표 — 4구성 × 2동시성 (MT-Bench 16 프롬프트, 출력 256tok 고정)**:

| 구성 | c=1 tok/s | c=1 TPOT | c=1 TTFT(중앙값) | c=16 tok/s | c=16 TPOT | c=16 TTFT |
|---|---|---|---|---|---|---|
| vanilla | 32.49 | 27.11 ms | 27.5 ms | 425.34 | 36.33 ms | 364 ms |
| n-gram (K=6, lookup 4–6) | 53.67 (**1.65x**) | 18.71 ms | 23.4 ms | 420.86 (0.99x) | 35.90 ms | 241 ms |
| improved n-gram (K=4, lookup 2–128) | 58.21 (**1.79x**) | 17.33 ms | 23.8 ms | 720.09 (**1.69x**) | 19.63 ms | 329 ms |
| EAGLE-3 (K=3) | 36.62 (1.13x) | 23.81 ms | 82.9 ms | 211.23 (**0.50x**) | 51.34 ms | **4,352 ms** |

**수락 통계 (벤치 출력의 Speculative Decoding 통계 — /metrics 누적 카운터 차분과 교차 확인함)**:

| 구성 | 부하 | drafts | draft tokens | accepted | 토큰 수락률 | draft당 추가 토큰 |
|---|---|---|---|---|---|---|
| n-gram | c=1 | 132 | 792 | 234 | 29.5% | 1.77 |
| n-gram | c=16 | 147 | 882 | 335 | 38.0% | 2.28 |
| improved | c=1 | 630 | 2,520 | 557 | 22.1% | 0.88 |
| improved | c=16 | 674 | 2,696 | 540 | 20.0% | 0.80 |
| EAGLE-3 | c=1 | 1,832 | 5,496 | 2,260 | 41.1% | 1.23 |
| EAGLE-3 | c=16 | 1,869 | 5,607 | 2,227 | 39.7% | 1.19 |

관찰 정리:
- **교재(A100·Qwen3-32B)와 순위가 뒤집혔다.** 교재는 EAGLE-3 약 2배 > improved n-gram +16% 순이었는데, 4090·8B-FP8에서는 improved n-gram(1.79x) > n-gram(1.65x) > EAGLE-3(1.13x). EAGLE-3의 수락률(41.1%)이 제일 높은데도 느리다 — speculator 자체 forward 비용이 8B 작은 타깃 모델 대비 상대적으로 커서, 수락 이득을 오버헤드가 갉아먹는 구도(해석).
- **c=16에서 EAGLE-3는 vanilla의 절반(0.50x)으로 붕괴.** TTFT 4.35초 — 16개 prefill이 speculator를 통과하며 직렬화된다. 교재 한계절의 "이미 compute-bound면 역효과" 명제가 고동시성에서 실측으로 재현됐다.
- **기본 n-gram은 c=16에서 중립(0.99x)** — 교재의 "고동시성 역효과"가 우리 조건에선 딱 상쇄점. 반면 improved(K=4, lookup 2–128)는 c=16에서도 1.69x로 우세 — draft가 자주 발화(630 vs 132)하면서도 K가 작아 검증 낭비가 적다.
- **하드웨어 함정 사전 점검**: vanilla c=1에서 GPU util 평균 94%(최대 100%)로 수치상 "포화"였지만 improved n-gram이 +79%를 냈다. nvidia-smi의 util%는 SM이 바쁜 시간 비율일 뿐 compute 여유의 지표가 아니다 — decode가 memory-bound면 util 90%+에서도 추측 디코딩 여지가 있다(운영자 16GB RTX 사례처럼 util만 보고 포기하면 이득을 놓친다).

- key point: 같은 실험이 카드·모델 크기에 따라 결론이 바뀐다 — 4090·8B-FP8에서는 improved n-gram(c=1 1.79x, c=16 1.69x)이 전 구간 최선이었고, EAGLE-3는 수락률 1위(41%)인데도 c=1 1.13x·c=16 0.50x로 최하위였다. 수락률이 아니라 "수락 이득 − speculator 비용"이 성능을 결정한다.

### Lab 2 — LMCache: KV 캐시 CPU 오프로딩 ✅

**원본**: 교재 Hands-on (A100, Qwen3-14B). 4090에서는 **Qwen3-14B-FP8**(약 15GB) 또는 Qwen3-8B-FP8로.

```bash
pip install lmcache   # venv 주의 — vLLM 버전 호환 확인

export LMCACHE_USE_EXPERIMENTAL=True
export LMCACHE_LOCAL_CPU=True
export LMCACHE_MAX_LOCAL_CPU_SIZE=24.0   # WSL 할당 RAM(31GiB)에서 여유를 남길 것

vllm serve Qwen/Qwen3-14B-FP8 \
  --kv-transfer-config '{"kv_connector":"LMCacheConnectorV1","kv_role":"kv_both"}' \
  --max-model-len 32768 --gpu-memory-utilization 0.9
```

교재 벤치마크 시나리오 재현:
1. **콜드**: 서로 다른 장문 프리픽스 30개 순차 요청 (unique prefix × 반복 2,000회 트릭으로 생성)
2. **웜**: 같은 30개를 다시 — vanilla vLLM은 GPU 캐시가 넘치는 시점(교재는 10번째)부터 재계산으로 회귀, LMCache는 CPU에서 히트
3. **동시 5요청**: 격차가 벌어지는지 (교재: 처리량 16배 차이)

**미리 알아둘 것**:
- ⚠️ 스터디에서 교재 핸즈온 Colab이 **버전이 안 맞아 제대로 돌지 않고, A100급에서만 의미 있는 결과**가 나온다고 했다. 로컬에서 막히면 버전 핀을 맞추느라 싸우지 말고 **Runpod A100으로 이동**하는 게 빠르다. LMCache 공식 Quickstart(MP 모드)가 더 안정적인 출발점
- 콜드에서는 LMCache가 **항상 더 느리다**(오버헤드) — 캐시 재사용 없는 워크로드면 켜지 말라는 게 교재 결론
- 4090은 A100(80GB)보다 GPU 캐시 공간이 작으므로 vanilla의 축출 시점이 교재보다 **일찍** 온다 → 격차가 더 크게 나올 가능성
- WSL2 변수: [최적화 실습](serving-bottlenecks-and-optimization.md)의 FP8 c300 붕괴 전례가 있다. 이상 결과가 나오면 동시성을 낮춰 재현성부터 확인

```shell
# 별도 venv (메인 venv 보호): lmcache가 vllm 의존성을 건드릴 수 있어 격리
uv venv ~/venvs/lmcache --python 3.12
source ~/venvs/lmcache/bin/activate
uv pip install vllm==0.27.1 lmcache
# → lmcache 0.5.4 / vllm 0.27.1  ← 스터디에서 경고한 "버전 안 맞아 안 돈다" 함정은 이 조합에서 없었다

# 문서 세트: 고유 마커 문장 + 필러 반복 = 문서당 1,256 토큰 × 30개 = 37,680 토큰
#  (GPU KV cache 29,248 토큰보다 29% 크게 설계 — vanilla의 축출을 강제)
# 시나리오: PHASE 1 콜드(순차 30) → PHASE 2 웜(같은 30 재요청) → PHASE 3 웜+동시 5

# ================= A) vanilla vLLM =================
vllm serve Qwen/Qwen3-14B-FP8 --max-model-len 8192 --gpu-memory-utilization 0.90 --port 8000
# (EngineCore) GPU KV cache size: 29,248 tokens
# (EngineCore) Maximum concurrency for 8,192 tokens per request: 3.57x

python lab2_kv_bench.py --model Qwen/Qwen3-14B-FP8 --num-docs 30 --doc-chars 7500

### PHASE 1 — 콜드 (순차 30) (요청 30개, 동시성 1)
  doc 01: TTFT    494.3 ms  total   1.50 s
  doc 02: TTFT    172.6 ms  total   1.17 s
...
  doc 30: TTFT    172.8 ms  total   1.17 s
  => phase 시간 35.0 s | TTFT mean 180.7 ms, median 172.3 ms, max 494.3 ms

### PHASE 2 — 웜 (같은 30 재요청) (요청 30개, 동시성 1)
  doc 01: TTFT    172.9 ms  total   1.17 s
  doc 02: TTFT    173.4 ms  total   1.17 s
...
  doc 30: TTFT    171.7 ms  total   1.17 s
  => phase 시간 34.4 s | TTFT mean 169.1 ms, median 172.3 ms, max 174.7 ms

### PHASE 3 — 웜 + 동시 5 (앞 10개 문서) (요청 10개, 동시성 5)
  doc 01: TTFT    176.2 ms  total   1.68 s
  doc 02: TTFT    430.7 ms  total   1.70 s
...
  doc 10: TTFT    644.2 ms  total   1.59 s
  => phase 시간 3.5 s | TTFT mean 543.8 ms, median 618.1 ms, max 816.8 ms

# 서버 prefix cache 통계 — 웜 패스인데 히트가 0이다:
curl -s http://127.0.0.1:8000/metrics | grep prefix_cache
vllm:prefix_cache_queries_total{engine="0",model_name="Qwen/Qwen3-14B-FP8"} 88027.0
vllm:prefix_cache_hits_total{engine="0",model_name="Qwen/Qwen3-14B-FP8"} 0.0
vllm:external_prefix_cache_queries_total{engine="0",model_name="Qwen/Qwen3-14B-FP8"} 0.0
vllm:external_prefix_cache_hits_total{engine="0",model_name="Qwen/Qwen3-14B-FP8"} 0.0

# ================= B) vLLM + LMCache =================
export LMCACHE_USE_EXPERIMENTAL=True
export LMCACHE_LOCAL_CPU=True
export LMCACHE_MAX_LOCAL_CPU_SIZE=16.0   # WSL 할당 31GiB에서 여유 확보 (계획서 24 → 16으로 하향)
vllm serve Qwen/Qwen3-14B-FP8 --max-model-len 8192 --gpu-memory-utilization 0.90 --port 8000 \
  --kv-transfer-config '{"kv_connector":"LMCacheConnectorV1","kv_role":"kv_both"}'
# (EngineCore) GPU KV cache size: 29,248 tokens  ← GPU 조건은 A와 동일
# 기동 직후 RAM (CPU 캐시 버퍼가 shared 16Gi로 선점됨):
# Mem:            31Gi        20Gi       5.4Gi        16Gi        21Gi        10Gi

python lab2_kv_bench.py --model Qwen/Qwen3-14B-FP8 --num-docs 30 --doc-chars 7500

### PHASE 1 — 콜드 (순차 30) (요청 30개, 동시성 1)
  doc 01: TTFT    469.3 ms  total   1.47 s
  doc 02: TTFT    192.4 ms  total   1.19 s
...
  doc 30: TTFT    180.6 ms  total   1.17 s
  => phase 시간 36.1 s | TTFT mean 195.1 ms, median 186.2 ms, max 469.3 ms

### PHASE 2 — 웜 (같은 30 재요청) (요청 30개, 동시성 1)
  doc 01: TTFT     73.5 ms  total   1.06 s
  doc 02: TTFT     61.2 ms  total   1.08 s
...
  doc 30: TTFT     61.5 ms  total   1.08 s
  => phase 시간 31.8 s | TTFT mean 63.0 ms, median 62.6 ms, max 73.5 ms

### PHASE 3 — 웜 + 동시 5 (앞 10개 문서) (요청 10개, 동시성 5)
  doc 01: TTFT    253.1 ms  total   1.27 s
  doc 02: TTFT    252.3 ms  total   1.27 s
...
  doc 10: TTFT    241.4 ms  total   1.27 s
  => phase 시간 2.5 s | TTFT mean 226.5 ms, median 242.8 ms, max 253.1 ms

# LMCache 조회 로그 (웜 요청 — 1,256 토큰 중 1,024를 CPU에서 복원):
(EngineCore pid=9970) [2026-08-27 20:44:17,910] LMCache INFO: Reqid: cmpl-ba5e29f3bb8f6312-0-a37b6982, Total tokens 1256, Inference Engine computed tokens: 0, LMCache hit tokens: 1024, need to load: 1024 (vllm_v1_adapter.py:1473:lmcache.integration.vllm.vllm_v1_adapter)
...
(EngineCore pid=9970) [2026-08-27 20:44:17,963] LMCache INFO: [req_id=cmpl-ac692826ab043ce4-0-8db1c8f2] Retrieved 1024 out of 1024 required tokens (from 1024 total tokens). size: 0.1562 gb, cost 9.5289 ms, throughput: 16.3974 GB/s; (cache_engine.py:958:lmcache.v1.cache_engine)
...
```

**결과 요약 (TTFT 평균, 30문서·문서당 1,256 토큰)**:

| 단계 | vanilla | LMCache | 비 |
|---|---|---|---|
| PHASE 1 콜드 (순차 30) | 180.7 ms | 195.1 ms | LMCache **+8% 느림** (저장 오버헤드) |
| PHASE 2 웜 (재요청 30) | 169.1 ms | **63.0 ms** | LMCache **2.7x 빠름** |
| PHASE 3 웜+동시 5 | 543.8 ms | **226.5 ms** | LMCache **2.4x 빠름** |

관찰 정리:
- **vanilla의 prefix cache 히트: 88,027 쿼리 중 0.** working set(37,680 토큰)이 GPU 캐시(29,248 토큰)를 29% 초과하는 상태에서 같은 순서로 순차 재접근하면, LRU가 각 문서를 재사용 직전에 축출한다 — 웜 TTFT(169ms)가 콜드(181ms)와 사실상 같다. 교재는 "10번째 문서부터 재계산 회귀"였는데 여기는 **전 문서 축출**로 더 극단적이다(캐시 대비 초과율과 접근 패턴 차이).
- **LMCache 웜 히트의 실체**: 요청당 1,256 토큰 중 1,024 토큰(256-토큰 청크 4개)을 CPU에서 복원 — 복원 비용 약 9.5ms/0.156GB ≈ **16GB/s**. 마지막 부분 청크(232토큰)는 청크 미만이라 캐시되지 않고 재계산된다.
- 콜드가 항상 느리다(+8%)는 교재 명제 재현 — 재사용 없는 워크로드에 LMCache를 켜면 손해다.
- WSL2에서 pinned CPU 버퍼(shared 16Gi 선점) 동작 확인 — `VLLM_WSL2_ENABLE_PIN_MEMORY=1` 환경에서 문제없이 돌았다.

- key point: GPU 캐시보다 29% 큰 working set의 순차 재접근에서 vanilla vLLM은 prefix cache 히트 0/88,027(LRU 최악 케이스)로 웜=콜드가 됐고, LMCache는 CPU에서 16GB/s로 KV를 복원해 웜 TTFT를 2.7배 줄였다. 대신 콜드는 8% 손해 — 재사용 있는 워크로드에서만 켤 것.

### Lab 3 — 서빙 프레임워크 4종 동일 조건 비교 ✅

**원칙 (교재 평가 방법론)**: 동일 모델, 동일 dtype/양자화, 동일 max-len, 동일 부하로 "사과 대 사과" 비교. 모델은 4종 모두에서 돌릴 수 있는 **Qwen2.5-0.5B-Instruct**(빠른 검증)와 **Qwen3-8B 계열**(실측)로.

| 프레임워크 | 설치 경로 | 디스크 | 4090(Ada/SM89) |
|---|---|---|---|
| vLLM | 이미 설치됨 (`~/llmso/.venv`) | — | ✅ |
| llama.cpp | 네이티브 빌드 or 바이너리, GGUF 변환 | 소형 | ✅ 가장 가벼움 |
| SGLang | `docker pull lmsysorg/sglang:latest` | **47GB** | ✅ |
| TensorRT-LLM | `nvcr.io/nvidia/tensorrt-llm/release` | **55GB** | ✅ 공식 지원 |

```bash
# SGLang (노션 실습 흐름 그대로)
docker run -d --name sglang-server --gpus all --shm-size 32g -p 30000:30000 \
  -v ~/.cache/huggingface:/root/.cache/huggingface --ipc=host \
  lmsysorg/sglang:latest python3 -m sglang.launch_server \
  --model-path Qwen/Qwen2.5-0.5B-Instruct --host 0.0.0.0 --port 30000

# TensorRT-LLM
docker run -d --name trtllm-server --gpus all --ipc host \
  --ulimit memlock=-1 --ulimit stack=67108864 \
  -v ~/.cache/huggingface:/root/.cache/huggingface -p 8000:8000 \
  nvcr.io/nvidia/tensorrt-llm/release:latest \
  trtllm-serve serve "Qwen/Qwen2.5-0.5B-Instruct" --host 0.0.0.0 --port 8000

# llama.cpp — GGUF 모델 직접 로드 (예: Qwen3-8B-GGUF Q8_0)
llama-server -hf Qwen/Qwen3-8B-GGUF:Q8_0 --port 8080
```

**⚠️ 디스크**: SGLang 47GB + TensorRT-LLM 55GB = 컨테이너 이미지만 100GB+. Triton 때처럼(27GB) 실습 후 즉시 `docker rmi`로 회수할 것. WSL vhdx는 자동으로 안 줄어든다.

측정: 동일 프롬프트 세트로 TTFT/TPOT/throughput. SGLang은 RadixAttention(prefix 재사용)이 기본이므로 **prefix 공유 프롬프트 세트와 랜덤 세트를 나눠 재면** 프레임워크 차이가 캐시 차이인지 구분된다.

```shell
# 함정 2: docker GPU가 바로 안 된다 —
docker run -d --gpus all ... lmsysorg/sglang:latest ...
# docker: Error response from daemon: could not select device driver "" with capabilities: [[gpu]]
# → NVIDIA Container Toolkit 미설치가 원인. 설치 + 런타임 구성 (sudo 비밀번호 프롬프트 줄은 기록에서 제거):
sudo apt-get install -y nvidia-container-toolkit     # → 1.20.0-1
sudo nvidia-ctk runtime configure --runtime=docker   # /etc/docker/daemon.json 갱신
sudo systemctl restart docker
docker run --rm --gpus all nvcr.io/nvidia/tensorrt-llm/release:1.2.1 nvidia-smi --query-gpu=name,driver_version --format=csv
# name, driver_version
# NVIDIA GeForce RTX 4090, 591.86        ← 통과

# 함정 3: nvcr.io의 tensorrt-llm/release에는 :latest 태그가 없다 — 태그 목록 API로 조회해 1.2.1(최신 안정판) 지정
# 이미지 크기 실측: sglang 47.4GB / tensorrt-llm 59.2GB / llama.cpp server-cuda 6.98GB

# 클라이언트는 4종 모두 동일: vllm bench serve --backend openai (OpenAI 호환 /v1/completions)
vllm bench serve --backend openai --base-url http://127.0.0.1:{PORT} --endpoint /v1/completions \
  --model Qwen/Qwen3-8B --tokenizer Qwen/Qwen3-8B \
  --dataset-name random --random-input-len {512|128} --random-output-len 128 [--random-prefix-len 1024] \
  --num-prompts {8|32} --max-concurrency {1|8}

# ================= 1) vLLM 0.27.1 (venv) =================
vllm serve Qwen/Qwen3-8B --max-model-len 4096 --gpu-memory-utilization 0.90 --port 8000

# --- vllm ① latency — random in512/out128, n=8, c=1 ---
...
Request throughput (req/s):              0.38      
Output token throughput (tok/s):         48.27     
Mean TTFT (ms):                          97.91     
Median TTFT (ms):                        64.97     
P99 TTFT (ms):                           315.57    
Mean TPOT (ms):                          20.11     
Median TPOT (ms):                        20.05     
P99 TPOT (ms):                           21.12     
...

# --- vllm ② random — in512/out128, n=32, c=8 ---
...
Request throughput (req/s):              1.99      
Output token throughput (tok/s):         255.26    
Mean TTFT (ms):                          1189.22   
Median TTFT (ms):                        400.49    
P99 TTFT (ms):                           3800.89   
Mean TPOT (ms):                          22.15     
Median TPOT (ms):                        22.58     
P99 TPOT (ms):                           24.00     
...

# --- vllm ③ prefix-shared — prefix1024 + in128/out128, n=32, c=8 ---
...
Request throughput (req/s):              2.70      
Output token throughput (tok/s):         345.99    
Mean TTFT (ms):                          250.69    
Median TTFT (ms):                        155.69    
P99 TTFT (ms):                           591.58    
Mean TPOT (ms):                          21.31     
Median TPOT (ms):                        21.60     
P99 TPOT (ms):                           22.26     
...

# ================= 2) SGLang 0.5.18 (docker) =================
docker run -d --rm --name sglang-server --gpus all --shm-size 32g -p 30000:30000 \
  -v ~/.cache/huggingface:/root/.cache/huggingface --ipc=host \
  lmsysorg/sglang:latest python3 -m sglang.launch_server \
  --model-path Qwen/Qwen3-8B --context-length 4096 --host 0.0.0.0 --port 30000

# --- sglang ① latency — random in512/out128, n=8, c=1 ---
...
Request throughput (req/s):              0.38      
Output token throughput (tok/s):         48.93     
Mean TTFT (ms):                          121.58    
Median TTFT (ms):                        78.96     
P99 TTFT (ms):                           402.85    
Mean TPOT (ms):                          19.64     
Median TPOT (ms):                        19.84     
P99 TPOT (ms):                           20.17     
...

# --- sglang ② random — in512/out128, n=32, c=8 ---
...
Request throughput (req/s):              2.57      
Output token throughput (tok/s):         329.02    
Mean TTFT (ms):                          321.22    
Median TTFT (ms):                        252.97    
P99 TTFT (ms):                           449.26    
Mean TPOT (ms):                          21.97     
Median TPOT (ms):                        21.37     
P99 TPOT (ms):                           23.00     
...

# --- sglang ③ prefix-shared — prefix1024 + in128/out128, n=32, c=8 ---
...
Request throughput (req/s):              2.78      
Output token throughput (tok/s):         356.41    
Mean TTFT (ms):                          261.91    
Median TTFT (ms):                        143.48    
P99 TTFT (ms):                           676.18    
Mean TPOT (ms):                          20.55     
Median TPOT (ms):                        21.07     
P99 TPOT (ms):                           22.12     
...

# ================= 3) TensorRT-LLM 1.2.1 (docker) =================
docker run -d --rm --name trtllm-server --gpus all --ipc host \
  --ulimit memlock=-1 --ulimit stack=67108864 \
  -v ~/.cache/huggingface:/root/.cache/huggingface -p 8000:8000 \
  nvcr.io/nvidia/tensorrt-llm/release:1.2.1 \
  trtllm-serve serve "Qwen/Qwen3-8B" --host 0.0.0.0 --port 8000

# --- trtllm ① latency — random in512/out128, n=8, c=1 ---
...
Request throughput (req/s):              0.12      
Output token throughput (tok/s):         15.34     
Mean TTFT (ms):                          5848.12   
Median TTFT (ms):                        62.42     
P99 TTFT (ms):                           43112.01  
Mean TPOT (ms):                          19.63     
Median TPOT (ms):                        19.59     
P99 TPOT (ms):                           21.45     
...

# --- trtllm ② random — in512/out128, n=32, c=8 ---
...
Request throughput (req/s):              2.64      
Output token throughput (tok/s):         338.03    
Mean TTFT (ms):                          360.21    
Median TTFT (ms):                        395.26    
P99 TTFT (ms):                           431.50    
Mean TPOT (ms):                          21.01     
Median TPOT (ms):                        21.12     
P99 TPOT (ms):                           23.71     
...

# --- trtllm ③ prefix-shared — prefix1024 + in128/out128, n=32, c=8 ---
...
Request throughput (req/s):              2.45      
Output token throughput (tok/s):         313.71    
Mean TTFT (ms):                          299.18    
Median TTFT (ms):                        90.31     
P99 TTFT (ms):                           1377.12   
Mean TPOT (ms):                          23.20     
Median TPOT (ms):                        22.65     
P99 TPOT (ms):                           28.62     
...

# ================= 4) llama.cpp build 10644 (docker server-cuda, GGUF Q8_0) =================
docker run -d --rm --name llamacpp-server --gpus all -p 8080:8080 \
  ghcr.io/ggml-org/llama.cpp:server-cuda \
  -hf Qwen/Qwen3-8B-GGUF:Q8_0 --alias Qwen/Qwen3-8B -ngl 999 -c 32768 -np 8 \
  --host 0.0.0.0 --port 8080

# --- llamacpp ① latency — random in512/out128, n=8, c=1 ---
...
Request throughput (req/s):              0.63      
Output token throughput (tok/s):         80.32     
Mean TTFT (ms):                          103.32    
Median TTFT (ms):                        82.56     
P99 TTFT (ms):                           234.92    
Mean TPOT (ms):                          11.73     
Median TPOT (ms):                        11.79     
P99 TPOT (ms):                           11.90     
...

# --- llamacpp ② random — in512/out128, n=32, c=8 ---
Successful requests:                     27        
Failed requests:                         5         
...
Request throughput (req/s):              2.70      
Output token throughput (tok/s):         345.20    
Mean TTFT (ms):                          497.61    
Median TTFT (ms):                        466.94    
P99 TTFT (ms):                           714.84    
Mean TPOT (ms):                          16.53     
Median TPOT (ms):                        16.39     
P99 TPOT (ms):                           19.50     
...

# --- llamacpp ③ prefix-shared — prefix1024 + in128/out128, n=32, c=8 ---
Successful requests:                     31        
Failed requests:                         1         
...
Request throughput (req/s):              2.67      
Output token throughput (tok/s):         340.93    
Mean TTFT (ms):                          449.94    
Median TTFT (ms):                        310.77    
P99 TTFT (ms):                           1363.49   
Mean TPOT (ms):                          19.52     
Median TPOT (ms):                        19.31     
P99 TPOT (ms):                           23.24     
...

# 실습 후 즉시 회수 (계획서 원칙):
docker rmi lmsysorg/sglang:latest nvcr.io/nvidia/tensorrt-llm/release:1.2.1 ghcr.io/ggml-org/llama.cpp:server-cuda
docker system df
# TYPE            TOTAL     ACTIVE    SIZE      RECLAIMABLE
# Images          0         0         0B        0B
```

**결과 표 — 동일 모델(Qwen3-8B)·동일 부하 (중앙값 기준)**:

| 프레임워크 | dtype | ① c=1 TPOT | ① TTFT | ② random c=8 tok/s | ② TTFT | ③ prefix c=8 tok/s | ③ TTFT |
|---|---|---|---|---|---|---|---|
| vLLM 0.27.1 | BF16 | 20.05 ms | 65.0 ms | 255.3 | 400 ms | 346.0 | 156 ms |
| SGLang 0.5.18 | BF16 | 19.84 ms | 79.0 ms | **329.0** | **253 ms** | **356.4** | 143 ms |
| TensorRT-LLM 1.2.1 | BF16 | 19.59 ms | 62.4 ms | 338.0 | 395 ms | 313.7 | **90 ms** |
| llama.cpp b10644 | **Q8_0** | **11.79 ms** | 82.6 ms | 345.2 *(27/32 성공)* | 467 ms | 340.9 *(31/32 성공)* | 311 ms |

관찰 정리:
- **BF16 3종의 c=1 TPOT는 19.6~20.1ms로 사실상 동률.** 단일 스트림 decode는 memory-bound(가중치 16GB를 매 토큰 읽음)라 엔진 구현 차이가 드러나지 않는다 — [병목과 최적화 노트](../serving-bottlenecks-and-optimization.md)의 대역폭 계산과 일치하는 결과.
- **llama.cpp의 TPOT 11.8ms는 프레임워크 우위가 아니라 dtype 효과다.** Q8_0 가중치(~8.7GB)가 BF16(16GB)의 절반이라 memory-bound decode가 그만큼 빨라진 것. dtype을 통일하지 않으면 프레임워크 비교가 양자화 비교로 오염된다는 교재 방법론('사과 대 사과')의 반면교사를 일부러 수치로 남긴다.
- **동시성에서 차이가 난다**: random c=8에서 SGLang 329 vs vLLM 255 tok/s (+29%), TTFT도 253 vs 400ms. prefix 세트에선 vLLM이 346으로 따라붙는다(+36% vs 자기 random) — vLLM도 prefix caching이 있으므로, SGLang RadixAttention의 이득은 "prefix 공유가 있는 워크로드에서 격차 축소"로 나타났다.
- **TensorRT-LLM은 첫 요청 워밍업 43초** (c=1 P99 TTFT 43,112ms — 첫 요청에서 그래프 캡처/컴파일). 이를 제외하면 TPOT·TTFT 모두 최상위권. 서빙 재시작이 잦은 환경에서는 이 워밍업이 실질 비용이다.
- **llama.cpp는 c=8 부하에서 32요청 중 5개 연결 끊김**(ServerDisconnectedError) — 단일 스트림 지연은 최강이지만 동시 부하 안정성은 전용 서빙 엔진 대비 약하다.
- 이미지 크기: SGLang 47.4GB, TensorRT-LLM 59.2GB — 실습 후 즉시 `docker rmi`로 회수 완료 (Images 0B).

- key point: 같은 BF16 모델이면 c=1 TPOT는 세 엔진이 19.6~20.1ms로 동률(decode는 memory-bound) — 프레임워크 차이는 동시성 스케줄링(SGLang random c=8 +29%)과 워밍업(TRT-LLM 첫 요청 43초), 안정성(llama.cpp 5/32 끊김)에서 났다. llama.cpp의 TPOT 11.8ms는 Q8_0 dtype 효과이므로 프레임워크 비교표에 낄 수 없는 숫자다.

### Lab 4 — 멀티 GPU 병렬화 (TP/DP) ⬜ *(Runpod, 유료)*

**4090 1장으로는 불가.** TP·PP·EP·PD 분리 전부 GPU 2장 이상이 전제다. 노션의 Runpod 흐름으로 대체한다:

1. Runpod 가입 + $10 선불 + MFA 설정
2. Secret 등록 (`vllm_api_key` — `openssl rand -hex 32`), SSH 공개키 등록
3. Pod: vLLM Verified 템플릿, **GPU 4장 구성** (A100 또는 L40S ×4)
4. `--tensor-parallel-size 4` vs `--tensor-parallel-size 2 --data-parallel-size 2` vs TP=1 DP=4 비교
5. 측정 후 **즉시 Stop → Terminate** (사용자 볼륨 삭제까지 확인 — 과금 지속됨)

참고: 스터디 멤버 공개 가이드(Runpod A100×4, TP=4) — https://ken-0913.github.io/myblog/posts/llm/llm-tp-dp-serving-lab/

**측정 포인트**: 같은 4장으로 TP=4(개별 지연↓)와 DP=4(처리량↑)의 트레이드오프. 스터디 노션에 공유된 SGLang 서빙 케이스북의 "TP는 크다고 좋지 않다" 명제 검증.

```shell
# 미진행 — 4090 1장으로는 TP/DP 실험 자체가 불가하고, Runpod은 유료 결제·계정이 필요해
# 데스크탑 자동 실습 범위에서 제외했다. 진행 시 계획서의 Runpod 흐름(A100 또는 L40S ×4,
# TP=4 vs TP=2·DP=2 vs DP=4, 측정 후 즉시 Terminate)을 그대로 따르면 된다.
```

- key point: (미진행 — Runpod 결제 필요. 멀티 GPU 병렬화는 스터디 후반 EKS 워크숍에서 클라우드로 대체 가능)

---

## ❌ 4090으로 못 하는 것

- **PD 분리 (prefill/decode 물리 분리)** — 최소 2장 + 고속 인터커넥트. llm-d/Dynamo 실습은 스터디 후반 EKS 워크숍과 겹치므로 그때 클라우드에서
- **NVLink/NVSwitch, 노드 간 IB/RoCE** — 하드웨어 부재. KV 전송 대역폭 계산(90Gb/s 요구)은 이론으로만
- **EP (expert parallelism)** — MoE 대형 모델 + 멀티 GPU 전제. DeepEP/EPLB는 개념 정리로
- **멀티노드 DP (Internal/Hybrid/External LB)** — 노드 2대 필요

---

## 정리

- 추측 디코딩의 승자는 부하·하드웨어에 따라 뒤집힌다. 4090·Qwen3-8B-FP8에서 improved n-gram이 c=1 1.79x, c=16 1.69x로 전 구간 최선이었고, 교재(A100·32B)에서 2배 우세였던 EAGLE-3는 c=1 1.13x, c=16 0.50x로 최하위였다. 수락률(EAGLE-3 41%로 1위)이 아니라 "수락 이득 − speculator 비용"이 결과를 정한다.
- nvidia-smi GPU util 94%는 추측 디코딩 포기 사유가 아니었다 — util%는 SM 점유 시간이지 compute 여유가 아니며, memory-bound decode에서는 util 90%+에서도 improved n-gram이 +79%를 냈다.
- LMCache는 설계 의도대로 동작했다: 콜드 +8% 손해, 웜 TTFT 2.7배 단축(63ms), CPU→GPU 복원 16GB/s. 대조군 vanilla는 GPU 캐시보다 29% 큰 working set의 순차 재접근에서 prefix cache 히트 0/88,027 — LRU 최악 케이스를 실측으로 확인했다.
- 같은 BF16 모델이면 단일 스트림 TPOT는 vLLM·SGLang·TensorRT-LLM이 19.6~20.1ms로 동률이다. decode가 memory-bound인 이상 엔진이 바꿀 수 있는 것은 스케줄링(동시성 처리량·TTFT)이지 토큰당 시간이 아니다.
- 프레임워크 차이는 다른 축에서 났다: SGLang은 random c=8 처리량 +29%(vs vLLM), TensorRT-LLM은 첫 요청 워밍업 43초, llama.cpp는 Q8_0 덕에 TPOT 11.8ms(dtype 효과)이지만 동시 부하에서 32요청 중 5개 연결 끊김.
- 인프라 함정 3개를 기록했다: ① vllm bench custom dataset은 pandas 필요 ② docker GPU는 nvidia-container-toolkit 설치+runtime configure 필요 ③ nvcr.io tensorrt-llm에는 :latest 태그가 없다(태그 API로 1.2.1 확인).

---

## Self-Check

- [x] vanilla c=1에서 GPU 사용률 확인 → mean 94%로 수치상 포화였지만 improved n-gram +79% — util%는 전제 판정 지표로 부적합 (Lab 1)
- [x] 4구성 × 2동시성 표 완성 — 교재와 순위 역전(EAGLE-3 최하위) 확인 (Lab 1)
- [x] n-gram 고동시성: 기본 설정은 0.99x로 중립, improved는 1.69x로 오히려 우세 — 교재 경향과 다름 (Lab 1)
- [x] LMCache 콜드 +8% vs 웜 2.7x 실측 (Lab 2)
- [x] vanilla 축출: 교재(10번째)보다 극단 — 순차 재접근에서 전 문서 축출(히트 0/88,027) (Lab 2)
- [x] 프레임워크 4종 동일 조건 벤치 표 (Lab 3)
- [x] prefix 공유 세트 분리 측정 — RadixAttention 이득은 prefix 워크로드에서 vLLM과의 격차 축소로 나타남 (Lab 3)
- [ ] Runpod TP=4 vs DP=4 트레이드오프 (Lab 4 — 유료, 사용자 진행 필요)
- [x] Docker 이미지 회수 완료 — sglang 47.4GB + trtllm 59.2GB + llama.cpp 6.98GB → Images 0B (Lab 3 후)
