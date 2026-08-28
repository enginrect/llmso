# 실습 환경 구축 기록 — Windows 데스크탑 (RTX 4090, WSL2)

**기록 원칙**: 실행한 명령과 stdout을 원문 그대로 남기되, 반복·노이즈(패키지 목록, 진행바, INFO 로그 벽)는 `...`로 생략한다. 에러 메시지와 측정값은 생략하지 않는다.

## 구축 완료된 환경 (as-built)

아래는 계획이 아니라 **구축이 끝난 시점의 실측값**이다. 이후 실습은 전부 이 환경 위에서 돈다.

| 구분 | 항목 | 값 |
|---|---|---|
| 호스트 | OS | Windows (호스트) + WSL2 |
| | GPU | NVIDIA GeForce RTX 4090 24GB (24,564 MiB) |
| | CPU / RAM | i9-13900K 32스레드 / WSL 할당 31 GiB |
| | 호스트 드라이버 | 591.86 (WSL 측 NVIDIA-SMI 590.57, CUDA 13.1) |
| WSL2 | 배포판 | Ubuntu 24.04.1 LTS |
| | 커널 | 5.15.90.1-microsoft-standard-WSL2 |
| | 디스크 여유 | 954 GB |
| Python | 버전 / 관리 | 3.12 / uv (`~/llmso/.venv`) |
| 서빙 | vLLM | 0.27.1 |
| | torch | 2.13.0+cu130 (CC 8.9 인식 확인) |
| | CUDA 툴킷 | 시스템 설치 없음 — venv 내 pip `nvidia/cu13` (nvcc 13.0.88로 정렬) |
| 영속 설정 | `~/.bashrc` | `VLLM_WSL2_ENABLE_PIN_MEMORY=1`, `CUDA_HOME=<venv>/nvidia/cu13`, uv PATH, npm prefix |
| Docker | docker.io (WSL2 직접 설치) | Docker Desktop 미사용 — [서빙 엔진 실습](serving-engine-and-production-architecture.md) Lab 6에서 설치 |

> **WSL2 설치 자체는 이 문서의 범위 밖이다.** 본 기록 시작 전에
> [네이버 블로그 가이드 (alfee0)](https://blog.naver.com/alfee0/224093937737)를 따라
> WSL2 + Ubuntu 24.04 설치를 먼저 완료했다. 이 순서가 중요하다 —
> Step 0(`nvidia-smi` 확인)이 통과한 상태에서 출발했기 때문에,
> 이후의 함정 6개는 전부 WSL2 자체가 아니라 **빌드 도구·CUDA 레이아웃** 쪽에서 나왔다.

**타임라인 주석**: vLLM 최초 설치와 서버 기동 시도 1·2는 최초 설치본에서 실행했다. 이후 기록을 처음부터 일관되게 남기기 위해 `.venv`를 삭제하고 동일 절차를 재실행했다(Step 2~4가 그 기록). uv 캐시에서 재설치되므로 패키지 버전은 완전히 동일하다(패키지 버전은 완전히 동일). 서버 시도 3~7은 재구축된 환경에서 실행했다.

---

## 실습 기록 (Labs)

### Step 0 — 사전 확인: WSL2에서 GPU가 보이는가

```shell
Linux DESKTOP-6KND187 5.15.90.1-microsoft-standard-WSL2 #1 SMP Fri Jan 27 02:56:13 UTC 2023 x86_64 x86_64 x86_64 GNU/Linux

Sat Aug 22 20:14:56 2026       
+-----------------------------------------------------------------------------------------+
| NVIDIA-SMI 590.57                 Driver Version: 591.86         CUDA Version: 13.1     |
+-----------------------------------------+------------------------+----------------------+
| GPU  Name                 Persistence-M | Bus-Id          Disp.A | Volatile Uncorr. ECC |
| Fan  Temp   Perf          Pwr:Usage/Cap |           Memory-Usage | GPU-Util  Compute M. |
|                                         |                        |               MIG M. |
|=========================================+========================+======================|
|   0  NVIDIA GeForce RTX 4090        On  |   00000000:01:00.0  On |                  Off |
|  0%   48C    P0             70W /  450W |    2864MiB /  24564MiB |      2%      Default |
|                                         |                        |                  N/A |
+-----------------------------------------+------------------------+----------------------+

+-----------------------------------------------------------------------------------------+
| Processes:                                                                              |
|  GPU   GI   CI              PID   Type   Process name                        GPU Memory |
|        ID   ID                                                               Usage      |
|=========================================================================================|
|    0   N/A  N/A              23      G   /Xwayland                             N/A      |
+-----------------------------------------------------------------------------------------+
```

- key point: WSL 안에 리눅스 드라이버를 설치하지 않은 상태에서 `nvidia-smi`가 4090(24564MiB)을 인식했다. Windows 호스트 드라이버(591.86)를 `/usr/lib/wsl/lib` 패스스루로 빌려 쓰는 구조가 동작 중이다.

### Step 1 — 기본 도구 확인

```shell
$ python3 --version
Python 3.12.3

$ gcc --version
/bin/bash: line 1: gcc: command not found
```

- key point: Ubuntu 24.04 기본 상태라 Python 3.12.3은 있지만 **gcc가 없다**. 이게 뒤에서 서버 기동 실패(시도 2)의 원인이 된다.

### Step 2 — uv 설치

```shell
$ curl -LsSf https://astral.sh/uv/install.sh | sh
downloading uv 0.12.5 x86_64-unknown-linux-gnu
installing to /home/enginrect/.local/bin
  uv
  uvx
everything's installed!
```

- key point: uv 0.12.5가 `~/.local/bin`에 설치됐다. PATH 반영은 Step 9에서 `.bashrc`로 영속화.

### Step 3 — 가상환경 생성

```shell
$ uv --version
uv 0.12.5 (x86_64-unknown-linux-gnu)

$ uv venv --python 3.12 .venv
Using CPython 3.12.3 interpreter at: /usr/bin/python3.12
Creating virtual environment at: .venv
Activate with: source .venv/bin/activate
```

- key point: 시스템 CPython 3.12.3 기반으로 `~/llmso/.venv` 생성.

### Step 4 — vLLM 설치

재구축 실행이라 uv 캐시에서 설치됐다.

```shell
$ uv pip install vllm
Resolved 195 packages in 21ms
Installed 195 packages in 206ms
...
 + fastapi==0.136.3
 + fastapi-cli==0.0.32
 + fastapi-cloud-cli==0.23.0
...
 + flashinfer-python==0.6.16.post3
...
 + numpy==2.3.5
...
 + nvidia-cuda-nvcc==13.3.73
...
 + nvidia-cuda-runtime==13.0.96
 + nvidia-cudnn-cu13==9.20.0.48
 + nvidia-cudnn-frontend==1.27.0
...
 + torch==2.13.0
 + torch-c-dlpack-ext==0.1.5
 + torchaudio==2.11.0
...
 + torchvision==0.28.0
...
 + transformers==5.15.1
 + triton==3.7.1
...
 + uvicorn==0.52.4
...
 + vllm==0.27.1
...
```

- key point: vllm 0.27.1 + torch 2.13.0(+cu130). CUDA 13 계열 라이브러리(cublas 403MB, cudnn 349MB, torch 502MB 등)가 전부 pip 패키지로 딸려온다 — 시스템에 CUDA 툴킷을 따로 설치하지 않았다. 195개 패키지가 캐시에서 206ms에 설치됐다.

### Step 5 — torch가 4090을 잡는지 확인

```shell
$ python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0), torch.cuda.get_device_capability(0))"
2.13.0+cu130 True NVIDIA GeForce RTX 4090 (8, 9)
```

- key point: `2.13.0+cu130`, CUDA available, **Compute Capability (8, 9)** — 계획 문서에서 정리한 Ada CC 8.9와 일치. FP8 W8A8 지원 하한을 충족한다.

---

## 서버 기동 — 함정 6개와 7번의 시도

`vllm serve Qwen/Qwen2.5-0.5B-Instruct --max-model-len 2048 --gpu-memory-utilization 0.85` 한 줄을 띄우기까지 함정 6개를 통과했다. 전부 사전 계획 문서의 "알아둘 함정" 목록에 **없던** 것들이다.

### 시도 1 — ❌ `RuntimeError: UVA is not available`

명령 (최초 설치본에서 실행):

```shell
$ vllm serve Qwen/Qwen2.5-0.5B-Instruct --max-model-len 2048 --gpu-memory-utilization 0.85
```

출력 (발췌):

```shell
(APIServer pid=8041) INFO 08-22 20:11:35 [api_utils.py:345] 
(APIServer pid=8041) INFO 08-22 20:11:35 [api_utils.py:345]        █     █     █▄   ▄█
(APIServer pid=8041) INFO 08-22 20:11:35 [api_utils.py:345]  ▄▄ ▄█ █     █     █ ▀▄▀ █  version 0.27.1
(APIServer pid=8041) INFO 08-22 20:11:35 [api_utils.py:345]   █▄█▀ █     █     █     █  model   Qwen/Qwen2.5-0.5B-Instruct
...
(APIServer pid=8041) INFO 08-22 20:11:35 [api_utils.py:273] non-default args: {'model_tag': 'Qwen/Qwen2.5-0.5B-Instruct', 'model': 'Qwen/Qwen2.5-0.5B-Instruct', 'max_model_len': 2048, 'gpu_memory_utilization': 0.85}
...
(EngineCore pid=8208) ERROR 08-22 20:11:54 [core.py:1349] EngineCore failed to start.
(EngineCore pid=8208) ERROR 08-22 20:11:54 [core.py:1349] Traceback (most recent call last):
(EngineCore pid=8208) ERROR 08-22 20:11:54 [core.py:1349]   File "/home/enginrect/llmso/.venv/lib/python3.12/site-packages/vllm/v1/engine/core.py", line 1318, in run_engine_core
(EngineCore pid=8208) ERROR 08-22 20:11:54 [core.py:1349]     engine_core = EngineCoreProc(*args, engine_index=dp_rank, **kwargs)
(EngineCore pid=8208) ERROR 08-22 20:11:54 [core.py:1349]                   ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
(EngineCore pid=8208) ERROR 08-22 20:11:54 [core.py:1349]   File "/home/enginrect/llmso/.venv/lib/python3.12/site-packages/vllm/tracing/otel.py", line 178, in sync_wrapper
(EngineCore pid=8208) ERROR 08-22 20:11:54 [core.py:1349]     return func(*args, **kwargs)
(EngineCore pid=8208) ERROR 08-22 20:11:54 [core.py:1349]            ^^^^^^^^^^^^^^^^^^^^^
(EngineCore pid=8208) ERROR 08-22 20:11:54 [core.py:1349]   File "/home/enginrect/llmso/.venv/lib/python3.12/site-packages/vllm/v1/engine/core.py", line 1074, in __init__
(EngineCore pid=8208) ERROR 08-22 20:11:54 [core.py:1349]     super().__init__(
(EngineCore pid=8208) ERROR 08-22 20:11:54 [core.py:1349]   File "/home/enginrect/llmso/.venv/lib/python3.12/site-packages/vllm/v1/engine/core.py", line 132, in __init__
(EngineCore pid=8208) ERROR 08-22 20:11:54 [core.py:1349]     self.model_executor = executor_class(vllm_config)
(EngineCore pid=8208) ERROR 08-22 20:11:54 [core.py:1349]                           ^^^^^^^^^^^^^^^^^^^^^^^^^^^
(EngineCore pid=8208) ERROR 08-22 20:11:54 [core.py:1349]   File "/home/enginrect/llmso/.venv/lib/python3.12/site-packages/vllm/tracing/otel.py", line 178, in sync_wrapper
(EngineCore pid=8208) ERROR 08-22 20:11:54 [core.py:1349]     return func(*args, **kwargs)
(EngineCore pid=8208) ERROR 08-22 20:11:54 [core.py:1349]            ^^^^^^^^^^^^^^^^^^^^^
(EngineCore pid=8208) ERROR 08-22 20:11:54 [core.py:1349]   File "/home/enginrect/llmso/.venv/lib/python3.12/site-packages/vllm/v1/executor/abstract.py", line 109, in __init__
(EngineCore pid=8208) ERROR 08-22 20:11:54 [core.py:1349]     self._init_executor()
(EngineCore pid=8208) ERROR 08-22 20:11:54 [core.py:1349]   File "/home/enginrect/llmso/.venv/lib/python3.12/site-packages/vllm/v1/executor/uniproc_executor.py", line 63, in _init_executor
(EngineCore pid=8208) ERROR 08-22 20:11:54 [core.py:1349]     self.driver_worker.init_device()
(EngineCore pid=8208) ERROR 08-22 20:11:54 [core.py:1349]   File "/home/enginrect/llmso/.venv/lib/python3.12/site-packages/vllm/v1/worker/worker_base.py", line 331, in init_device
(EngineCore pid=8208) ERROR 08-22 20:11:54 [core.py:1349]     self.worker.init_device()  # type: ignore
(EngineCore pid=8208) ERROR 08-22 20:11:54 [core.py:1349]     ^^^^^^^^^^^^^^^^^^^^^^^^^
(EngineCore pid=8208) ERROR 08-22 20:11:54 [core.py:1349]   File "/home/enginrect/llmso/.venv/lib/python3.12/site-packages/vllm/tracing/otel.py", line 178, in sync_wrapper
(EngineCore pid=8208) ERROR 08-22 20:11:54 [core.py:1349]     return func(*args, **kwargs)
(EngineCore pid=8208) ERROR 08-22 20:11:54 [core.py:1349]            ^^^^^^^^^^^^^^^^^^^^^
(EngineCore pid=8208) ERROR 08-22 20:11:54 [core.py:1349]   File "/home/enginrect/llmso/.venv/lib/python3.12/site-packages/vllm/v1/worker/gpu_worker.py", line 415, in init_device
(EngineCore pid=8208) ERROR 08-22 20:11:54 [core.py:1349]     self.model_runner: GPUModelRunner = GPUModelRunnerV2(  # type: ignore
(EngineCore pid=8208) ERROR 08-22 20:11:54 [core.py:1349]                                         ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
(EngineCore pid=8208) ERROR 08-22 20:11:54 [core.py:1349]   File "/home/enginrect/llmso/.venv/lib/python3.12/site-packages/vllm/v1/worker/gpu/model_runner.py", line 237, in __init__
(EngineCore pid=8208) ERROR 08-22 20:11:54 [core.py:1349]     self.req_states = RequestState(
(EngineCore pid=8208) ERROR 08-22 20:11:54 [core.py:1349]                       ^^^^^^^^^^^^^
(EngineCore pid=8208) ERROR 08-22 20:11:54 [core.py:1349]   File "/home/enginrect/llmso/.venv/lib/python3.12/site-packages/vllm/v1/worker/gpu/states.py", line 34, in __init__
(EngineCore pid=8208) ERROR 08-22 20:11:54 [core.py:1349]     self.all_token_ids = StagedWriteTensor(
(EngineCore pid=8208) ERROR 08-22 20:11:54 [core.py:1349]                          ^^^^^^^^^^^^^^^^^^
(EngineCore pid=8208) ERROR 08-22 20:11:54 [core.py:1349]   File "/home/enginrect/llmso/.venv/lib/python3.12/site-packages/vllm/v1/worker/gpu/buffer_utils.py", line 141, in __init__
(EngineCore pid=8208) ERROR 08-22 20:11:54 [core.py:1349]     self._uva_buf = UvaBuffer(size, dtype)
(EngineCore pid=8208) ERROR 08-22 20:11:54 [core.py:1349]                     ^^^^^^^^^^^^^^^^^^^^^^
(EngineCore pid=8208) ERROR 08-22 20:11:54 [core.py:1349]   File "/home/enginrect/llmso/.venv/lib/python3.12/site-packages/vllm/v1/worker/gpu/buffer_utils.py", line 47, in __init__
(EngineCore pid=8208) ERROR 08-22 20:11:54 [core.py:1349]     raise RuntimeError("UVA is not available")
(EngineCore pid=8208) ERROR 08-22 20:11:54 [core.py:1349] RuntimeError: UVA is not available
...
(EngineCore pid=8208) Traceback (most recent call last):
...
(EngineCore pid=8208)     raise e
...
(EngineCore pid=8208)     raise RuntimeError("UVA is not available")
(EngineCore pid=8208) RuntimeError: UVA is not available
...
(APIServer pid=8041) Traceback (most recent call last):
...
(APIServer pid=8041)   File "/home/enginrect/llmso/.venv/lib/python3.12/site-packages/vllm/v1/engine/utils.py", line 1253, in wait_for_engine_startup
(APIServer pid=8041)     raise RuntimeError(
(APIServer pid=8041) RuntimeError: Engine core initialization failed. See root cause above. Failed core proc(s): {}
```

**원인 분석** (vLLM 0.27.1 소스에서 확인):

- V2 Model Runner가 요청 상태 버퍼를 UVA(Unified Virtual Addressing) 버퍼로 만든다 → `is_uva_available()` = `is_pin_memory_available()`.
- `vllm/platforms/interface.py:993`: WSL이면 무조건 `pin_memory=False` (보수적 기본값).
- `vllm/platforms/cuda.py:286`: CUDA 플랫폼 오버라이드 — WSL2 커널이 4.19.121 이상이면 pinned memory를 지원하지만 **기본 비활성**이고, `VLLM_WSL2_ENABLE_PIN_MEMORY=1`로 opt-in해야 한다.

```python
# vllm/platforms/cuda.py:300-304 (발췌)
            # On compatible WSL2 kernels, pinned memory is supported but
            # disabled by default. Enable it via VLLM_WSL2_ENABLE_PIN_MEMORY=1.
            import vllm.envs as envs

            return envs.VLLM_WSL2_ENABLE_PIN_MEMORY
```

이 데스크탑의 커널은 5.15.90.1 > 4.19.121이므로 환경변수만 켜면 된다.

**해결**: `export VLLM_WSL2_ENABLE_PIN_MEMORY=1` (이후 모든 시도에 적용, Step 9에서 영속화)

- key point: vLLM 0.27.1 V2 Model Runner는 pinned memory가 없으면 기동 자체가 안 된다. WSL2에서는 명시적 opt-in이 필요하다.

### 시도 2 — ❌ `Failed to find C compiler`

명령 (최초 설치본, `VLLM_WSL2_ENABLE_PIN_MEMORY=1` 적용):

```shell
$ export VLLM_WSL2_ENABLE_PIN_MEMORY=1
$ vllm serve Qwen/Qwen2.5-0.5B-Instruct --max-model-len 2048 --gpu-memory-utilization 0.85
```

출력 (발췌):

```shell
(APIServer pid=8545) INFO 08-22 20:12:58 [api_utils.py:345] 
(APIServer pid=8545) INFO 08-22 20:12:58 [api_utils.py:345]        █     █     █▄   ▄█
(APIServer pid=8545) INFO 08-22 20:12:58 [api_utils.py:345]  ▄▄ ▄█ █     █     █ ▀▄▀ █  version 0.27.1
(APIServer pid=8545) INFO 08-22 20:12:58 [api_utils.py:345]   █▄█▀ █     █     █     █  model   Qwen/Qwen2.5-0.5B-Instruct
...
(APIServer pid=8545) INFO 08-22 20:12:58 [api_utils.py:273] non-default args: {'model_tag': 'Qwen/Qwen2.5-0.5B-Instruct', 'model': 'Qwen/Qwen2.5-0.5B-Instruct', 'max_model_len': 2048, 'gpu_memory_utilization': 0.85}
...
(EngineCore pid=8654) INFO 08-22 20:13:25 [weight_utils.py:867] Filesystem type for checkpoints: EXT4. Checkpoint size: 0.92 GiB. Available RAM: 28.06 GiB.
...
(EngineCore pid=8654) INFO 08-22 20:13:25 [model_runner.py:329] Model loading took 0.93 GiB and 15.393893 seconds
...
(EngineCore pid=8654) ERROR 08-22 20:13:29 [core.py:1349] EngineCore failed to start.
(EngineCore pid=8654) ERROR 08-22 20:13:29 [core.py:1349] Traceback (most recent call last):
(EngineCore pid=8654) ERROR 08-22 20:13:29 [core.py:1349]   File "/home/enginrect/llmso/.venv/lib/python3.12/site-packages/vllm/v1/engine/core.py", line 1318, in run_engine_core
(EngineCore pid=8654) ERROR 08-22 20:13:29 [core.py:1349]     engine_core = EngineCoreProc(*args, engine_index=dp_rank, **kwargs)
(EngineCore pid=8654) ERROR 08-22 20:13:29 [core.py:1349]                   ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
(EngineCore pid=8654) ERROR 08-22 20:13:29 [core.py:1349]   File "/home/enginrect/llmso/.venv/lib/python3.12/site-packages/vllm/tracing/otel.py", line 178, in sync_wrapper
(EngineCore pid=8654) ERROR 08-22 20:13:29 [core.py:1349]     return func(*args, **kwargs)
(EngineCore pid=8654) ERROR 08-22 20:13:29 [core.py:1349]            ^^^^^^^^^^^^^^^^^^^^^
(EngineCore pid=8654) ERROR 08-22 20:13:29 [core.py:1349]   File "/home/enginrect/llmso/.venv/lib/python3.12/site-packages/vllm/v1/engine/core.py", line 1074, in __init__
(EngineCore pid=8654) ERROR 08-22 20:13:29 [core.py:1349]     super().__init__(
(EngineCore pid=8654) ERROR 08-22 20:13:29 [core.py:1349]   File "/home/enginrect/llmso/.venv/lib/python3.12/site-packages/vllm/v1/engine/core.py", line 143, in __init__
(EngineCore pid=8654) ERROR 08-22 20:13:29 [core.py:1349]     kv_cache_config = self._initialize_kv_caches(vllm_config)
(EngineCore pid=8654) ERROR 08-22 20:13:29 [core.py:1349]                       ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
(EngineCore pid=8654) ERROR 08-22 20:13:29 [core.py:1349]   File "/home/enginrect/llmso/.venv/lib/python3.12/site-packages/vllm/tracing/otel.py", line 178, in sync_wrapper
(EngineCore pid=8654) ERROR 08-22 20:13:29 [core.py:1349]     return func(*args, **kwargs)
(EngineCore pid=8654) ERROR 08-22 20:13:29 [core.py:1349]            ^^^^^^^^^^^^^^^^^^^^^
(EngineCore pid=8654) ERROR 08-22 20:13:29 [core.py:1349]   File "/home/enginrect/llmso/.venv/lib/python3.12/site-packages/vllm/v1/engine/core.py", line 293, in _initialize_kv_caches
(EngineCore pid=8654) ERROR 08-22 20:13:29 [core.py:1349]     available_gpu_memory = self.model_executor.determine_available_memory()
(EngineCore pid=8654) ERROR 08-22 20:13:29 [core.py:1349]                            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
(EngineCore pid=8654) ERROR 08-22 20:13:29 [core.py:1349]   File "/home/enginrect/llmso/.venv/lib/python3.12/site-packages/vllm/v1/executor/abstract.py", line 147, in determine_available_memory
(EngineCore pid=8654) ERROR 08-22 20:13:29 [core.py:1349]     return self.collective_rpc("determine_available_memory")
(EngineCore pid=8654) ERROR 08-22 20:13:29 [core.py:1349]            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
(EngineCore pid=8654) ERROR 08-22 20:13:29 [core.py:1349]   File "/home/enginrect/llmso/.venv/lib/python3.12/site-packages/vllm/v1/executor/uniproc_executor.py", line 92, in collective_rpc
(EngineCore pid=8654) ERROR 08-22 20:13:29 [core.py:1349]     result = run_method(self.driver_worker, method, args, kwargs)
(EngineCore pid=8654) ERROR 08-22 20:13:29 [core.py:1349]              ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
(EngineCore pid=8654) ERROR 08-22 20:13:29 [core.py:1349]   File "/home/enginrect/llmso/.venv/lib/python3.12/site-packages/vllm/v1/serial_utils.py", line 510, in run_method
(EngineCore pid=8654) ERROR 08-22 20:13:29 [core.py:1349]     return func(*args, **kwargs)
(EngineCore pid=8654) ERROR 08-22 20:13:29 [core.py:1349]            ^^^^^^^^^^^^^^^^^^^^^
(EngineCore pid=8654) ERROR 08-22 20:13:29 [core.py:1349]   File "/home/enginrect/llmso/.venv/lib/python3.12/site-packages/torch/utils/_contextlib.py", line 124, in decorate_context
(EngineCore pid=8654) ERROR 08-22 20:13:29 [core.py:1349]     return func(*args, **kwargs)
(EngineCore pid=8654) ERROR 08-22 20:13:29 [core.py:1349]            ^^^^^^^^^^^^^^^^^^^^^
(EngineCore pid=8654) ERROR 08-22 20:13:29 [core.py:1349]   File "/home/enginrect/llmso/.venv/lib/python3.12/site-packages/vllm/v1/worker/gpu_worker.py", line 504, in determine_available_memory
(EngineCore pid=8654) ERROR 08-22 20:13:29 [core.py:1349]     self.model_runner.profile_run()
(EngineCore pid=8654) ERROR 08-22 20:13:29 [core.py:1349]   File "/home/enginrect/llmso/.venv/lib/python3.12/site-packages/torch/utils/_contextlib.py", line 124, in decorate_context
(EngineCore pid=8654) ERROR 08-22 20:13:29 [core.py:1349]     return func(*args, **kwargs)
(EngineCore pid=8654) ERROR 08-22 20:13:29 [core.py:1349]            ^^^^^^^^^^^^^^^^^^^^^
(EngineCore pid=8654) ERROR 08-22 20:13:29 [core.py:1349]   File "/home/enginrect/llmso/.venv/lib/python3.12/site-packages/vllm/v1/worker/gpu/model_runner.py", line 719, in profile_run
(EngineCore pid=8654) ERROR 08-22 20:13:29 [core.py:1349]     hidden_states, sample_hidden_states = self._dummy_run(
(EngineCore pid=8654) ERROR 08-22 20:13:29 [core.py:1349]                                           ^^^^^^^^^^^^^^^^
(EngineCore pid=8654) ERROR 08-22 20:13:29 [core.py:1349]   File "/home/enginrect/llmso/.venv/lib/python3.12/site-packages/torch/utils/_contextlib.py", line 124, in decorate_context
(EngineCore pid=8654) ERROR 08-22 20:13:29 [core.py:1349]     return func(*args, **kwargs)
(EngineCore pid=8654) ERROR 08-22 20:13:29 [core.py:1349]            ^^^^^^^^^^^^^^^^^^^^^
(EngineCore pid=8654) ERROR 08-22 20:13:29 [core.py:1349]   File "/home/enginrect/llmso/.venv/lib/python3.12/site-packages/vllm/v1/worker/gpu/eplb_utils.py", line 38, in wrapper
(EngineCore pid=8654) ERROR 08-22 20:13:29 [core.py:1349]     result = fn(self, *args, **kwargs)
(EngineCore pid=8654) ERROR 08-22 20:13:29 [core.py:1349]              ^^^^^^^^^^^^^^^^^^^^^^^^^
(EngineCore pid=8654) ERROR 08-22 20:13:29 [core.py:1349]   File "/home/enginrect/llmso/.venv/lib/python3.12/site-packages/vllm/v1/worker/gpu/model_runner.py", line 613, in _dummy_run
(EngineCore pid=8654) ERROR 08-22 20:13:29 [core.py:1349]     self.execute_model(
(EngineCore pid=8654) ERROR 08-22 20:13:29 [core.py:1349]   File "/home/enginrect/llmso/.venv/lib/python3.12/site-packages/torch/utils/_contextlib.py", line 124, in decorate_context
(EngineCore pid=8654) ERROR 08-22 20:13:29 [core.py:1349]     return func(*args, **kwargs)
(EngineCore pid=8654) ERROR 08-22 20:13:29 [core.py:1349]            ^^^^^^^^^^^^^^^^^^^^^
(EngineCore pid=8654) ERROR 08-22 20:13:29 [core.py:1349]   File "/home/enginrect/llmso/.venv/lib/python3.12/site-packages/vllm/v1/worker/gpu/model_runner.py", line 1423, in execute_model
(EngineCore pid=8654) ERROR 08-22 20:13:29 [core.py:1349]     model_output = self.model(**model_inputs)
(EngineCore pid=8654) ERROR 08-22 20:13:29 [core.py:1349]                    ^^^^^^^^^^^^^^^^^^^^^^^^^^
(EngineCore pid=8654) ERROR 08-22 20:13:29 [core.py:1349]   File "/home/enginrect/llmso/.venv/lib/python3.12/site-packages/torch/nn/modules/module.py", line 1778, in _wrapped_call_impl
(EngineCore pid=8654) ERROR 08-22 20:13:29 [core.py:1349]     return self._call_impl(*args, **kwargs)
(EngineCore pid=8654) ERROR 08-22 20:13:29 [core.py:1349]            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
(EngineCore pid=8654) ERROR 08-22 20:13:29 [core.py:1349]   File "/home/enginrect/llmso/.venv/lib/python3.12/site-packages/torch/nn/modules/module.py", line 1789, in _call_impl
(EngineCore pid=8654) ERROR 08-22 20:13:29 [core.py:1349]     return forward_call(*args, **kwargs)
(EngineCore pid=8654) ERROR 08-22 20:13:29 [core.py:1349]            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
(EngineCore pid=8654) ERROR 08-22 20:13:29 [core.py:1349]   File "/home/enginrect/llmso/.venv/lib/python3.12/site-packages/vllm/model_executor/models/qwen2.py", line 491, in forward
(EngineCore pid=8654) ERROR 08-22 20:13:29 [core.py:1349]     hidden_states = self.model(
(EngineCore pid=8654) ERROR 08-22 20:13:29 [core.py:1349]                     ^^^^^^^^^^^
(EngineCore pid=8654) ERROR 08-22 20:13:29 [core.py:1349]   File "/home/enginrect/llmso/.venv/lib/python3.12/site-packages/vllm/compilation/decorators.py", line 663, in __call__
(EngineCore pid=8654) ERROR 08-22 20:13:29 [core.py:1349]     self.aot_compiled_fn = self.aot_compile(*args, **kwargs)
(EngineCore pid=8654) ERROR 08-22 20:13:29 [core.py:1349]                            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
(EngineCore pid=8654) ERROR 08-22 20:13:29 [core.py:1349]   File "/home/enginrect/llmso/.venv/lib/python3.12/site-packages/vllm/compilation/wrapper.py", line 169, in aot_compile
(EngineCore pid=8654) ERROR 08-22 20:13:29 [core.py:1349]     return self._compiled_callable.aot_compile((args, kwargs))
(EngineCore pid=8654) ERROR 08-22 20:13:29 [core.py:1349]            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
(EngineCore pid=8654) ERROR 08-22 20:13:29 [core.py:1349]   File "/home/enginrect/llmso/.venv/lib/python3.12/site-packages/torch/_dynamo/eval_frame.py", line 964, in aot_compile
(EngineCore pid=8654) ERROR 08-22 20:13:29 [core.py:1349]     return aot_compile_fullgraph(
(EngineCore pid=8654) ERROR 08-22 20:13:29 [core.py:1349]            ^^^^^^^^^^^^^^^^^^^^^^
(EngineCore pid=8654) ERROR 08-22 20:13:29 [core.py:1349]   File "/home/enginrect/llmso/.venv/lib/python3.12/site-packages/torch/_dynamo/aot_compile.py", line 396, in aot_compile_fullgraph
(EngineCore pid=8654) ERROR 08-22 20:13:29 [core.py:1349]     compiled_fn = backend(
(EngineCore pid=8654) ERROR 08-22 20:13:29 [core.py:1349]                   ^^^^^^^^
(EngineCore pid=8654) ERROR 08-22 20:13:29 [core.py:1349]   File "/home/enginrect/llmso/.venv/lib/python3.12/site-packages/torch/__init__.py", line 2568, in __call__
(EngineCore pid=8654) ERROR 08-22 20:13:29 [core.py:1349]     return self.compiler_fn(model_, inputs_, **self.kwargs)
(EngineCore pid=8654) ERROR 08-22 20:13:29 [core.py:1349]            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
(EngineCore pid=8654) ERROR 08-22 20:13:29 [core.py:1349]   File "/usr/lib/python3.12/contextlib.py", line 81, in inner
(EngineCore pid=8654) ERROR 08-22 20:13:29 [core.py:1349]     return func(*args, **kwds)
(EngineCore pid=8654) ERROR 08-22 20:13:29 [core.py:1349]            ^^^^^^^^^^^^^^^^^^^
(EngineCore pid=8654) ERROR 08-22 20:13:29 [core.py:1349]   File "/home/enginrect/llmso/.venv/lib/python3.12/site-packages/vllm/compilation/backends.py", line 1222, in __call__
(EngineCore pid=8654) ERROR 08-22 20:13:29 [core.py:1349]     PiecewiseCompileInterpreter(
(EngineCore pid=8654) ERROR 08-22 20:13:29 [core.py:1349]   File "/home/enginrect/llmso/.venv/lib/python3.12/site-packages/vllm/tracing/otel.py", line 178, in sync_wrapper
(EngineCore pid=8654) ERROR 08-22 20:13:29 [core.py:1349]     return func(*args, **kwargs)
(EngineCore pid=8654) ERROR 08-22 20:13:29 [core.py:1349]            ^^^^^^^^^^^^^^^^^^^^^
(EngineCore pid=8654) ERROR 08-22 20:13:29 [core.py:1349]   File "/home/enginrect/llmso/.venv/lib/python3.12/site-packages/vllm/compilation/backends.py", line 728, in run
(EngineCore pid=8654) ERROR 08-22 20:13:29 [core.py:1349]     return super().run(*args)
(EngineCore pid=8654) ERROR 08-22 20:13:29 [core.py:1349]            ^^^^^^^^^^^^^^^^^^
(EngineCore pid=8654) ERROR 08-22 20:13:29 [core.py:1349]   File "/home/enginrect/llmso/.venv/lib/python3.12/site-packages/torch/fx/interpreter.py", line 197, in run
(EngineCore pid=8654) ERROR 08-22 20:13:29 [core.py:1349]     self.env[node] = self.run_node(node)
(EngineCore pid=8654) ERROR 08-22 20:13:29 [core.py:1349]                      ^^^^^^^^^^^^^^^^^^^
(EngineCore pid=8654) ERROR 08-22 20:13:29 [core.py:1349]   File "/home/enginrect/llmso/.venv/lib/python3.12/site-packages/torch/fx/interpreter.py", line 294, in run_node
(EngineCore pid=8654) ERROR 08-22 20:13:29 [core.py:1349]     return getattr(self, n.op)(n.target, args, kwargs)
(EngineCore pid=8654) ERROR 08-22 20:13:29 [core.py:1349]            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
(EngineCore pid=8654) ERROR 08-22 20:13:29 [core.py:1349]   File "/home/enginrect/llmso/.venv/lib/python3.12/site-packages/vllm/compilation/backends.py", line 755, in call_module
(EngineCore pid=8654) ERROR 08-22 20:13:29 [core.py:1349]     piecewise_backend = PiecewiseBackend(
(EngineCore pid=8654) ERROR 08-22 20:13:29 [core.py:1349]                         ^^^^^^^^^^^^^^^^^
(EngineCore pid=8654) ERROR 08-22 20:13:29 [core.py:1349]   File "/home/enginrect/llmso/.venv/lib/python3.12/site-packages/vllm/compilation/piecewise_backend.py", line 190, in __init__
(EngineCore pid=8654) ERROR 08-22 20:13:29 [core.py:1349]     self.compile_all_ranges()
(EngineCore pid=8654) ERROR 08-22 20:13:29 [core.py:1349]   File "/home/enginrect/llmso/.venv/lib/python3.12/site-packages/vllm/compilation/piecewise_backend.py", line 266, in compile_all_ranges
(EngineCore pid=8654) ERROR 08-22 20:13:29 [core.py:1349]     range_entry.runnable = self.vllm_backend.compiler_manager.compile(
(EngineCore pid=8654) ERROR 08-22 20:13:29 [core.py:1349]                            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
(EngineCore pid=8654) ERROR 08-22 20:13:29 [core.py:1349]   File "/home/enginrect/llmso/.venv/lib/python3.12/site-packages/vllm/tracing/otel.py", line 178, in sync_wrapper
(EngineCore pid=8654) ERROR 08-22 20:13:29 [core.py:1349]     return func(*args, **kwargs)
(EngineCore pid=8654) ERROR 08-22 20:13:29 [core.py:1349]            ^^^^^^^^^^^^^^^^^^^^^
(EngineCore pid=8654) ERROR 08-22 20:13:29 [core.py:1349]   File "/home/enginrect/llmso/.venv/lib/python3.12/site-packages/vllm/compilation/backends.py", line 353, in compile
(EngineCore pid=8654) ERROR 08-22 20:13:29 [core.py:1349]     compiled_graph, handle = self.compiler.compile(
(EngineCore pid=8654) ERROR 08-22 20:13:29 [core.py:1349]                              ^^^^^^^^^^^^^^^^^^^^^^
(EngineCore pid=8654) ERROR 08-22 20:13:29 [core.py:1349]   File "/home/enginrect/llmso/.venv/lib/python3.12/site-packages/vllm/compilation/compiler_interface.py", line 376, in compile
(EngineCore pid=8654) ERROR 08-22 20:13:29 [core.py:1349]     compiled_graph = standalone_compile(graph, example_inputs, **compile_kwargs)
(EngineCore pid=8654) ERROR 08-22 20:13:29 [core.py:1349]                      ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
(EngineCore pid=8654) ERROR 08-22 20:13:29 [core.py:1349]   File "/home/enginrect/llmso/.venv/lib/python3.12/site-packages/torch/_inductor/__init__.py", line 452, in standalone_compile
(EngineCore pid=8654) ERROR 08-22 20:13:29 [core.py:1349]     return standalone_compile(
(EngineCore pid=8654) ERROR 08-22 20:13:29 [core.py:1349]            ^^^^^^^^^^^^^^^^^^^
(EngineCore pid=8654) ERROR 08-22 20:13:29 [core.py:1349]   File "/home/enginrect/llmso/.venv/lib/python3.12/site-packages/torch/_inductor/standalone_compile.py", line 510, in standalone_compile
(EngineCore pid=8654) ERROR 08-22 20:13:29 [core.py:1349]     compiled_fn = compile_fx(
(EngineCore pid=8654) ERROR 08-22 20:13:29 [core.py:1349]                   ^^^^^^^^^^^
(EngineCore pid=8654) ERROR 08-22 20:13:29 [core.py:1349]   File "/home/enginrect/llmso/.venv/lib/python3.12/site-packages/torch/_inductor/compile_fx.py", line 2723, in compile_fx
(EngineCore pid=8654) ERROR 08-22 20:13:29 [core.py:1349]     return compile_fx(
(EngineCore pid=8654) ERROR 08-22 20:13:29 [core.py:1349]            ^^^^^^^^^^^
(EngineCore pid=8654) ERROR 08-22 20:13:29 [core.py:1349]   File "/home/enginrect/llmso/.venv/lib/python3.12/site-packages/torch/_inductor/compile_fx.py", line 2782, in compile_fx
(EngineCore pid=8654) ERROR 08-22 20:13:29 [core.py:1349]     return _maybe_wrap_and_compile_fx_main(
(EngineCore pid=8654) ERROR 08-22 20:13:29 [core.py:1349]            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
(EngineCore pid=8654) ERROR 08-22 20:13:29 [core.py:1349]   File "/home/enginrect/llmso/.venv/lib/python3.12/site-packages/torch/_inductor/compile_fx.py", line 2863, in _maybe_wrap_and_compile_fx_main
(EngineCore pid=8654) ERROR 08-22 20:13:29 [core.py:1349]     return _compile_fx_main(
(EngineCore pid=8654) ERROR 08-22 20:13:29 [core.py:1349]            ^^^^^^^^^^^^^^^^^
(EngineCore pid=8654) ERROR 08-22 20:13:29 [core.py:1349]   File "/home/enginrect/llmso/.venv/lib/python3.12/site-packages/torch/_inductor/compile_fx.py", line 3076, in _compile_fx_main
(EngineCore pid=8654) ERROR 08-22 20:13:29 [core.py:1349]     raise e.remove_dynamo_frames() from None  # see TORCHDYNAMO_VERBOSE=1
(EngineCore pid=8654) ERROR 08-22 20:13:29 [core.py:1349]     ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
(EngineCore pid=8654) ERROR 08-22 20:13:29 [core.py:1349]   File "/home/enginrect/llmso/.venv/lib/python3.12/site-packages/torch/_inductor/compile_fx.py", line 1020, in _compile_fx_inner
(EngineCore pid=8654) ERROR 08-22 20:13:29 [core.py:1349]     raise InductorError(e, currentframe()).with_traceback(
(EngineCore pid=8654) ERROR 08-22 20:13:29 [core.py:1349]   File "/home/enginrect/llmso/.venv/lib/python3.12/site-packages/torch/_inductor/compile_fx.py", line 1004, in _compile_fx_inner
(EngineCore pid=8654) ERROR 08-22 20:13:29 [core.py:1349]     mb_compiled_graph = fx_codegen_and_compile(
(EngineCore pid=8654) ERROR 08-22 20:13:29 [core.py:1349]                         ^^^^^^^^^^^^^^^^^^^^^^^
(EngineCore pid=8654) ERROR 08-22 20:13:29 [core.py:1349]   File "/home/enginrect/llmso/.venv/lib/python3.12/site-packages/torch/_inductor/compile_fx.py", line 1847, in fx_codegen_and_compile
(EngineCore pid=8654) ERROR 08-22 20:13:29 [core.py:1349]     return scheme.codegen_and_compile(gm, example_inputs, inputs_to_check, graph_kwargs)
(EngineCore pid=8654) ERROR 08-22 20:13:29 [core.py:1349]            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
(EngineCore pid=8654) ERROR 08-22 20:13:29 [core.py:1349]   File "/home/enginrect/llmso/.venv/lib/python3.12/site-packages/torch/_inductor/compile_fx.py", line 1608, in codegen_and_compile
(EngineCore pid=8654) ERROR 08-22 20:13:29 [core.py:1349]     compiled_module = graph.compile_to_module()
(EngineCore pid=8654) ERROR 08-22 20:13:29 [core.py:1349]                       ^^^^^^^^^^^^^^^^^^^^^^^^^
(EngineCore pid=8654) ERROR 08-22 20:13:29 [core.py:1349]   File "/home/enginrect/llmso/.venv/lib/python3.12/site-packages/torch/_inductor/graph.py", line 2669, in compile_to_module
(EngineCore pid=8654) ERROR 08-22 20:13:29 [core.py:1349]     return self._compile_to_module()
(EngineCore pid=8654) ERROR 08-22 20:13:29 [core.py:1349]            ^^^^^^^^^^^^^^^^^^^^^^^^^
(EngineCore pid=8654) ERROR 08-22 20:13:29 [core.py:1349]   File "/home/enginrect/llmso/.venv/lib/python3.12/site-packages/torch/_inductor/graph.py", line 2675, in _compile_to_module
(EngineCore pid=8654) ERROR 08-22 20:13:29 [core.py:1349]     self.codegen_with_cpp_wrapper() if self.cpp_wrapper else self.codegen()
(EngineCore pid=8654) ERROR 08-22 20:13:29 [core.py:1349]                                                              ^^^^^^^^^^^^^^
(EngineCore pid=8654) ERROR 08-22 20:13:29 [core.py:1349]   File "/home/enginrect/llmso/.venv/lib/python3.12/site-packages/torch/_inductor/graph.py", line 2607, in codegen
(EngineCore pid=8654) ERROR 08-22 20:13:29 [core.py:1349]     self._update_scheduler()
(EngineCore pid=8654) ERROR 08-22 20:13:29 [core.py:1349]   File "/home/enginrect/llmso/.venv/lib/python3.12/site-packages/torch/_inductor/graph.py", line 2601, in _update_scheduler
(EngineCore pid=8654) ERROR 08-22 20:13:29 [core.py:1349]     self.scheduler = Scheduler(self.operations)
(EngineCore pid=8654) ERROR 08-22 20:13:29 [core.py:1349]                      ^^^^^^^^^^^^^^^^^^^^^^^^^^
(EngineCore pid=8654) ERROR 08-22 20:13:29 [core.py:1349]   File "/home/enginrect/llmso/.venv/lib/python3.12/site-packages/torch/_inductor/scheduler.py", line 4036, in __init__
(EngineCore pid=8654) ERROR 08-22 20:13:29 [core.py:1349]     self._init(nodes)
(EngineCore pid=8654) ERROR 08-22 20:13:29 [core.py:1349]   File "/home/enginrect/llmso/.venv/lib/python3.12/site-packages/torch/_inductor/scheduler.py", line 4165, in _init
(EngineCore pid=8654) ERROR 08-22 20:13:29 [core.py:1349]     self.create_combo_kernel_nodes(num_ck_nodes=None)
(EngineCore pid=8654) ERROR 08-22 20:13:29 [core.py:1349]   File "/home/enginrect/llmso/.venv/lib/python3.12/site-packages/torch/_inductor/scheduler.py", line 6141, in create_combo_kernel_nodes
(EngineCore pid=8654) ERROR 08-22 20:13:29 [core.py:1349]     if len(window) < 2 or not self.speedup_by_combo_kernel(window):
(EngineCore pid=8654) ERROR 08-22 20:13:29 [core.py:1349]                               ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
(EngineCore pid=8654) ERROR 08-22 20:13:29 [core.py:1349]   File "/home/enginrect/llmso/.venv/lib/python3.12/site-packages/torch/_inductor/scheduler.py", line 9592, in speedup_by_combo_kernel
(EngineCore pid=8654) ERROR 08-22 20:13:29 [core.py:1349]     ms, path = self.benchmark_fused_nodes(node_list)
(EngineCore pid=8654) ERROR 08-22 20:13:29 [core.py:1349]                ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
(EngineCore pid=8654) ERROR 08-22 20:13:29 [core.py:1349]   File "/home/enginrect/llmso/.venv/lib/python3.12/site-packages/torch/_inductor/scheduler.py", line 5048, in benchmark_fused_nodes
(EngineCore pid=8654) ERROR 08-22 20:13:29 [core.py:1349]     return backend.benchmark_fused_nodes(nodes)
(EngineCore pid=8654) ERROR 08-22 20:13:29 [core.py:1349]            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
(EngineCore pid=8654) ERROR 08-22 20:13:29 [core.py:1349]   File "/home/enginrect/llmso/.venv/lib/python3.12/site-packages/torch/_inductor/codegen/cuda_combined_scheduling.py", line 169, in benchmark_fused_nodes
(EngineCore pid=8654) ERROR 08-22 20:13:29 [core.py:1349]     return self._triton_scheduling.benchmark_fused_nodes(nodes)
(EngineCore pid=8654) ERROR 08-22 20:13:29 [core.py:1349]            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
(EngineCore pid=8654) ERROR 08-22 20:13:29 [core.py:1349]   File "/home/enginrect/llmso/.venv/lib/python3.12/site-packages/torch/_inductor/codegen/triton.py", line 7325, in benchmark_fused_nodes
(EngineCore pid=8654) ERROR 08-22 20:13:29 [core.py:1349]     src_code = self.generate_kernel_code_from_nodes(nodes, benchmark_kernel=True)
(EngineCore pid=8654) ERROR 08-22 20:13:29 [core.py:1349]                ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
(EngineCore pid=8654) ERROR 08-22 20:13:29 [core.py:1349]   File "/home/enginrect/llmso/.venv/lib/python3.12/site-packages/torch/_inductor/codegen/simd.py", line 4510, in generate_kernel_code_from_nodes
(EngineCore pid=8654) ERROR 08-22 20:13:29 [core.py:1349]     src_code = kernel.codegen_kernel()
(EngineCore pid=8654) ERROR 08-22 20:13:29 [core.py:1349]                ^^^^^^^^^^^^^^^^^^^^^^^
(EngineCore pid=8654) ERROR 08-22 20:13:29 [core.py:1349]   File "/home/enginrect/llmso/.venv/lib/python3.12/site-packages/torch/_inductor/codegen/triton.py", line 6477, in codegen_kernel
(EngineCore pid=8654) ERROR 08-22 20:13:29 [core.py:1349]     **self.inductor_meta_common(),
(EngineCore pid=8654) ERROR 08-22 20:13:29 [core.py:1349]       ^^^^^^^^^^^^^^^^^^^^^^^^^^^
(EngineCore pid=8654) ERROR 08-22 20:13:29 [core.py:1349]   File "/home/enginrect/llmso/.venv/lib/python3.12/site-packages/torch/_inductor/codegen/triton.py", line 6187, in inductor_meta_common
(EngineCore pid=8654) ERROR 08-22 20:13:29 [core.py:1349]     "backend_hash": torch.utils._triton.triton_hash_with_backend(),
(EngineCore pid=8654) ERROR 08-22 20:13:29 [core.py:1349]                     ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
(EngineCore pid=8654) ERROR 08-22 20:13:29 [core.py:1349]   File "/home/enginrect/llmso/.venv/lib/python3.12/site-packages/torch/utils/_triton.py", line 253, in triton_hash_with_backend
(EngineCore pid=8654) ERROR 08-22 20:13:29 [core.py:1349]     backend = triton_backend()
(EngineCore pid=8654) ERROR 08-22 20:13:29 [core.py:1349]               ^^^^^^^^^^^^^^^^
(EngineCore pid=8654) ERROR 08-22 20:13:29 [core.py:1349]   File "/home/enginrect/llmso/.venv/lib/python3.12/site-packages/torch/utils/_triton.py", line 223, in triton_backend
(EngineCore pid=8654) ERROR 08-22 20:13:29 [core.py:1349]     target = driver.active.get_current_target()
(EngineCore pid=8654) ERROR 08-22 20:13:29 [core.py:1349]              ^^^^^^^^^^^^^
(EngineCore pid=8654) ERROR 08-22 20:13:29 [core.py:1349]   File "/home/enginrect/llmso/.venv/lib/python3.12/site-packages/triton/runtime/driver.py", line 39, in active
(EngineCore pid=8654) ERROR 08-22 20:13:29 [core.py:1349]     self._active = self.default
(EngineCore pid=8654) ERROR 08-22 20:13:29 [core.py:1349]                    ^^^^^^^^^^^^
(EngineCore pid=8654) ERROR 08-22 20:13:29 [core.py:1349]   File "/home/enginrect/llmso/.venv/lib/python3.12/site-packages/triton/runtime/driver.py", line 33, in default
(EngineCore pid=8654) ERROR 08-22 20:13:29 [core.py:1349]     self._default = _create_driver()
(EngineCore pid=8654) ERROR 08-22 20:13:29 [core.py:1349]                     ^^^^^^^^^^^^^^^^
(EngineCore pid=8654) ERROR 08-22 20:13:29 [core.py:1349]   File "/home/enginrect/llmso/.venv/lib/python3.12/site-packages/triton/runtime/driver.py", line 21, in _create_driver
(EngineCore pid=8654) ERROR 08-22 20:13:29 [core.py:1349]     return active_drivers[0]()
(EngineCore pid=8654) ERROR 08-22 20:13:29 [core.py:1349]            ^^^^^^^^^^^^^^^^^^^
(EngineCore pid=8654) ERROR 08-22 20:13:29 [core.py:1349]   File "/home/enginrect/llmso/.venv/lib/python3.12/site-packages/triton/backends/nvidia/driver.py", line 336, in __init__
(EngineCore pid=8654) ERROR 08-22 20:13:29 [core.py:1349]     self.utils = CudaUtils()  # TODO: make static
(EngineCore pid=8654) ERROR 08-22 20:13:29 [core.py:1349]                  ^^^^^^^^^^^
(EngineCore pid=8654) ERROR 08-22 20:13:29 [core.py:1349]   File "/home/enginrect/llmso/.venv/lib/python3.12/site-packages/triton/backends/nvidia/driver.py", line 66, in __init__
(EngineCore pid=8654) ERROR 08-22 20:13:29 [core.py:1349]     mod = compile_module_from_src(
(EngineCore pid=8654) ERROR 08-22 20:13:29 [core.py:1349]           ^^^^^^^^^^^^^^^^^^^^^^^^
(EngineCore pid=8654) ERROR 08-22 20:13:29 [core.py:1349]   File "/home/enginrect/llmso/.venv/lib/python3.12/site-packages/triton/runtime/build.py", line 93, in compile_module_from_src
(EngineCore pid=8654) ERROR 08-22 20:13:29 [core.py:1349]     so = _build(name, src_path, tmpdir, library_dirs or [], include_dirs or [], libraries or [], ccflags or [])
(EngineCore pid=8654) ERROR 08-22 20:13:29 [core.py:1349]          ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
(EngineCore pid=8654) ERROR 08-22 20:13:29 [core.py:1349]   File "/home/enginrect/llmso/.venv/lib/python3.12/site-packages/triton/runtime/build.py", line 32, in _build
(EngineCore pid=8654) ERROR 08-22 20:13:29 [core.py:1349]     raise RuntimeError(
(EngineCore pid=8654) ERROR 08-22 20:13:29 [core.py:1349] torch._inductor.exc.InductorError: RuntimeError: Failed to find C compiler. Please specify via CC environment variable or set triton.knobs.build.impl.
...
(EngineCore pid=8654) Traceback (most recent call last):
...
(EngineCore pid=8654)     raise e
...
(EngineCore pid=8654)     raise e.remove_dynamo_frames() from None  # see TORCHDYNAMO_VERBOSE=1
...
(EngineCore pid=8654)     raise InductorError(e, currentframe()).with_traceback(
...
(EngineCore pid=8654)     raise RuntimeError(
(EngineCore pid=8654) torch._inductor.exc.InductorError: RuntimeError: Failed to find C compiler. Please specify via CC environment variable or set triton.knobs.build.impl.
...
(APIServer pid=8545) Traceback (most recent call last):
...
(APIServer pid=8545)   File "/home/enginrect/llmso/.venv/lib/python3.12/site-packages/vllm/v1/engine/utils.py", line 1253, in wait_for_engine_startup
(APIServer pid=8545)     raise RuntimeError(
(APIServer pid=8545) RuntimeError: Engine core initialization failed. See root cause above. Failed core proc(s): {}
```

- key point: UVA 에러는 사라졌고(환경변수가 먹혔다), 다음 단계인 torch.compile(inductor)이 C 컴파일러를 요구하며 죽었다. `torch._inductor.exc.InductorError: RuntimeError: Failed to find C compiler` — Step 1에서 확인한 gcc 부재가 여기서 터진다.

### Step 6 — gcc 설치 (build-essential)

Claude Code 세션 셸은 비대화형이라 sudo 비밀번호 프롬프트를 띄울 수 없어, 비밀번호를 stdin으로 전달해 실행했다. 아래 출력에서 프롬프트 줄만 제거했다.

```shell
$ sudo apt-get update 2>&1 | tail -15
Hit:1 https://downloads.claude.ai/claude-desktop/apt/stable stable InRelease
Hit:2 http://security.ubuntu.com/ubuntu noble-security InRelease
Hit:3 https://deb.nodesource.com/node_20.x nodistro InRelease
Hit:4 http://archive.ubuntu.com/ubuntu noble InRelease
Hit:5 http://archive.ubuntu.com/ubuntu noble-updates InRelease
Hit:6 http://archive.ubuntu.com/ubuntu noble-backports InRelease
Reading package lists...
```

```shell
$ sudo apt-get install -y build-essential
Building dependency tree...
Reading state information...
...
7 upgraded, 52 newly installed, 0 to remove and 209 not upgraded.
...
Setting up gcc-14-base:amd64 (14.2.0-4ubuntu2~24.04.1) ...
...
Setting up gcc-13-base:amd64 (13.3.0-6ubuntu2~24.04.1) ...
...
Setting up gcc-13-x86-64-linux-gnu (13.3.0-6ubuntu2~24.04.1) ...
Setting up gcc-13 (13.3.0-6ubuntu2~24.04.1) ...
...
Setting up g++-13-x86-64-linux-gnu (13.3.0-6ubuntu2~24.04.1) ...
Setting up gcc-x86-64-linux-gnu (4:13.2.0-7ubuntu1) ...
Setting up gcc (4:13.2.0-7ubuntu1) ...
Setting up g++-x86-64-linux-gnu (4:13.2.0-7ubuntu1) ...
Setting up g++-13 (13.3.0-6ubuntu2~24.04.1) ...
Setting up g++ (4:13.2.0-7ubuntu1) ...
...
$ gcc --version
...
This is free software; see the source for copying conditions.  There is NO
warranty; not even for MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.

```

- key point: gcc 13.3.0 (Ubuntu 24.04 기본 계열) 설치 완료.

### 시도 3 — ❌ `Python.h: No such file or directory`

명령 (재구축 환경, 이하 동일):

```shell
$ export VLLM_WSL2_ENABLE_PIN_MEMORY=1
$ vllm serve Qwen/Qwen2.5-0.5B-Instruct --max-model-len 2048 --gpu-memory-utilization 0.85
```

출력 (발췌):

```shell
(APIServer pid=11444) INFO 08-22 20:17:34 [api_utils.py:345] 
(APIServer pid=11444) INFO 08-22 20:17:34 [api_utils.py:345]        █     █     █▄   ▄█
(APIServer pid=11444) INFO 08-22 20:17:34 [api_utils.py:345]  ▄▄ ▄█ █     █     █ ▀▄▀ █  version 0.27.1
(APIServer pid=11444) INFO 08-22 20:17:34 [api_utils.py:345]   █▄█▀ █     █     █     █  model   Qwen/Qwen2.5-0.5B-Instruct
...
(APIServer pid=11444) INFO 08-22 20:17:34 [api_utils.py:273] non-default args: {'model_tag': 'Qwen/Qwen2.5-0.5B-Instruct', 'model': 'Qwen/Qwen2.5-0.5B-Instruct', 'max_model_len': 2048, 'gpu_memory_utilization': 0.85}
...
(EngineCore pid=11560) INFO 08-22 20:17:49 [weight_utils.py:867] Filesystem type for checkpoints: EXT4. Checkpoint size: 0.92 GiB. Available RAM: 28.07 GiB.
...
(EngineCore pid=11560) INFO 08-22 20:17:49 [model_runner.py:329] Model loading took 0.93 GiB and 3.369489 seconds
...
/tmp/tmpjs_u_4yu/cuda_utils.c:9:10: fatal error: Python.h: No such file or directory
...
(EngineCore pid=11560) ERROR 08-22 20:17:53 [core.py:1349] EngineCore failed to start.
(EngineCore pid=11560) ERROR 08-22 20:17:53 [core.py:1349] Traceback (most recent call last):
(EngineCore pid=11560) ERROR 08-22 20:17:53 [core.py:1349]   File "/home/enginrect/llmso/.venv/lib/python3.12/site-packages/vllm/v1/engine/core.py", line 1318, in run_engine_core
(EngineCore pid=11560) ERROR 08-22 20:17:53 [core.py:1349]     engine_core = EngineCoreProc(*args, engine_index=dp_rank, **kwargs)
(EngineCore pid=11560) ERROR 08-22 20:17:53 [core.py:1349]                   ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
(EngineCore pid=11560) ERROR 08-22 20:17:53 [core.py:1349]   File "/home/enginrect/llmso/.venv/lib/python3.12/site-packages/vllm/tracing/otel.py", line 178, in sync_wrapper
(EngineCore pid=11560) ERROR 08-22 20:17:53 [core.py:1349]     return func(*args, **kwargs)
(EngineCore pid=11560) ERROR 08-22 20:17:53 [core.py:1349]            ^^^^^^^^^^^^^^^^^^^^^
(EngineCore pid=11560) ERROR 08-22 20:17:53 [core.py:1349]   File "/home/enginrect/llmso/.venv/lib/python3.12/site-packages/vllm/v1/engine/core.py", line 1074, in __init__
(EngineCore pid=11560) ERROR 08-22 20:17:53 [core.py:1349]     super().__init__(
(EngineCore pid=11560) ERROR 08-22 20:17:53 [core.py:1349]   File "/home/enginrect/llmso/.venv/lib/python3.12/site-packages/vllm/v1/engine/core.py", line 143, in __init__
(EngineCore pid=11560) ERROR 08-22 20:17:53 [core.py:1349]     kv_cache_config = self._initialize_kv_caches(vllm_config)
(EngineCore pid=11560) ERROR 08-22 20:17:53 [core.py:1349]                       ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
(EngineCore pid=11560) ERROR 08-22 20:17:53 [core.py:1349]   File "/home/enginrect/llmso/.venv/lib/python3.12/site-packages/vllm/tracing/otel.py", line 178, in sync_wrapper
(EngineCore pid=11560) ERROR 08-22 20:17:53 [core.py:1349]     return func(*args, **kwargs)
(EngineCore pid=11560) ERROR 08-22 20:17:53 [core.py:1349]            ^^^^^^^^^^^^^^^^^^^^^
(EngineCore pid=11560) ERROR 08-22 20:17:53 [core.py:1349]   File "/home/enginrect/llmso/.venv/lib/python3.12/site-packages/vllm/v1/engine/core.py", line 293, in _initialize_kv_caches
(EngineCore pid=11560) ERROR 08-22 20:17:53 [core.py:1349]     available_gpu_memory = self.model_executor.determine_available_memory()
(EngineCore pid=11560) ERROR 08-22 20:17:53 [core.py:1349]                            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
(EngineCore pid=11560) ERROR 08-22 20:17:53 [core.py:1349]   File "/home/enginrect/llmso/.venv/lib/python3.12/site-packages/vllm/v1/executor/abstract.py", line 147, in determine_available_memory
(EngineCore pid=11560) ERROR 08-22 20:17:53 [core.py:1349]     return self.collective_rpc("determine_available_memory")
(EngineCore pid=11560) ERROR 08-22 20:17:53 [core.py:1349]            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
(EngineCore pid=11560) ERROR 08-22 20:17:53 [core.py:1349]   File "/home/enginrect/llmso/.venv/lib/python3.12/site-packages/vllm/v1/executor/uniproc_executor.py", line 92, in collective_rpc
(EngineCore pid=11560) ERROR 08-22 20:17:53 [core.py:1349]     result = run_method(self.driver_worker, method, args, kwargs)
(EngineCore pid=11560) ERROR 08-22 20:17:53 [core.py:1349]              ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
(EngineCore pid=11560) ERROR 08-22 20:17:53 [core.py:1349]   File "/home/enginrect/llmso/.venv/lib/python3.12/site-packages/vllm/v1/serial_utils.py", line 510, in run_method
(EngineCore pid=11560) ERROR 08-22 20:17:53 [core.py:1349]     return func(*args, **kwargs)
(EngineCore pid=11560) ERROR 08-22 20:17:53 [core.py:1349]            ^^^^^^^^^^^^^^^^^^^^^
(EngineCore pid=11560) ERROR 08-22 20:17:53 [core.py:1349]   File "/home/enginrect/llmso/.venv/lib/python3.12/site-packages/torch/utils/_contextlib.py", line 124, in decorate_context
(EngineCore pid=11560) ERROR 08-22 20:17:53 [core.py:1349]     return func(*args, **kwargs)
(EngineCore pid=11560) ERROR 08-22 20:17:53 [core.py:1349]            ^^^^^^^^^^^^^^^^^^^^^
(EngineCore pid=11560) ERROR 08-22 20:17:53 [core.py:1349]   File "/home/enginrect/llmso/.venv/lib/python3.12/site-packages/vllm/v1/worker/gpu_worker.py", line 504, in determine_available_memory
(EngineCore pid=11560) ERROR 08-22 20:17:53 [core.py:1349]     self.model_runner.profile_run()
(EngineCore pid=11560) ERROR 08-22 20:17:53 [core.py:1349]   File "/home/enginrect/llmso/.venv/lib/python3.12/site-packages/torch/utils/_contextlib.py", line 124, in decorate_context
(EngineCore pid=11560) ERROR 08-22 20:17:53 [core.py:1349]     return func(*args, **kwargs)
(EngineCore pid=11560) ERROR 08-22 20:17:53 [core.py:1349]            ^^^^^^^^^^^^^^^^^^^^^
(EngineCore pid=11560) ERROR 08-22 20:17:53 [core.py:1349]   File "/home/enginrect/llmso/.venv/lib/python3.12/site-packages/vllm/v1/worker/gpu/model_runner.py", line 719, in profile_run
(EngineCore pid=11560) ERROR 08-22 20:17:53 [core.py:1349]     hidden_states, sample_hidden_states = self._dummy_run(
(EngineCore pid=11560) ERROR 08-22 20:17:53 [core.py:1349]                                           ^^^^^^^^^^^^^^^^
(EngineCore pid=11560) ERROR 08-22 20:17:53 [core.py:1349]   File "/home/enginrect/llmso/.venv/lib/python3.12/site-packages/torch/utils/_contextlib.py", line 124, in decorate_context
(EngineCore pid=11560) ERROR 08-22 20:17:53 [core.py:1349]     return func(*args, **kwargs)
(EngineCore pid=11560) ERROR 08-22 20:17:53 [core.py:1349]            ^^^^^^^^^^^^^^^^^^^^^
(EngineCore pid=11560) ERROR 08-22 20:17:53 [core.py:1349]   File "/home/enginrect/llmso/.venv/lib/python3.12/site-packages/vllm/v1/worker/gpu/eplb_utils.py", line 38, in wrapper
(EngineCore pid=11560) ERROR 08-22 20:17:53 [core.py:1349]     result = fn(self, *args, **kwargs)
(EngineCore pid=11560) ERROR 08-22 20:17:53 [core.py:1349]              ^^^^^^^^^^^^^^^^^^^^^^^^^
(EngineCore pid=11560) ERROR 08-22 20:17:53 [core.py:1349]   File "/home/enginrect/llmso/.venv/lib/python3.12/site-packages/vllm/v1/worker/gpu/model_runner.py", line 613, in _dummy_run
(EngineCore pid=11560) ERROR 08-22 20:17:53 [core.py:1349]     self.execute_model(
(EngineCore pid=11560) ERROR 08-22 20:17:53 [core.py:1349]   File "/home/enginrect/llmso/.venv/lib/python3.12/site-packages/torch/utils/_contextlib.py", line 124, in decorate_context
(EngineCore pid=11560) ERROR 08-22 20:17:53 [core.py:1349]     return func(*args, **kwargs)
(EngineCore pid=11560) ERROR 08-22 20:17:53 [core.py:1349]            ^^^^^^^^^^^^^^^^^^^^^
(EngineCore pid=11560) ERROR 08-22 20:17:53 [core.py:1349]   File "/home/enginrect/llmso/.venv/lib/python3.12/site-packages/vllm/v1/worker/gpu/model_runner.py", line 1423, in execute_model
(EngineCore pid=11560) ERROR 08-22 20:17:53 [core.py:1349]     model_output = self.model(**model_inputs)
(EngineCore pid=11560) ERROR 08-22 20:17:53 [core.py:1349]                    ^^^^^^^^^^^^^^^^^^^^^^^^^^
(EngineCore pid=11560) ERROR 08-22 20:17:53 [core.py:1349]   File "/home/enginrect/llmso/.venv/lib/python3.12/site-packages/torch/nn/modules/module.py", line 1778, in _wrapped_call_impl
(EngineCore pid=11560) ERROR 08-22 20:17:53 [core.py:1349]     return self._call_impl(*args, **kwargs)
(EngineCore pid=11560) ERROR 08-22 20:17:53 [core.py:1349]            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
(EngineCore pid=11560) ERROR 08-22 20:17:53 [core.py:1349]   File "/home/enginrect/llmso/.venv/lib/python3.12/site-packages/torch/nn/modules/module.py", line 1789, in _call_impl
(EngineCore pid=11560) ERROR 08-22 20:17:53 [core.py:1349]     return forward_call(*args, **kwargs)
(EngineCore pid=11560) ERROR 08-22 20:17:53 [core.py:1349]            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
(EngineCore pid=11560) ERROR 08-22 20:17:53 [core.py:1349]   File "/home/enginrect/llmso/.venv/lib/python3.12/site-packages/vllm/model_executor/models/qwen2.py", line 491, in forward
(EngineCore pid=11560) ERROR 08-22 20:17:53 [core.py:1349]     hidden_states = self.model(
(EngineCore pid=11560) ERROR 08-22 20:17:53 [core.py:1349]                     ^^^^^^^^^^^
(EngineCore pid=11560) ERROR 08-22 20:17:53 [core.py:1349]   File "/home/enginrect/llmso/.venv/lib/python3.12/site-packages/vllm/compilation/decorators.py", line 663, in __call__
(EngineCore pid=11560) ERROR 08-22 20:17:53 [core.py:1349]     self.aot_compiled_fn = self.aot_compile(*args, **kwargs)
(EngineCore pid=11560) ERROR 08-22 20:17:53 [core.py:1349]                            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
(EngineCore pid=11560) ERROR 08-22 20:17:53 [core.py:1349]   File "/home/enginrect/llmso/.venv/lib/python3.12/site-packages/vllm/compilation/wrapper.py", line 169, in aot_compile
(EngineCore pid=11560) ERROR 08-22 20:17:53 [core.py:1349]     return self._compiled_callable.aot_compile((args, kwargs))
(EngineCore pid=11560) ERROR 08-22 20:17:53 [core.py:1349]            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
(EngineCore pid=11560) ERROR 08-22 20:17:53 [core.py:1349]   File "/home/enginrect/llmso/.venv/lib/python3.12/site-packages/torch/_dynamo/eval_frame.py", line 964, in aot_compile
(EngineCore pid=11560) ERROR 08-22 20:17:53 [core.py:1349]     return aot_compile_fullgraph(
(EngineCore pid=11560) ERROR 08-22 20:17:53 [core.py:1349]            ^^^^^^^^^^^^^^^^^^^^^^
(EngineCore pid=11560) ERROR 08-22 20:17:53 [core.py:1349]   File "/home/enginrect/llmso/.venv/lib/python3.12/site-packages/torch/_dynamo/aot_compile.py", line 396, in aot_compile_fullgraph
(EngineCore pid=11560) ERROR 08-22 20:17:53 [core.py:1349]     compiled_fn = backend(
(EngineCore pid=11560) ERROR 08-22 20:17:53 [core.py:1349]                   ^^^^^^^^
(EngineCore pid=11560) ERROR 08-22 20:17:53 [core.py:1349]   File "/home/enginrect/llmso/.venv/lib/python3.12/site-packages/torch/__init__.py", line 2568, in __call__
(EngineCore pid=11560) ERROR 08-22 20:17:53 [core.py:1349]     return self.compiler_fn(model_, inputs_, **self.kwargs)
(EngineCore pid=11560) ERROR 08-22 20:17:53 [core.py:1349]            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
(EngineCore pid=11560) ERROR 08-22 20:17:53 [core.py:1349]   File "/usr/lib/python3.12/contextlib.py", line 81, in inner
(EngineCore pid=11560) ERROR 08-22 20:17:53 [core.py:1349]     return func(*args, **kwds)
(EngineCore pid=11560) ERROR 08-22 20:17:53 [core.py:1349]            ^^^^^^^^^^^^^^^^^^^
(EngineCore pid=11560) ERROR 08-22 20:17:53 [core.py:1349]   File "/home/enginrect/llmso/.venv/lib/python3.12/site-packages/vllm/compilation/backends.py", line 1222, in __call__
(EngineCore pid=11560) ERROR 08-22 20:17:53 [core.py:1349]     PiecewiseCompileInterpreter(
(EngineCore pid=11560) ERROR 08-22 20:17:53 [core.py:1349]   File "/home/enginrect/llmso/.venv/lib/python3.12/site-packages/vllm/tracing/otel.py", line 178, in sync_wrapper
(EngineCore pid=11560) ERROR 08-22 20:17:53 [core.py:1349]     return func(*args, **kwargs)
(EngineCore pid=11560) ERROR 08-22 20:17:53 [core.py:1349]            ^^^^^^^^^^^^^^^^^^^^^
(EngineCore pid=11560) ERROR 08-22 20:17:53 [core.py:1349]   File "/home/enginrect/llmso/.venv/lib/python3.12/site-packages/vllm/compilation/backends.py", line 728, in run
(EngineCore pid=11560) ERROR 08-22 20:17:53 [core.py:1349]     return super().run(*args)
(EngineCore pid=11560) ERROR 08-22 20:17:53 [core.py:1349]            ^^^^^^^^^^^^^^^^^^
(EngineCore pid=11560) ERROR 08-22 20:17:53 [core.py:1349]   File "/home/enginrect/llmso/.venv/lib/python3.12/site-packages/torch/fx/interpreter.py", line 197, in run
(EngineCore pid=11560) ERROR 08-22 20:17:53 [core.py:1349]     self.env[node] = self.run_node(node)
(EngineCore pid=11560) ERROR 08-22 20:17:53 [core.py:1349]                      ^^^^^^^^^^^^^^^^^^^
(EngineCore pid=11560) ERROR 08-22 20:17:53 [core.py:1349]   File "/home/enginrect/llmso/.venv/lib/python3.12/site-packages/torch/fx/interpreter.py", line 294, in run_node
(EngineCore pid=11560) ERROR 08-22 20:17:53 [core.py:1349]     return getattr(self, n.op)(n.target, args, kwargs)
(EngineCore pid=11560) ERROR 08-22 20:17:53 [core.py:1349]            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
(EngineCore pid=11560) ERROR 08-22 20:17:53 [core.py:1349]   File "/home/enginrect/llmso/.venv/lib/python3.12/site-packages/vllm/compilation/backends.py", line 755, in call_module
(EngineCore pid=11560) ERROR 08-22 20:17:53 [core.py:1349]     piecewise_backend = PiecewiseBackend(
(EngineCore pid=11560) ERROR 08-22 20:17:53 [core.py:1349]                         ^^^^^^^^^^^^^^^^^
(EngineCore pid=11560) ERROR 08-22 20:17:53 [core.py:1349]   File "/home/enginrect/llmso/.venv/lib/python3.12/site-packages/vllm/compilation/piecewise_backend.py", line 190, in __init__
(EngineCore pid=11560) ERROR 08-22 20:17:53 [core.py:1349]     self.compile_all_ranges()
(EngineCore pid=11560) ERROR 08-22 20:17:53 [core.py:1349]   File "/home/enginrect/llmso/.venv/lib/python3.12/site-packages/vllm/compilation/piecewise_backend.py", line 266, in compile_all_ranges
(EngineCore pid=11560) ERROR 08-22 20:17:53 [core.py:1349]     range_entry.runnable = self.vllm_backend.compiler_manager.compile(
(EngineCore pid=11560) ERROR 08-22 20:17:53 [core.py:1349]                            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
(EngineCore pid=11560) ERROR 08-22 20:17:53 [core.py:1349]   File "/home/enginrect/llmso/.venv/lib/python3.12/site-packages/vllm/tracing/otel.py", line 178, in sync_wrapper
(EngineCore pid=11560) ERROR 08-22 20:17:53 [core.py:1349]     return func(*args, **kwargs)
(EngineCore pid=11560) ERROR 08-22 20:17:53 [core.py:1349]            ^^^^^^^^^^^^^^^^^^^^^
(EngineCore pid=11560) ERROR 08-22 20:17:53 [core.py:1349]   File "/home/enginrect/llmso/.venv/lib/python3.12/site-packages/vllm/compilation/backends.py", line 353, in compile
(EngineCore pid=11560) ERROR 08-22 20:17:53 [core.py:1349]     compiled_graph, handle = self.compiler.compile(
(EngineCore pid=11560) ERROR 08-22 20:17:53 [core.py:1349]                              ^^^^^^^^^^^^^^^^^^^^^^
(EngineCore pid=11560) ERROR 08-22 20:17:53 [core.py:1349]   File "/home/enginrect/llmso/.venv/lib/python3.12/site-packages/vllm/compilation/compiler_interface.py", line 376, in compile
(EngineCore pid=11560) ERROR 08-22 20:17:53 [core.py:1349]     compiled_graph = standalone_compile(graph, example_inputs, **compile_kwargs)
(EngineCore pid=11560) ERROR 08-22 20:17:53 [core.py:1349]                      ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
(EngineCore pid=11560) ERROR 08-22 20:17:53 [core.py:1349]   File "/home/enginrect/llmso/.venv/lib/python3.12/site-packages/torch/_inductor/__init__.py", line 452, in standalone_compile
(EngineCore pid=11560) ERROR 08-22 20:17:53 [core.py:1349]     return standalone_compile(
(EngineCore pid=11560) ERROR 08-22 20:17:53 [core.py:1349]            ^^^^^^^^^^^^^^^^^^^
(EngineCore pid=11560) ERROR 08-22 20:17:53 [core.py:1349]   File "/home/enginrect/llmso/.venv/lib/python3.12/site-packages/torch/_inductor/standalone_compile.py", line 510, in standalone_compile
(EngineCore pid=11560) ERROR 08-22 20:17:53 [core.py:1349]     compiled_fn = compile_fx(
(EngineCore pid=11560) ERROR 08-22 20:17:53 [core.py:1349]                   ^^^^^^^^^^^
(EngineCore pid=11560) ERROR 08-22 20:17:53 [core.py:1349]   File "/home/enginrect/llmso/.venv/lib/python3.12/site-packages/torch/_inductor/compile_fx.py", line 2723, in compile_fx
(EngineCore pid=11560) ERROR 08-22 20:17:53 [core.py:1349]     return compile_fx(
(EngineCore pid=11560) ERROR 08-22 20:17:53 [core.py:1349]            ^^^^^^^^^^^
(EngineCore pid=11560) ERROR 08-22 20:17:53 [core.py:1349]   File "/home/enginrect/llmso/.venv/lib/python3.12/site-packages/torch/_inductor/compile_fx.py", line 2782, in compile_fx
(EngineCore pid=11560) ERROR 08-22 20:17:53 [core.py:1349]     return _maybe_wrap_and_compile_fx_main(
(EngineCore pid=11560) ERROR 08-22 20:17:53 [core.py:1349]            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
(EngineCore pid=11560) ERROR 08-22 20:17:53 [core.py:1349]   File "/home/enginrect/llmso/.venv/lib/python3.12/site-packages/torch/_inductor/compile_fx.py", line 2863, in _maybe_wrap_and_compile_fx_main
(EngineCore pid=11560) ERROR 08-22 20:17:53 [core.py:1349]     return _compile_fx_main(
(EngineCore pid=11560) ERROR 08-22 20:17:53 [core.py:1349]            ^^^^^^^^^^^^^^^^^
(EngineCore pid=11560) ERROR 08-22 20:17:53 [core.py:1349]   File "/home/enginrect/llmso/.venv/lib/python3.12/site-packages/torch/_inductor/compile_fx.py", line 3076, in _compile_fx_main
(EngineCore pid=11560) ERROR 08-22 20:17:53 [core.py:1349]     raise e.remove_dynamo_frames() from None  # see TORCHDYNAMO_VERBOSE=1
(EngineCore pid=11560) ERROR 08-22 20:17:53 [core.py:1349]     ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
(EngineCore pid=11560) ERROR 08-22 20:17:53 [core.py:1349]   File "/home/enginrect/llmso/.venv/lib/python3.12/site-packages/torch/_inductor/compile_fx.py", line 1020, in _compile_fx_inner
(EngineCore pid=11560) ERROR 08-22 20:17:53 [core.py:1349]     raise InductorError(e, currentframe()).with_traceback(
(EngineCore pid=11560) ERROR 08-22 20:17:53 [core.py:1349]   File "/home/enginrect/llmso/.venv/lib/python3.12/site-packages/torch/_inductor/compile_fx.py", line 1004, in _compile_fx_inner
(EngineCore pid=11560) ERROR 08-22 20:17:53 [core.py:1349]     mb_compiled_graph = fx_codegen_and_compile(
(EngineCore pid=11560) ERROR 08-22 20:17:53 [core.py:1349]                         ^^^^^^^^^^^^^^^^^^^^^^^
(EngineCore pid=11560) ERROR 08-22 20:17:53 [core.py:1349]   File "/home/enginrect/llmso/.venv/lib/python3.12/site-packages/torch/_inductor/compile_fx.py", line 1847, in fx_codegen_and_compile
(EngineCore pid=11560) ERROR 08-22 20:17:53 [core.py:1349]     return scheme.codegen_and_compile(gm, example_inputs, inputs_to_check, graph_kwargs)
(EngineCore pid=11560) ERROR 08-22 20:17:53 [core.py:1349]            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
(EngineCore pid=11560) ERROR 08-22 20:17:53 [core.py:1349]   File "/home/enginrect/llmso/.venv/lib/python3.12/site-packages/torch/_inductor/compile_fx.py", line 1608, in codegen_and_compile
(EngineCore pid=11560) ERROR 08-22 20:17:53 [core.py:1349]     compiled_module = graph.compile_to_module()
(EngineCore pid=11560) ERROR 08-22 20:17:53 [core.py:1349]                       ^^^^^^^^^^^^^^^^^^^^^^^^^
(EngineCore pid=11560) ERROR 08-22 20:17:53 [core.py:1349]   File "/home/enginrect/llmso/.venv/lib/python3.12/site-packages/torch/_inductor/graph.py", line 2669, in compile_to_module
(EngineCore pid=11560) ERROR 08-22 20:17:53 [core.py:1349]     return self._compile_to_module()
(EngineCore pid=11560) ERROR 08-22 20:17:53 [core.py:1349]            ^^^^^^^^^^^^^^^^^^^^^^^^^
(EngineCore pid=11560) ERROR 08-22 20:17:53 [core.py:1349]   File "/home/enginrect/llmso/.venv/lib/python3.12/site-packages/torch/_inductor/graph.py", line 2675, in _compile_to_module
(EngineCore pid=11560) ERROR 08-22 20:17:53 [core.py:1349]     self.codegen_with_cpp_wrapper() if self.cpp_wrapper else self.codegen()
(EngineCore pid=11560) ERROR 08-22 20:17:53 [core.py:1349]                                                              ^^^^^^^^^^^^^^
(EngineCore pid=11560) ERROR 08-22 20:17:53 [core.py:1349]   File "/home/enginrect/llmso/.venv/lib/python3.12/site-packages/torch/_inductor/graph.py", line 2607, in codegen
(EngineCore pid=11560) ERROR 08-22 20:17:53 [core.py:1349]     self._update_scheduler()
(EngineCore pid=11560) ERROR 08-22 20:17:53 [core.py:1349]   File "/home/enginrect/llmso/.venv/lib/python3.12/site-packages/torch/_inductor/graph.py", line 2601, in _update_scheduler
(EngineCore pid=11560) ERROR 08-22 20:17:53 [core.py:1349]     self.scheduler = Scheduler(self.operations)
(EngineCore pid=11560) ERROR 08-22 20:17:53 [core.py:1349]                      ^^^^^^^^^^^^^^^^^^^^^^^^^^
(EngineCore pid=11560) ERROR 08-22 20:17:53 [core.py:1349]   File "/home/enginrect/llmso/.venv/lib/python3.12/site-packages/torch/_inductor/scheduler.py", line 4036, in __init__
(EngineCore pid=11560) ERROR 08-22 20:17:53 [core.py:1349]     self._init(nodes)
(EngineCore pid=11560) ERROR 08-22 20:17:53 [core.py:1349]   File "/home/enginrect/llmso/.venv/lib/python3.12/site-packages/torch/_inductor/scheduler.py", line 4165, in _init
(EngineCore pid=11560) ERROR 08-22 20:17:53 [core.py:1349]     self.create_combo_kernel_nodes(num_ck_nodes=None)
(EngineCore pid=11560) ERROR 08-22 20:17:53 [core.py:1349]   File "/home/enginrect/llmso/.venv/lib/python3.12/site-packages/torch/_inductor/scheduler.py", line 6141, in create_combo_kernel_nodes
(EngineCore pid=11560) ERROR 08-22 20:17:53 [core.py:1349]     if len(window) < 2 or not self.speedup_by_combo_kernel(window):
(EngineCore pid=11560) ERROR 08-22 20:17:53 [core.py:1349]                               ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
(EngineCore pid=11560) ERROR 08-22 20:17:53 [core.py:1349]   File "/home/enginrect/llmso/.venv/lib/python3.12/site-packages/torch/_inductor/scheduler.py", line 9592, in speedup_by_combo_kernel
(EngineCore pid=11560) ERROR 08-22 20:17:53 [core.py:1349]     ms, path = self.benchmark_fused_nodes(node_list)
(EngineCore pid=11560) ERROR 08-22 20:17:53 [core.py:1349]                ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
(EngineCore pid=11560) ERROR 08-22 20:17:53 [core.py:1349]   File "/home/enginrect/llmso/.venv/lib/python3.12/site-packages/torch/_inductor/scheduler.py", line 5048, in benchmark_fused_nodes
(EngineCore pid=11560) ERROR 08-22 20:17:53 [core.py:1349]     return backend.benchmark_fused_nodes(nodes)
(EngineCore pid=11560) ERROR 08-22 20:17:53 [core.py:1349]            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
(EngineCore pid=11560) ERROR 08-22 20:17:53 [core.py:1349]   File "/home/enginrect/llmso/.venv/lib/python3.12/site-packages/torch/_inductor/codegen/cuda_combined_scheduling.py", line 169, in benchmark_fused_nodes
(EngineCore pid=11560) ERROR 08-22 20:17:53 [core.py:1349]     return self._triton_scheduling.benchmark_fused_nodes(nodes)
(EngineCore pid=11560) ERROR 08-22 20:17:53 [core.py:1349]            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
(EngineCore pid=11560) ERROR 08-22 20:17:53 [core.py:1349]   File "/home/enginrect/llmso/.venv/lib/python3.12/site-packages/torch/_inductor/codegen/triton.py", line 7325, in benchmark_fused_nodes
(EngineCore pid=11560) ERROR 08-22 20:17:53 [core.py:1349]     src_code = self.generate_kernel_code_from_nodes(nodes, benchmark_kernel=True)
(EngineCore pid=11560) ERROR 08-22 20:17:53 [core.py:1349]                ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
(EngineCore pid=11560) ERROR 08-22 20:17:53 [core.py:1349]   File "/home/enginrect/llmso/.venv/lib/python3.12/site-packages/torch/_inductor/codegen/simd.py", line 4510, in generate_kernel_code_from_nodes
(EngineCore pid=11560) ERROR 08-22 20:17:53 [core.py:1349]     src_code = kernel.codegen_kernel()
(EngineCore pid=11560) ERROR 08-22 20:17:53 [core.py:1349]                ^^^^^^^^^^^^^^^^^^^^^^^
(EngineCore pid=11560) ERROR 08-22 20:17:53 [core.py:1349]   File "/home/enginrect/llmso/.venv/lib/python3.12/site-packages/torch/_inductor/codegen/triton.py", line 6477, in codegen_kernel
(EngineCore pid=11560) ERROR 08-22 20:17:53 [core.py:1349]     **self.inductor_meta_common(),
(EngineCore pid=11560) ERROR 08-22 20:17:53 [core.py:1349]       ^^^^^^^^^^^^^^^^^^^^^^^^^^^
(EngineCore pid=11560) ERROR 08-22 20:17:53 [core.py:1349]   File "/home/enginrect/llmso/.venv/lib/python3.12/site-packages/torch/_inductor/codegen/triton.py", line 6187, in inductor_meta_common
(EngineCore pid=11560) ERROR 08-22 20:17:53 [core.py:1349]     "backend_hash": torch.utils._triton.triton_hash_with_backend(),
(EngineCore pid=11560) ERROR 08-22 20:17:53 [core.py:1349]                     ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
(EngineCore pid=11560) ERROR 08-22 20:17:53 [core.py:1349]   File "/home/enginrect/llmso/.venv/lib/python3.12/site-packages/torch/utils/_triton.py", line 253, in triton_hash_with_backend
(EngineCore pid=11560) ERROR 08-22 20:17:53 [core.py:1349]     backend = triton_backend()
(EngineCore pid=11560) ERROR 08-22 20:17:53 [core.py:1349]               ^^^^^^^^^^^^^^^^
(EngineCore pid=11560) ERROR 08-22 20:17:53 [core.py:1349]   File "/home/enginrect/llmso/.venv/lib/python3.12/site-packages/torch/utils/_triton.py", line 223, in triton_backend
(EngineCore pid=11560) ERROR 08-22 20:17:53 [core.py:1349]     target = driver.active.get_current_target()
(EngineCore pid=11560) ERROR 08-22 20:17:53 [core.py:1349]              ^^^^^^^^^^^^^
(EngineCore pid=11560) ERROR 08-22 20:17:53 [core.py:1349]   File "/home/enginrect/llmso/.venv/lib/python3.12/site-packages/triton/runtime/driver.py", line 39, in active
(EngineCore pid=11560) ERROR 08-22 20:17:53 [core.py:1349]     self._active = self.default
(EngineCore pid=11560) ERROR 08-22 20:17:53 [core.py:1349]                    ^^^^^^^^^^^^
(EngineCore pid=11560) ERROR 08-22 20:17:53 [core.py:1349]   File "/home/enginrect/llmso/.venv/lib/python3.12/site-packages/triton/runtime/driver.py", line 33, in default
(EngineCore pid=11560) ERROR 08-22 20:17:53 [core.py:1349]     self._default = _create_driver()
(EngineCore pid=11560) ERROR 08-22 20:17:53 [core.py:1349]                     ^^^^^^^^^^^^^^^^
(EngineCore pid=11560) ERROR 08-22 20:17:53 [core.py:1349]   File "/home/enginrect/llmso/.venv/lib/python3.12/site-packages/triton/runtime/driver.py", line 21, in _create_driver
(EngineCore pid=11560) ERROR 08-22 20:17:53 [core.py:1349]     return active_drivers[0]()
(EngineCore pid=11560) ERROR 08-22 20:17:53 [core.py:1349]            ^^^^^^^^^^^^^^^^^^^
(EngineCore pid=11560) ERROR 08-22 20:17:53 [core.py:1349]   File "/home/enginrect/llmso/.venv/lib/python3.12/site-packages/triton/backends/nvidia/driver.py", line 336, in __init__
(EngineCore pid=11560) ERROR 08-22 20:17:53 [core.py:1349]     self.utils = CudaUtils()  # TODO: make static
(EngineCore pid=11560) ERROR 08-22 20:17:53 [core.py:1349]                  ^^^^^^^^^^^
(EngineCore pid=11560) ERROR 08-22 20:17:53 [core.py:1349]   File "/home/enginrect/llmso/.venv/lib/python3.12/site-packages/triton/backends/nvidia/driver.py", line 66, in __init__
(EngineCore pid=11560) ERROR 08-22 20:17:53 [core.py:1349]     mod = compile_module_from_src(
(EngineCore pid=11560) ERROR 08-22 20:17:53 [core.py:1349]           ^^^^^^^^^^^^^^^^^^^^^^^^
(EngineCore pid=11560) ERROR 08-22 20:17:53 [core.py:1349]   File "/home/enginrect/llmso/.venv/lib/python3.12/site-packages/triton/runtime/build.py", line 93, in compile_module_from_src
(EngineCore pid=11560) ERROR 08-22 20:17:53 [core.py:1349]     so = _build(name, src_path, tmpdir, library_dirs or [], include_dirs or [], libraries or [], ccflags or [])
(EngineCore pid=11560) ERROR 08-22 20:17:53 [core.py:1349]          ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
(EngineCore pid=11560) ERROR 08-22 20:17:53 [core.py:1349]   File "/home/enginrect/llmso/.venv/lib/python3.12/site-packages/triton/runtime/build.py", line 48, in _build
(EngineCore pid=11560) ERROR 08-22 20:17:53 [core.py:1349]     subprocess.check_call(cc_cmd, stdout=subprocess.DEVNULL)
(EngineCore pid=11560) ERROR 08-22 20:17:53 [core.py:1349]   File "/usr/lib/python3.12/subprocess.py", line 413, in check_call
(EngineCore pid=11560) ERROR 08-22 20:17:53 [core.py:1349]     raise CalledProcessError(retcode, cmd)
(EngineCore pid=11560) ERROR 08-22 20:17:53 [core.py:1349] torch._inductor.exc.InductorError: CalledProcessError: Command '['/usr/bin/gcc', '/tmp/tmpjs_u_4yu/cuda_utils.c', '-O3', '-shared', '-fPIC', '-Wno-psabi', '-o', '/tmp/tmpjs_u_4yu/cuda_utils.cpython-312-x86_64-linux-gnu.so', '-l:libcuda.so.1', '-L/home/enginrect/llmso/.venv/lib/python3.12/site-packages/triton/backends/nvidia/lib', '-L/usr/lib/wsl/lib', '-I/home/enginrect/llmso/.venv/lib/python3.12/site-packages/triton/backends/nvidia/include', '-I/tmp/tmpjs_u_4yu', '-I/usr/include/python3.12']' returned non-zero exit status 1.
...
(EngineCore pid=11560) Traceback (most recent call last):
...
(EngineCore pid=11560)     raise e
...
(EngineCore pid=11560)     raise e.remove_dynamo_frames() from None  # see TORCHDYNAMO_VERBOSE=1
...
(EngineCore pid=11560)     raise InductorError(e, currentframe()).with_traceback(
...
(EngineCore pid=11560)     raise CalledProcessError(retcode, cmd)
(EngineCore pid=11560) torch._inductor.exc.InductorError: CalledProcessError: Command '['/usr/bin/gcc', '/tmp/tmpjs_u_4yu/cuda_utils.c', '-O3', '-shared', '-fPIC', '-Wno-psabi', '-o', '/tmp/tmpjs_u_4yu/cuda_utils.cpython-312-x86_64-linux-gnu.so', '-l:libcuda.so.1', '-L/home/enginrect/llmso/.venv/lib/python3.12/site-packages/triton/backends/nvidia/lib', '-L/usr/lib/wsl/lib', '-I/home/enginrect/llmso/.venv/lib/python3.12/site-packages/triton/backends/nvidia/include', '-I/tmp/tmpjs_u_4yu', '-I/usr/include/python3.12']' returned non-zero exit status 1.
...
(APIServer pid=11444) Traceback (most recent call last):
...
(APIServer pid=11444)   File "/home/enginrect/llmso/.venv/lib/python3.12/site-packages/vllm/v1/engine/utils.py", line 1253, in wait_for_engine_startup
(APIServer pid=11444)     raise RuntimeError(
(APIServer pid=11444) RuntimeError: Engine core initialization failed. See root cause above. Failed core proc(s): {}
```

원인 확인:

```shell
$ ls /usr/include/python3.12/Python.h
Python.h 없음
$ grep -B2 -A8 "cuda_utils.c" (serve 로그) | grep -E "error|fatal"
/tmp/tmpjs_u_4yu/cuda_utils.c:9:10: fatal error: Python.h: No such file or directory
```

- key point: gcc는 생겼지만 triton이 컴파일하는 `cuda_utils.c`가 Python C 헤더를 include한다. `python3.12-dev`가 없으면 여기서 죽는다.

### Step 7 — python3.12-dev 설치

```shell
$ sudo apt-get install -y python3.12-dev
Setting up zlib1g-dev:amd64 (1:1.3.dfsg-3.1ubuntu2.1) ...
Setting up python3.12-minimal (3.12.3-1ubuntu0.15) ...
Setting up libpython3.12-stdlib:amd64 (3.12.3-1ubuntu0.15) ...
Setting up python3.12 (3.12.3-1ubuntu0.15) ...
Setting up libpython3.12t64:amd64 (3.12.3-1ubuntu0.15) ...
Setting up libpython3.12-dev:amd64 (3.12.3-1ubuntu0.15) ...
Setting up python3.12-dev (3.12.3-1ubuntu0.15) ...
Processing triggers for systemd (255.4-1ubuntu8.4) ...
Processing triggers for man-db (2.12.0-4build2) ...
Processing triggers for libc-bin (2.39-0ubuntu8.8) ...

$ ls /usr/include/python3.12/Python.h
/usr/include/python3.12/Python.h
```

- key point: `/usr/include/python3.12/Python.h` 확보.

### Step 8 — 기본 도구 정비 (pip / npm)

vLLM과 직접 관련은 없지만 이후 실습(스크립트, 노트북, 도구 설치)에 필요한 기본기를 이 시점에 정비했다.

```shell
$ sudo apt-get install -y python3-pip
running python post-rtupdate hooks for python3.12...
Setting up python3-wheel (0.42.0-2) ...
Setting up python3-pip (24.0+dfsg-1ubuntu1.3) ...
Setting up libjs-sphinxdoc (7.2.6-6) ...
Setting up python3-dev (3.12.3-0ubuntu2.1) ...
Processing triggers for man-db (2.12.0-4build2) ...

$ pip3 --version
pip 24.0 from /usr/lib/python3/dist-packages/pip (python 3.12)
```

```shell
$ npm config set prefix ~/.npm-global && PATH 추가
npm prefix: /home/enginrect/.npm-global

$ node --version && npm --version
v20.20.2
10.8.2
```

- key point: pip 24.0 설치. node v20.20.2/npm 10.8.2는 이미 있었고(nodesource repo), npm 전역 prefix를 `~/.npm-global`로 바꿔 이후 `npm i -g`에 sudo가 필요 없게 했다.

### Step 9 — 환경변수 영속화 (~/.bashrc)

```shell
$ ~/.bashrc 정리 (uv PATH / VLLM_WSL2_ENABLE_PIN_MEMORY)
uv PATH: 이미 있음
PIN_MEMORY: 추가함

$ tail -5 ~/.bashrc
fi

. "$HOME/.local/bin/env"
export PATH="$HOME/.npm-global/bin:$PATH"
export VLLM_WSL2_ENABLE_PIN_MEMORY=1
```

- key point: uv PATH(설치 스크립트가 이미 추가), npm prefix, `VLLM_WSL2_ENABLE_PIN_MEMORY=1`이 `.bashrc`에 들어갔다. `CUDA_HOME`은 시도 4를 겪은 뒤 추가된다(Step 10).

### 시도 4 — ❌ `Could not find nvcc and default cuda_home='/usr/local/cuda' doesn't exist`

출력 (발췌):

```shell
(APIServer pid=12654) INFO 08-22 20:19:51 [api_utils.py:345] 
(APIServer pid=12654) INFO 08-22 20:19:51 [api_utils.py:345]        █     █     █▄   ▄█
(APIServer pid=12654) INFO 08-22 20:19:51 [api_utils.py:345]  ▄▄ ▄█ █     █     █ ▀▄▀ █  version 0.27.1
(APIServer pid=12654) INFO 08-22 20:19:51 [api_utils.py:345]   █▄█▀ █     █     █     █  model   Qwen/Qwen2.5-0.5B-Instruct
...
(APIServer pid=12654) INFO 08-22 20:19:51 [api_utils.py:273] non-default args: {'model_tag': 'Qwen/Qwen2.5-0.5B-Instruct', 'model': 'Qwen/Qwen2.5-0.5B-Instruct', 'max_model_len': 2048, 'gpu_memory_utilization': 0.85}
...
(EngineCore pid=12764) INFO 08-22 20:20:04 [weight_utils.py:867] Filesystem type for checkpoints: EXT4. Checkpoint size: 0.92 GiB. Available RAM: 27.42 GiB.
...
(EngineCore pid=12764) INFO 08-22 20:20:05 [model_runner.py:329] Model loading took 0.93 GiB and 2.078159 seconds
...
(EngineCore pid=12764) INFO 08-22 20:20:14 [gpu_worker.py:563] Available KV cache memory: 18.65 GiB
(EngineCore pid=12764) INFO 08-22 20:20:14 [kv_cache_utils.py:2235] GPU KV cache size: 1,629,520 tokens
...
(EngineCore pid=12764) WARNING 08-22 20:20:14 [import_utils.py:408] Module vllm.third_party.deep_gemm was found but failed to import
(EngineCore pid=12764) WARNING 08-22 20:20:14 [import_utils.py:408] Traceback (most recent call last):
...
(EngineCore pid=12764) WARNING 08-22 20:20:14 [import_utils.py:408] AssertionError
...
(EngineCore pid=12764) INFO 08-22 20:20:16 [model_runner.py:791] Graph capturing finished in 2 secs, took 0.37 GiB
(EngineCore pid=12764) INFO 08-22 20:20:16 [gpu_worker.py:789] Free memory on device (22.45/23.99 GiB) on startup. Desired GPU memory utilization is (0.85, 20.39 GiB). Actual usage is 1.49 GiB for consumed memory (weights + non-torch), 0.25 GiB for peak activation, and 0.37 GiB for CUDAGraph memory. Replace gpu_memory_utilization config with `--kv-cache-memory=19468210074` (18.13 GiB) to fit into requested memory, or `--kv-cache-memory=21679683072` (20.19 GiB) to fully utilize gpu memory. Current kv cache memory in use is 18.65 GiB.
(EngineCore pid=12764) ERROR 08-22 20:20:17 [core.py:1349] EngineCore failed to start.
(EngineCore pid=12764) ERROR 08-22 20:20:17 [core.py:1349] Traceback (most recent call last):
(EngineCore pid=12764) ERROR 08-22 20:20:17 [core.py:1349]   File "/home/enginrect/llmso/.venv/lib/python3.12/site-packages/vllm/v1/engine/core.py", line 1318, in run_engine_core
(EngineCore pid=12764) ERROR 08-22 20:20:17 [core.py:1349]     engine_core = EngineCoreProc(*args, engine_index=dp_rank, **kwargs)
(EngineCore pid=12764) ERROR 08-22 20:20:17 [core.py:1349]                   ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
(EngineCore pid=12764) ERROR 08-22 20:20:17 [core.py:1349]   File "/home/enginrect/llmso/.venv/lib/python3.12/site-packages/vllm/tracing/otel.py", line 178, in sync_wrapper
(EngineCore pid=12764) ERROR 08-22 20:20:17 [core.py:1349]     return func(*args, **kwargs)
(EngineCore pid=12764) ERROR 08-22 20:20:17 [core.py:1349]            ^^^^^^^^^^^^^^^^^^^^^
(EngineCore pid=12764) ERROR 08-22 20:20:17 [core.py:1349]   File "/home/enginrect/llmso/.venv/lib/python3.12/site-packages/vllm/v1/engine/core.py", line 1074, in __init__
(EngineCore pid=12764) ERROR 08-22 20:20:17 [core.py:1349]     super().__init__(
(EngineCore pid=12764) ERROR 08-22 20:20:17 [core.py:1349]   File "/home/enginrect/llmso/.venv/lib/python3.12/site-packages/vllm/v1/engine/core.py", line 143, in __init__
(EngineCore pid=12764) ERROR 08-22 20:20:17 [core.py:1349]     kv_cache_config = self._initialize_kv_caches(vllm_config)
(EngineCore pid=12764) ERROR 08-22 20:20:17 [core.py:1349]                       ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
(EngineCore pid=12764) ERROR 08-22 20:20:17 [core.py:1349]   File "/home/enginrect/llmso/.venv/lib/python3.12/site-packages/vllm/tracing/otel.py", line 178, in sync_wrapper
(EngineCore pid=12764) ERROR 08-22 20:20:17 [core.py:1349]     return func(*args, **kwargs)
(EngineCore pid=12764) ERROR 08-22 20:20:17 [core.py:1349]            ^^^^^^^^^^^^^^^^^^^^^
(EngineCore pid=12764) ERROR 08-22 20:20:17 [core.py:1349]   File "/home/enginrect/llmso/.venv/lib/python3.12/site-packages/vllm/v1/engine/core.py", line 332, in _initialize_kv_caches
(EngineCore pid=12764) ERROR 08-22 20:20:17 [core.py:1349]     self.model_executor.compile_or_warm_up_model()
(EngineCore pid=12764) ERROR 08-22 20:20:17 [core.py:1349]   File "/home/enginrect/llmso/.venv/lib/python3.12/site-packages/vllm/v1/executor/abstract.py", line 124, in compile_or_warm_up_model
(EngineCore pid=12764) ERROR 08-22 20:20:17 [core.py:1349]     compilation_times: list[CompilationTimes] = self.collective_rpc(
(EngineCore pid=12764) ERROR 08-22 20:20:17 [core.py:1349]                                                 ^^^^^^^^^^^^^^^^^^^^
(EngineCore pid=12764) ERROR 08-22 20:20:17 [core.py:1349]   File "/home/enginrect/llmso/.venv/lib/python3.12/site-packages/vllm/v1/executor/uniproc_executor.py", line 92, in collective_rpc
(EngineCore pid=12764) ERROR 08-22 20:20:17 [core.py:1349]     result = run_method(self.driver_worker, method, args, kwargs)
(EngineCore pid=12764) ERROR 08-22 20:20:17 [core.py:1349]              ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
(EngineCore pid=12764) ERROR 08-22 20:20:17 [core.py:1349]   File "/home/enginrect/llmso/.venv/lib/python3.12/site-packages/vllm/v1/serial_utils.py", line 510, in run_method
(EngineCore pid=12764) ERROR 08-22 20:20:17 [core.py:1349]     return func(*args, **kwargs)
(EngineCore pid=12764) ERROR 08-22 20:20:17 [core.py:1349]            ^^^^^^^^^^^^^^^^^^^^^
(EngineCore pid=12764) ERROR 08-22 20:20:17 [core.py:1349]   File "/home/enginrect/llmso/.venv/lib/python3.12/site-packages/vllm/tracing/otel.py", line 178, in sync_wrapper
(EngineCore pid=12764) ERROR 08-22 20:20:17 [core.py:1349]     return func(*args, **kwargs)
(EngineCore pid=12764) ERROR 08-22 20:20:17 [core.py:1349]            ^^^^^^^^^^^^^^^^^^^^^
(EngineCore pid=12764) ERROR 08-22 20:20:17 [core.py:1349]   File "/home/enginrect/llmso/.venv/lib/python3.12/site-packages/vllm/v1/worker/gpu_worker.py", line 795, in compile_or_warm_up_model
(EngineCore pid=12764) ERROR 08-22 20:20:17 [core.py:1349]     warmup_kernels(self.model_runner, self.execute_model, self.sample_tokens)
(EngineCore pid=12764) ERROR 08-22 20:20:17 [core.py:1349]   File "/home/enginrect/llmso/.venv/lib/python3.12/site-packages/torch/utils/_contextlib.py", line 124, in decorate_context
(EngineCore pid=12764) ERROR 08-22 20:20:17 [core.py:1349]     return func(*args, **kwargs)
(EngineCore pid=12764) ERROR 08-22 20:20:17 [core.py:1349]            ^^^^^^^^^^^^^^^^^^^^^
(EngineCore pid=12764) ERROR 08-22 20:20:17 [core.py:1349]   File "/home/enginrect/llmso/.venv/lib/python3.12/site-packages/vllm/v1/worker/gpu/warmup.py", line 296, in warmup_kernels
(EngineCore pid=12764) ERROR 08-22 20:20:17 [core.py:1349]     worker_sample_tokens(grammar_output)
(EngineCore pid=12764) ERROR 08-22 20:20:17 [core.py:1349]   File "/home/enginrect/llmso/.venv/lib/python3.12/site-packages/torch/utils/_contextlib.py", line 124, in decorate_context
(EngineCore pid=12764) ERROR 08-22 20:20:17 [core.py:1349]     return func(*args, **kwargs)
(EngineCore pid=12764) ERROR 08-22 20:20:17 [core.py:1349]            ^^^^^^^^^^^^^^^^^^^^^
(EngineCore pid=12764) ERROR 08-22 20:20:17 [core.py:1349]   File "/home/enginrect/llmso/.venv/lib/python3.12/site-packages/vllm/v1/worker/gpu_worker.py", line 1015, in sample_tokens
(EngineCore pid=12764) ERROR 08-22 20:20:17 [core.py:1349]     return self.model_runner.sample_tokens(grammar_output)
(EngineCore pid=12764) ERROR 08-22 20:20:17 [core.py:1349]            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
(EngineCore pid=12764) ERROR 08-22 20:20:17 [core.py:1349]   File "/home/enginrect/llmso/.venv/lib/python3.12/site-packages/torch/utils/_contextlib.py", line 124, in decorate_context
(EngineCore pid=12764) ERROR 08-22 20:20:17 [core.py:1349]     return func(*args, **kwargs)
(EngineCore pid=12764) ERROR 08-22 20:20:17 [core.py:1349]            ^^^^^^^^^^^^^^^^^^^^^
(EngineCore pid=12764) ERROR 08-22 20:20:17 [core.py:1349]   File "/home/enginrect/llmso/.venv/lib/python3.12/site-packages/vllm/v1/worker/gpu/eplb_utils.py", line 38, in wrapper
(EngineCore pid=12764) ERROR 08-22 20:20:17 [core.py:1349]     result = fn(self, *args, **kwargs)
(EngineCore pid=12764) ERROR 08-22 20:20:17 [core.py:1349]              ^^^^^^^^^^^^^^^^^^^^^^^^^
(EngineCore pid=12764) ERROR 08-22 20:20:17 [core.py:1349]   File "/home/enginrect/llmso/.venv/lib/python3.12/site-packages/vllm/v1/worker/gpu/model_runner.py", line 1495, in sample_tokens
(EngineCore pid=12764) ERROR 08-22 20:20:17 [core.py:1349]     sampler_output, num_sampled, num_rejected = self.sample(
(EngineCore pid=12764) ERROR 08-22 20:20:17 [core.py:1349]                                                 ^^^^^^^^^^^^
(EngineCore pid=12764) ERROR 08-22 20:20:17 [core.py:1349]   File "/home/enginrect/llmso/.venv/lib/python3.12/site-packages/vllm/v1/worker/gpu/model_runner.py", line 1163, in sample
(EngineCore pid=12764) ERROR 08-22 20:20:17 [core.py:1349]     sampler_output = self.sampler(logits, input_batch)
(EngineCore pid=12764) ERROR 08-22 20:20:17 [core.py:1349]                      ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
(EngineCore pid=12764) ERROR 08-22 20:20:17 [core.py:1349]   File "/home/enginrect/llmso/.venv/lib/python3.12/site-packages/vllm/v1/worker/gpu/sample/sampler.py", line 92, in __call__
(EngineCore pid=12764) ERROR 08-22 20:20:17 [core.py:1349]     sampled, processed_logits = self.sample(
(EngineCore pid=12764) ERROR 08-22 20:20:17 [core.py:1349]                                 ^^^^^^^^^^^^
(EngineCore pid=12764) ERROR 08-22 20:20:17 [core.py:1349]   File "/home/enginrect/llmso/.venv/lib/python3.12/site-packages/vllm/v1/worker/gpu/sample/sampler.py", line 252, in sample
(EngineCore pid=12764) ERROR 08-22 20:20:17 [core.py:1349]     sampled = flashinfer_sample(processed_logits, top_k, top_p).to(torch.int64)
(EngineCore pid=12764) ERROR 08-22 20:20:17 [core.py:1349]               ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
(EngineCore pid=12764) ERROR 08-22 20:20:17 [core.py:1349]   File "/home/enginrect/llmso/.venv/lib/python3.12/site-packages/vllm/v1/sample/ops/topk_topp_sampler.py", line 508, in flashinfer_sample
(EngineCore pid=12764) ERROR 08-22 20:20:17 [core.py:1349]     next_token_ids = flashinfer.sampling.top_k_top_p_sampling_from_logits(
(EngineCore pid=12764) ERROR 08-22 20:20:17 [core.py:1349]                      ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
(EngineCore pid=12764) ERROR 08-22 20:20:17 [core.py:1349]   File "/home/enginrect/llmso/.venv/lib/python3.12/site-packages/flashinfer/api_logging.py", line 2333, in _auto_dump_wrapper
(EngineCore pid=12764) ERROR 08-22 20:20:17 [core.py:1349]     return _inner(*args, **kwargs)
(EngineCore pid=12764) ERROR 08-22 20:20:17 [core.py:1349]            ^^^^^^^^^^^^^^^^^^^^^^^
(EngineCore pid=12764) ERROR 08-22 20:20:17 [core.py:1349]   File "/home/enginrect/llmso/.venv/lib/python3.12/site-packages/flashinfer/sampling.py", line 1547, in top_k_top_p_sampling_from_logits
(EngineCore pid=12764) ERROR 08-22 20:20:17 [core.py:1349]     masked_logits = top_k_mask_logits(logits, top_k)
(EngineCore pid=12764) ERROR 08-22 20:20:17 [core.py:1349]                     ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
(EngineCore pid=12764) ERROR 08-22 20:20:17 [core.py:1349]   File "/home/enginrect/llmso/.venv/lib/python3.12/site-packages/flashinfer/api_logging.py", line 2333, in _auto_dump_wrapper
(EngineCore pid=12764) ERROR 08-22 20:20:17 [core.py:1349]     return _inner(*args, **kwargs)
(EngineCore pid=12764) ERROR 08-22 20:20:17 [core.py:1349]            ^^^^^^^^^^^^^^^^^^^^^^^
(EngineCore pid=12764) ERROR 08-22 20:20:17 [core.py:1349]   File "/home/enginrect/llmso/.venv/lib/python3.12/site-packages/flashinfer/sampling.py", line 1974, in top_k_mask_logits
(EngineCore pid=12764) ERROR 08-22 20:20:17 [core.py:1349]     return get_sampling_module().top_k_mask_logits(
(EngineCore pid=12764) ERROR 08-22 20:20:17 [core.py:1349]            ^^^^^^^^^^^^^^^^^^^^^
(EngineCore pid=12764) ERROR 08-22 20:20:17 [core.py:1349]   File "/home/enginrect/llmso/.venv/lib/python3.12/site-packages/flashinfer/sampling.py", line 68, in get_sampling_module
(EngineCore pid=12764) ERROR 08-22 20:20:17 [core.py:1349]     module = gen_sampling_module().build_and_load()
(EngineCore pid=12764) ERROR 08-22 20:20:17 [core.py:1349]              ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
(EngineCore pid=12764) ERROR 08-22 20:20:17 [core.py:1349]   File "/home/enginrect/llmso/.venv/lib/python3.12/site-packages/flashinfer/jit/core.py", line 320, in build_and_load
(EngineCore pid=12764) ERROR 08-22 20:20:17 [core.py:1349]     self.build()
(EngineCore pid=12764) ERROR 08-22 20:20:17 [core.py:1349]   File "/home/enginrect/llmso/.venv/lib/python3.12/site-packages/flashinfer/jit/core.py", line 426, in build
(EngineCore pid=12764) ERROR 08-22 20:20:17 [core.py:1349]     self.write_ninja()
(EngineCore pid=12764) ERROR 08-22 20:20:17 [core.py:1349]   File "/home/enginrect/llmso/.venv/lib/python3.12/site-packages/flashinfer/jit/core.py", line 381, in write_ninja
(EngineCore pid=12764) ERROR 08-22 20:20:17 [core.py:1349]     content = generate_ninja_build_for_op(
(EngineCore pid=12764) ERROR 08-22 20:20:17 [core.py:1349]               ^^^^^^^^^^^^^^^^^^^^^^^^^^^^
(EngineCore pid=12764) ERROR 08-22 20:20:17 [core.py:1349]   File "/home/enginrect/llmso/.venv/lib/python3.12/site-packages/flashinfer/jit/cpp_ext.py", line 249, in generate_ninja_build_for_op
(EngineCore pid=12764) ERROR 08-22 20:20:17 [core.py:1349]     cuda_home = get_cuda_path()
(EngineCore pid=12764) ERROR 08-22 20:20:17 [core.py:1349]                 ^^^^^^^^^^^^^^^
(EngineCore pid=12764) ERROR 08-22 20:20:17 [core.py:1349]   File "/home/enginrect/llmso/.venv/lib/python3.12/site-packages/flashinfer/jit/cpp_ext.py", line 61, in get_cuda_path
(EngineCore pid=12764) ERROR 08-22 20:20:17 [core.py:1349]     raise RuntimeError(
(EngineCore pid=12764) ERROR 08-22 20:20:17 [core.py:1349] RuntimeError: Could not find nvcc and default cuda_home='/usr/local/cuda' doesn't exist
...
(EngineCore pid=12764) Traceback (most recent call last):
...
(EngineCore pid=12764)     raise e
...
(EngineCore pid=12764)     raise RuntimeError(
(EngineCore pid=12764) RuntimeError: Could not find nvcc and default cuda_home='/usr/local/cuda' doesn't exist
...
(APIServer pid=12654) Traceback (most recent call last):
...
(APIServer pid=12654)   File "/home/enginrect/llmso/.venv/lib/python3.12/site-packages/vllm/v1/engine/utils.py", line 1253, in wait_for_engine_startup
(APIServer pid=12654)     raise RuntimeError(
(APIServer pid=12654) RuntimeError: Engine core initialization failed. See root cause above. Failed core proc(s): {}
```

**원인 분석**: 이번엔 KV cache 초기화(1,629,520 토큰)까지 통과한 뒤, **flashinfer**가 sampling 커널을 JIT 빌드하려고 CUDA 툴킷을 찾다 죽었다. `flashinfer/jit/cpp_ext.py:get_cuda_path()`는 ①`CUDA_HOME`/`CUDA_PATH` 환경변수 ②`which nvcc` ③`/usr/local/cuda` 순으로 찾는데 셋 다 없다.

그런데 Step 4에서 확인했듯 vLLM 의존성에 `nvidia-cuda-nvcc` pip 패키지가 딸려왔다:

```shell
$ find ~/llmso/.venv -name "nvcc" -type f
/home/enginrect/llmso/.venv/lib/python3.12/site-packages/nvidia/cu13/bin/nvcc
$ ls ~/llmso/.venv/lib/python3.12/site-packages/nvidia/cu13/
bin
cccl
include
lib
nvvm
```

`nvidia/cu13`이 bin/include/lib/nvvm을 갖춘 완전한 툴킷 레이아웃이다. 시스템에 CUDA 툴킷을 따로 설치하는 대신 이걸 `CUDA_HOME`으로 지정한다.

### Step 10 — CUDA_HOME을 venv 내 pip CUDA 툴킷으로 지정

```shell
$ CUDA_HOME을 venv 내 pip CUDA 툴킷으로 지정 (~/.bashrc 영속화)
CUDA_HOME: 추가함

$ nvcc 동작 확인
Cuda compilation tools, release 13.3, V13.3.73
Build cuda_13.3.r13.3/compiler.38244171_0
```

- key point: apt로 수 GB짜리 CUDA 툴킷을 또 설치하지 않고, pip로 이미 들어온 툴킷을 재사용했다. 단 이 시점의 nvcc는 13.3이다 — 이게 다음 함정이 된다.

### 시도 5 — ❌ `CUDA compiler and CUDA toolkit headers are incompatible`

출력 (발췌):

```shell
(APIServer pid=13257) INFO 08-22 20:21:19 [api_utils.py:345] 
(APIServer pid=13257) INFO 08-22 20:21:19 [api_utils.py:345]        █     █     █▄   ▄█
(APIServer pid=13257) INFO 08-22 20:21:19 [api_utils.py:345]  ▄▄ ▄█ █     █     █ ▀▄▀ █  version 0.27.1
(APIServer pid=13257) INFO 08-22 20:21:19 [api_utils.py:345]   █▄█▀ █     █     █     █  model   Qwen/Qwen2.5-0.5B-Instruct
...
(APIServer pid=13257) INFO 08-22 20:21:19 [api_utils.py:273] non-default args: {'model_tag': 'Qwen/Qwen2.5-0.5B-Instruct', 'model': 'Qwen/Qwen2.5-0.5B-Instruct', 'max_model_len': 2048, 'gpu_memory_utilization': 0.85}
...
(EngineCore pid=13366) INFO 08-22 20:21:32 [weight_utils.py:867] Filesystem type for checkpoints: EXT4. Checkpoint size: 0.92 GiB. Available RAM: 28.11 GiB.
...
(EngineCore pid=13366) INFO 08-22 20:21:33 [model_runner.py:329] Model loading took 0.93 GiB and 2.076413 seconds
...
(EngineCore pid=13366) INFO 08-22 20:21:41 [gpu_worker.py:563] Available KV cache memory: 18.65 GiB
(EngineCore pid=13366) INFO 08-22 20:21:41 [kv_cache_utils.py:2235] GPU KV cache size: 1,629,520 tokens
...
(EngineCore pid=13366) INFO 08-22 20:21:43 [model_runner.py:791] Graph capturing finished in 2 secs, took 0.18 GiB
(EngineCore pid=13366) INFO 08-22 20:21:43 [gpu_worker.py:789] Free memory on device (22.45/23.99 GiB) on startup. Desired GPU memory utilization is (0.85, 20.39 GiB). Actual usage is 1.49 GiB for consumed memory (weights + non-torch), 0.25 GiB for peak activation, and 0.18 GiB for CUDAGraph memory. Replace gpu_memory_utilization config with `--kv-cache-memory=19674042266` (18.32 GiB) to fit into requested memory, or `--kv-cache-memory=21885515264` (20.38 GiB) to fully utilize gpu memory. Current kv cache memory in use is 18.65 GiB.
(EngineCore pid=13366) ERROR 08-22 20:21:50 [core.py:1349] EngineCore failed to start.
(EngineCore pid=13366) ERROR 08-22 20:21:50 [core.py:1349] Traceback (most recent call last):
(EngineCore pid=13366) ERROR 08-22 20:21:50 [core.py:1349]   File "/home/enginrect/llmso/.venv/lib/python3.12/site-packages/flashinfer/jit/cpp_ext.py", line 370, in run_ninja
(EngineCore pid=13366) ERROR 08-22 20:21:50 [core.py:1349]     subprocess.run(
(EngineCore pid=13366) ERROR 08-22 20:21:50 [core.py:1349]   File "/usr/lib/python3.12/subprocess.py", line 571, in run
(EngineCore pid=13366) ERROR 08-22 20:21:50 [core.py:1349]     raise CalledProcessError(retcode, process.args,
(EngineCore pid=13366) ERROR 08-22 20:21:50 [core.py:1349] subprocess.CalledProcessError: Command '['ninja', '-v', '-C', '/home/enginrect/.cache/flashinfer/0.6.16.post3/89/cached_ops/sampling', '-f', '/home/enginrect/.cache/flashinfer/0.6.16.post3/89/cached_ops/sampling/build.ninja']' returned non-zero exit status 1.
(EngineCore pid=13366) ERROR 08-22 20:21:50 [core.py:1349] 
(EngineCore pid=13366) ERROR 08-22 20:21:50 [core.py:1349] The above exception was the direct cause of the following exception:
(EngineCore pid=13366) ERROR 08-22 20:21:50 [core.py:1349] 
(EngineCore pid=13366) ERROR 08-22 20:21:50 [core.py:1349] Traceback (most recent call last):
(EngineCore pid=13366) ERROR 08-22 20:21:50 [core.py:1349]   File "/home/enginrect/llmso/.venv/lib/python3.12/site-packages/vllm/v1/engine/core.py", line 1318, in run_engine_core
(EngineCore pid=13366) ERROR 08-22 20:21:50 [core.py:1349]     engine_core = EngineCoreProc(*args, engine_index=dp_rank, **kwargs)
(EngineCore pid=13366) ERROR 08-22 20:21:50 [core.py:1349]                   ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
(EngineCore pid=13366) ERROR 08-22 20:21:50 [core.py:1349]   File "/home/enginrect/llmso/.venv/lib/python3.12/site-packages/vllm/tracing/otel.py", line 178, in sync_wrapper
(EngineCore pid=13366) ERROR 08-22 20:21:50 [core.py:1349]     return func(*args, **kwargs)
(EngineCore pid=13366) ERROR 08-22 20:21:50 [core.py:1349]            ^^^^^^^^^^^^^^^^^^^^^
(EngineCore pid=13366) ERROR 08-22 20:21:50 [core.py:1349]   File "/home/enginrect/llmso/.venv/lib/python3.12/site-packages/vllm/v1/engine/core.py", line 1074, in __init__
(EngineCore pid=13366) ERROR 08-22 20:21:50 [core.py:1349]     super().__init__(
(EngineCore pid=13366) ERROR 08-22 20:21:50 [core.py:1349]   File "/home/enginrect/llmso/.venv/lib/python3.12/site-packages/vllm/v1/engine/core.py", line 143, in __init__
(EngineCore pid=13366) ERROR 08-22 20:21:50 [core.py:1349]     kv_cache_config = self._initialize_kv_caches(vllm_config)
(EngineCore pid=13366) ERROR 08-22 20:21:50 [core.py:1349]                       ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
(EngineCore pid=13366) ERROR 08-22 20:21:50 [core.py:1349]   File "/home/enginrect/llmso/.venv/lib/python3.12/site-packages/vllm/tracing/otel.py", line 178, in sync_wrapper
(EngineCore pid=13366) ERROR 08-22 20:21:50 [core.py:1349]     return func(*args, **kwargs)
(EngineCore pid=13366) ERROR 08-22 20:21:50 [core.py:1349]            ^^^^^^^^^^^^^^^^^^^^^
(EngineCore pid=13366) ERROR 08-22 20:21:50 [core.py:1349]   File "/home/enginrect/llmso/.venv/lib/python3.12/site-packages/vllm/v1/engine/core.py", line 332, in _initialize_kv_caches
(EngineCore pid=13366) ERROR 08-22 20:21:50 [core.py:1349]     self.model_executor.compile_or_warm_up_model()
(EngineCore pid=13366) ERROR 08-22 20:21:50 [core.py:1349]   File "/home/enginrect/llmso/.venv/lib/python3.12/site-packages/vllm/v1/executor/abstract.py", line 124, in compile_or_warm_up_model
(EngineCore pid=13366) ERROR 08-22 20:21:50 [core.py:1349]     compilation_times: list[CompilationTimes] = self.collective_rpc(
(EngineCore pid=13366) ERROR 08-22 20:21:50 [core.py:1349]                                                 ^^^^^^^^^^^^^^^^^^^^
(EngineCore pid=13366) ERROR 08-22 20:21:50 [core.py:1349]   File "/home/enginrect/llmso/.venv/lib/python3.12/site-packages/vllm/v1/executor/uniproc_executor.py", line 92, in collective_rpc
(EngineCore pid=13366) ERROR 08-22 20:21:50 [core.py:1349]     result = run_method(self.driver_worker, method, args, kwargs)
(EngineCore pid=13366) ERROR 08-22 20:21:50 [core.py:1349]              ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
(EngineCore pid=13366) ERROR 08-22 20:21:50 [core.py:1349]   File "/home/enginrect/llmso/.venv/lib/python3.12/site-packages/vllm/v1/serial_utils.py", line 510, in run_method
(EngineCore pid=13366) ERROR 08-22 20:21:50 [core.py:1349]     return func(*args, **kwargs)
(EngineCore pid=13366) ERROR 08-22 20:21:50 [core.py:1349]            ^^^^^^^^^^^^^^^^^^^^^
(EngineCore pid=13366) ERROR 08-22 20:21:50 [core.py:1349]   File "/home/enginrect/llmso/.venv/lib/python3.12/site-packages/vllm/tracing/otel.py", line 178, in sync_wrapper
(EngineCore pid=13366) ERROR 08-22 20:21:50 [core.py:1349]     return func(*args, **kwargs)
(EngineCore pid=13366) ERROR 08-22 20:21:50 [core.py:1349]            ^^^^^^^^^^^^^^^^^^^^^
(EngineCore pid=13366) ERROR 08-22 20:21:50 [core.py:1349]   File "/home/enginrect/llmso/.venv/lib/python3.12/site-packages/vllm/v1/worker/gpu_worker.py", line 795, in compile_or_warm_up_model
(EngineCore pid=13366) ERROR 08-22 20:21:50 [core.py:1349]     warmup_kernels(self.model_runner, self.execute_model, self.sample_tokens)
(EngineCore pid=13366) ERROR 08-22 20:21:50 [core.py:1349]   File "/home/enginrect/llmso/.venv/lib/python3.12/site-packages/torch/utils/_contextlib.py", line 124, in decorate_context
(EngineCore pid=13366) ERROR 08-22 20:21:50 [core.py:1349]     return func(*args, **kwargs)
(EngineCore pid=13366) ERROR 08-22 20:21:50 [core.py:1349]            ^^^^^^^^^^^^^^^^^^^^^
(EngineCore pid=13366) ERROR 08-22 20:21:50 [core.py:1349]   File "/home/enginrect/llmso/.venv/lib/python3.12/site-packages/vllm/v1/worker/gpu/warmup.py", line 296, in warmup_kernels
(EngineCore pid=13366) ERROR 08-22 20:21:50 [core.py:1349]     worker_sample_tokens(grammar_output)
(EngineCore pid=13366) ERROR 08-22 20:21:50 [core.py:1349]   File "/home/enginrect/llmso/.venv/lib/python3.12/site-packages/torch/utils/_contextlib.py", line 124, in decorate_context
(EngineCore pid=13366) ERROR 08-22 20:21:50 [core.py:1349]     return func(*args, **kwargs)
(EngineCore pid=13366) ERROR 08-22 20:21:50 [core.py:1349]            ^^^^^^^^^^^^^^^^^^^^^
(EngineCore pid=13366) ERROR 08-22 20:21:50 [core.py:1349]   File "/home/enginrect/llmso/.venv/lib/python3.12/site-packages/vllm/v1/worker/gpu_worker.py", line 1015, in sample_tokens
(EngineCore pid=13366) ERROR 08-22 20:21:50 [core.py:1349]     return self.model_runner.sample_tokens(grammar_output)
(EngineCore pid=13366) ERROR 08-22 20:21:50 [core.py:1349]            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
(EngineCore pid=13366) ERROR 08-22 20:21:50 [core.py:1349]   File "/home/enginrect/llmso/.venv/lib/python3.12/site-packages/torch/utils/_contextlib.py", line 124, in decorate_context
(EngineCore pid=13366) ERROR 08-22 20:21:50 [core.py:1349]     return func(*args, **kwargs)
(EngineCore pid=13366) ERROR 08-22 20:21:50 [core.py:1349]            ^^^^^^^^^^^^^^^^^^^^^
(EngineCore pid=13366) ERROR 08-22 20:21:50 [core.py:1349]   File "/home/enginrect/llmso/.venv/lib/python3.12/site-packages/vllm/v1/worker/gpu/eplb_utils.py", line 38, in wrapper
(EngineCore pid=13366) ERROR 08-22 20:21:50 [core.py:1349]     result = fn(self, *args, **kwargs)
(EngineCore pid=13366) ERROR 08-22 20:21:50 [core.py:1349]              ^^^^^^^^^^^^^^^^^^^^^^^^^
(EngineCore pid=13366) ERROR 08-22 20:21:50 [core.py:1349]   File "/home/enginrect/llmso/.venv/lib/python3.12/site-packages/vllm/v1/worker/gpu/model_runner.py", line 1495, in sample_tokens
(EngineCore pid=13366) ERROR 08-22 20:21:50 [core.py:1349]     sampler_output, num_sampled, num_rejected = self.sample(
(EngineCore pid=13366) ERROR 08-22 20:21:50 [core.py:1349]                                                 ^^^^^^^^^^^^
(EngineCore pid=13366) ERROR 08-22 20:21:50 [core.py:1349]   File "/home/enginrect/llmso/.venv/lib/python3.12/site-packages/vllm/v1/worker/gpu/model_runner.py", line 1163, in sample
(EngineCore pid=13366) ERROR 08-22 20:21:50 [core.py:1349]     sampler_output = self.sampler(logits, input_batch)
(EngineCore pid=13366) ERROR 08-22 20:21:50 [core.py:1349]                      ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
(EngineCore pid=13366) ERROR 08-22 20:21:50 [core.py:1349]   File "/home/enginrect/llmso/.venv/lib/python3.12/site-packages/vllm/v1/worker/gpu/sample/sampler.py", line 92, in __call__
(EngineCore pid=13366) ERROR 08-22 20:21:50 [core.py:1349]     sampled, processed_logits = self.sample(
(EngineCore pid=13366) ERROR 08-22 20:21:50 [core.py:1349]                                 ^^^^^^^^^^^^
(EngineCore pid=13366) ERROR 08-22 20:21:50 [core.py:1349]   File "/home/enginrect/llmso/.venv/lib/python3.12/site-packages/vllm/v1/worker/gpu/sample/sampler.py", line 252, in sample
(EngineCore pid=13366) ERROR 08-22 20:21:50 [core.py:1349]     sampled = flashinfer_sample(processed_logits, top_k, top_p).to(torch.int64)
(EngineCore pid=13366) ERROR 08-22 20:21:50 [core.py:1349]               ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
(EngineCore pid=13366) ERROR 08-22 20:21:50 [core.py:1349]   File "/home/enginrect/llmso/.venv/lib/python3.12/site-packages/vllm/v1/sample/ops/topk_topp_sampler.py", line 508, in flashinfer_sample
(EngineCore pid=13366) ERROR 08-22 20:21:50 [core.py:1349]     next_token_ids = flashinfer.sampling.top_k_top_p_sampling_from_logits(
(EngineCore pid=13366) ERROR 08-22 20:21:50 [core.py:1349]                      ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
(EngineCore pid=13366) ERROR 08-22 20:21:50 [core.py:1349]   File "/home/enginrect/llmso/.venv/lib/python3.12/site-packages/flashinfer/api_logging.py", line 2333, in _auto_dump_wrapper
(EngineCore pid=13366) ERROR 08-22 20:21:50 [core.py:1349]     return _inner(*args, **kwargs)
(EngineCore pid=13366) ERROR 08-22 20:21:50 [core.py:1349]            ^^^^^^^^^^^^^^^^^^^^^^^
(EngineCore pid=13366) ERROR 08-22 20:21:50 [core.py:1349]   File "/home/enginrect/llmso/.venv/lib/python3.12/site-packages/flashinfer/sampling.py", line 1547, in top_k_top_p_sampling_from_logits
(EngineCore pid=13366) ERROR 08-22 20:21:50 [core.py:1349]     masked_logits = top_k_mask_logits(logits, top_k)
(EngineCore pid=13366) ERROR 08-22 20:21:50 [core.py:1349]                     ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
(EngineCore pid=13366) ERROR 08-22 20:21:50 [core.py:1349]   File "/home/enginrect/llmso/.venv/lib/python3.12/site-packages/flashinfer/api_logging.py", line 2333, in _auto_dump_wrapper
(EngineCore pid=13366) ERROR 08-22 20:21:50 [core.py:1349]     return _inner(*args, **kwargs)
(EngineCore pid=13366) ERROR 08-22 20:21:50 [core.py:1349]            ^^^^^^^^^^^^^^^^^^^^^^^
(EngineCore pid=13366) ERROR 08-22 20:21:50 [core.py:1349]   File "/home/enginrect/llmso/.venv/lib/python3.12/site-packages/flashinfer/sampling.py", line 1974, in top_k_mask_logits
(EngineCore pid=13366) ERROR 08-22 20:21:50 [core.py:1349]     return get_sampling_module().top_k_mask_logits(
(EngineCore pid=13366) ERROR 08-22 20:21:50 [core.py:1349]            ^^^^^^^^^^^^^^^^^^^^^
(EngineCore pid=13366) ERROR 08-22 20:21:50 [core.py:1349]   File "/home/enginrect/llmso/.venv/lib/python3.12/site-packages/flashinfer/sampling.py", line 68, in get_sampling_module
(EngineCore pid=13366) ERROR 08-22 20:21:50 [core.py:1349]     module = gen_sampling_module().build_and_load()
(EngineCore pid=13366) ERROR 08-22 20:21:50 [core.py:1349]              ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
(EngineCore pid=13366) ERROR 08-22 20:21:50 [core.py:1349]   File "/home/enginrect/llmso/.venv/lib/python3.12/site-packages/flashinfer/jit/core.py", line 320, in build_and_load
(EngineCore pid=13366) ERROR 08-22 20:21:50 [core.py:1349]     self.build()
(EngineCore pid=13366) ERROR 08-22 20:21:50 [core.py:1349]   File "/home/enginrect/llmso/.venv/lib/python3.12/site-packages/flashinfer/jit/core.py", line 427, in build
(EngineCore pid=13366) ERROR 08-22 20:21:50 [core.py:1349]     run_ninja(self.build_dir, self.ninja_path, verbose)
(EngineCore pid=13366) ERROR 08-22 20:21:50 [core.py:1349]   File "/home/enginrect/llmso/.venv/lib/python3.12/site-packages/flashinfer/jit/cpp_ext.py", line 382, in run_ninja
(EngineCore pid=13366) ERROR 08-22 20:21:50 [core.py:1349]     raise RuntimeError(msg) from e
(EngineCore pid=13366) ERROR 08-22 20:21:50 [core.py:1349] RuntimeError: Ninja build failed. Ninja output:
(EngineCore pid=13366) ERROR 08-22 20:21:50 [core.py:1349] ninja: Entering directory `/home/enginrect/.cache/flashinfer/0.6.16.post3/89/cached_ops/sampling'
(EngineCore pid=13366) ERROR 08-22 20:21:50 [core.py:1349] [1/4]  /home/enginrect/llmso/.venv/lib/python3.12/site-packages/nvidia/cu13/bin/nvcc --generate-dependencies-with-compile -MF /home/enginrect/.cache/flashinfer/0.6.16.post3/89/cached_ops/sampling/csrc_sampling.cuda.o.d -DPy_LIMITED_API=0x03090000 -D_GLIBCXX_USE_CXX11_ABI=1 -I/home/enginrect/llmso/.venv/lib/python3.12/site-packages/flashinfer/data/cccl/cub -I/home/enginrect/llmso/.venv/lib/python3.12/site-packages/flashinfer/data/cccl/libcudacxx/include -I/home/enginrect/llmso/.venv/lib/python3.12/site-packages/flashinfer/data/cccl/thrust -isystem /usr/include/python3.12 -isystem /home/enginrect/llmso/.venv/lib/python3.12/site-packages/nvidia/cu13/include -isystem /home/enginrect/llmso/.venv/lib/python3.12/site-packages/tvm_ffi/include -isystem /home/enginrect/llmso/.venv/lib/python3.12/site-packages/tvm_ffi/include -isystem /home/enginrect/llmso/.venv/lib/python3.12/site-packages/flashinfer/data/include -isystem /home/enginrect/llmso/.venv/lib/python3.12/site-packages/flashinfer/data/csrc -isystem /home/enginrect/llmso/.venv/lib/python3.12/site-packages/flashinfer/data/cutlass/include -isystem /home/enginrect/llmso/.venv/lib/python3.12/site-packages/flashinfer/data/cutlass/tools/util/include -isystem /home/enginrect/llmso/.venv/lib/python3.12/site-packages/flashinfer/data/spdlog/include --compiler-options=-fPIC --expt-relaxed-constexpr -static-global-template-stub=false -gencode=arch=compute_89,code=sm_89 -DFLASHINFER_ENABLE_FP8_E8M0 -DFLASHINFER_ENABLE_FP4_E2M1 -std=c++17 --threads=1 -use_fast_math -Xfatbin=-compress-all -DFLASHINFER_ENABLE_F16 -DFLASHINFER_ENABLE_BF16 -DFLASHINFER_ENABLE_FP8_E4M3 -DFLASHINFER_ENABLE_FP8_E5M2 -DNDEBUG -O3 -c /home/enginrect/llmso/.venv/lib/python3.12/site-packages/flashinfer/data/csrc/sampling.cu -o /home/enginrect/.cache/flashinfer/0.6.16.post3/89/cached_ops/sampling/csrc_sampling.cuda.o 
(EngineCore pid=13366) ERROR 08-22 20:21:50 [core.py:1349] FAILED: [code=1] /home/enginrect/.cache/flashinfer/0.6.16.post3/89/cached_ops/sampling/csrc_sampling.cuda.o 
(EngineCore pid=13366) ERROR 08-22 20:21:50 [core.py:1349]  /home/enginrect/llmso/.venv/lib/python3.12/site-packages/nvidia/cu13/bin/nvcc --generate-dependencies-with-compile -MF /home/enginrect/.cache/flashinfer/0.6.16.post3/89/cached_ops/sampling/csrc_sampling.cuda.o.d -DPy_LIMITED_API=0x03090000 -D_GLIBCXX_USE_CXX11_ABI=1 -I/home/enginrect/llmso/.venv/lib/python3.12/site-packages/flashinfer/data/cccl/cub -I/home/enginrect/llmso/.venv/lib/python3.12/site-packages/flashinfer/data/cccl/libcudacxx/include -I/home/enginrect/llmso/.venv/lib/python3.12/site-packages/flashinfer/data/cccl/thrust -isystem /usr/include/python3.12 -isystem /home/enginrect/llmso/.venv/lib/python3.12/site-packages/nvidia/cu13/include -isystem /home/enginrect/llmso/.venv/lib/python3.12/site-packages/tvm_ffi/include -isystem /home/enginrect/llmso/.venv/lib/python3.12/site-packages/tvm_ffi/include -isystem /home/enginrect/llmso/.venv/lib/python3.12/site-packages/flashinfer/data/include -isystem /home/enginrect/llmso/.venv/lib/python3.12/site-packages/flashinfer/data/csrc -isystem /home/enginrect/llmso/.venv/lib/python3.12/site-packages/flashinfer/data/cutlass/include -isystem /home/enginrect/llmso/.venv/lib/python3.12/site-packages/flashinfer/data/cutlass/tools/util/include -isystem /home/enginrect/llmso/.venv/lib/python3.12/site-packages/flashinfer/data/spdlog/include --compiler-options=-fPIC --expt-relaxed-constexpr -static-global-template-stub=false -gencode=arch=compute_89,code=sm_89 -DFLASHINFER_ENABLE_FP8_E8M0 -DFLASHINFER_ENABLE_FP4_E2M1 -std=c++17 --threads=1 -use_fast_math -Xfatbin=-compress-all -DFLASHINFER_ENABLE_F16 -DFLASHINFER_ENABLE_BF16 -DFLASHINFER_ENABLE_FP8_E4M3 -DFLASHINFER_ENABLE_FP8_E5M2 -DNDEBUG -O3 -c /home/enginrect/llmso/.venv/lib/python3.12/site-packages/flashinfer/data/csrc/sampling.cu -o /home/enginrect/.cache/flashinfer/0.6.16.post3/89/cached_ops/sampling/csrc_sampling.cuda.o 
(EngineCore pid=13366) ERROR 08-22 20:21:50 [core.py:1349] In file included from /home/enginrect/llmso/.venv/lib/python3.12/site-packages/flashinfer/data/cccl/libcudacxx/include/cuda/std/__cccl/extended_data_types.h:26,
(EngineCore pid=13366) ERROR 08-22 20:21:50 [core.py:1349]                  from /home/enginrect/llmso/.venv/lib/python3.12/site-packages/flashinfer/data/cccl/libcudacxx/include/cuda/std/__cccl/builtin.h:26,
(EngineCore pid=13366) ERROR 08-22 20:21:50 [core.py:1349]                  from /home/enginrect/llmso/.venv/lib/python3.12/site-packages/flashinfer/data/cccl/libcudacxx/include/cuda/std/__cccl/dialect.h:25,
(EngineCore pid=13366) ERROR 08-22 20:21:50 [core.py:1349]                  from /home/enginrect/llmso/.venv/lib/python3.12/site-packages/flashinfer/data/cccl/libcudacxx/include/cuda/std/__cccl/attributes.h:26,
(EngineCore pid=13366) ERROR 08-22 20:21:50 [core.py:1349]                  from /home/enginrect/llmso/.venv/lib/python3.12/site-packages/flashinfer/data/cccl/libcudacxx/include/cuda/std/__cccl/assert.h:25,
(EngineCore pid=13366) ERROR 08-22 20:21:50 [core.py:1349]                  from /home/enginrect/llmso/.venv/lib/python3.12/site-packages/flashinfer/data/cccl/libcudacxx/include/cuda/__cccl_config:15,
(EngineCore pid=13366) ERROR 08-22 20:21:50 [core.py:1349]                  from /home/enginrect/llmso/.venv/lib/python3.12/site-packages/flashinfer/data/cccl/cub/cub/config.cuh:12,
(EngineCore pid=13366) ERROR 08-22 20:21:50 [core.py:1349]                  from /home/enginrect/llmso/.venv/lib/python3.12/site-packages/flashinfer/data/cccl/cub/cub/cub.cuh:13,
(EngineCore pid=13366) ERROR 08-22 20:21:50 [core.py:1349]                  from /home/enginrect/llmso/.venv/lib/python3.12/site-packages/flashinfer/data/include/flashinfer/sampling.cuh:26,
(EngineCore pid=13366) ERROR 08-22 20:21:50 [core.py:1349]                  from /home/enginrect/llmso/.venv/lib/python3.12/site-packages/flashinfer/data/csrc/sampling.cu:16:
(EngineCore pid=13366) ERROR 08-22 20:21:50 [core.py:1349] /home/enginrect/llmso/.venv/lib/python3.12/site-packages/flashinfer/data/cccl/libcudacxx/include/cuda/std/__cccl/cuda_toolkit.h:41:8: error: #error "CUDA compiler and CUDA toolkit headers are incompatible, please check your include paths"
(EngineCore pid=13366) ERROR 08-22 20:21:50 [core.py:1349]    41 | #      error "CUDA compiler and CUDA toolkit headers are incompatible, please check your include paths"
(EngineCore pid=13366) ERROR 08-22 20:21:50 [core.py:1349]       |        ^~~~~
(EngineCore pid=13366) ERROR 08-22 20:21:50 [core.py:1349] [2/4]  /home/enginrect/llmso/.venv/lib/python3.12/site-packages/nvidia/cu13/bin/nvcc --generate-dependencies-with-compile -MF /home/enginrect/.cache/flashinfer/0.6.16.post3/89/cached_ops/sampling/csrc_renorm.cuda.o.d -DPy_LIMITED_API=0x03090000 -D_GLIBCXX_USE_CXX11_ABI=1 -I/home/enginrect/llmso/.venv/lib/python3.12/site-packages/flashinfer/data/cccl/cub -I/home/enginrect/llmso/.venv/lib/python3.12/site-packages/flashinfer/data/cccl/libcudacxx/include -I/home/enginrect/llmso/.venv/lib/python3.12/site-packages/flashinfer/data/cccl/thrust -isystem /usr/include/python3.12 -isystem /home/enginrect/llmso/.venv/lib/python3.12/site-packages/nvidia/cu13/include -isystem /home/enginrect/llmso/.venv/lib/python3.12/site-packages/tvm_ffi/include -isystem /home/enginrect/llmso/.venv/lib/python3.12/site-packages/tvm_ffi/include -isystem /home/enginrect/llmso/.venv/lib/python3.12/site-packages/flashinfer/data/include -isystem /home/enginrect/llmso/.venv/lib/python3.12/site-packages/flashinfer/data/csrc -isystem /home/enginrect/llmso/.venv/lib/python3.12/site-packages/flashinfer/data/cutlass/include -isystem /home/enginrect/llmso/.venv/lib/python3.12/site-packages/flashinfer/data/cutlass/tools/util/include -isystem /home/enginrect/llmso/.venv/lib/python3.12/site-packages/flashinfer/data/spdlog/include --compiler-options=-fPIC --expt-relaxed-constexpr -static-global-template-stub=false -gencode=arch=compute_89,code=sm_89 -DFLASHINFER_ENABLE_FP8_E8M0 -DFLASHINFER_ENABLE_FP4_E2M1 -std=c++17 --threads=1 -use_fast_math -Xfatbin=-compress-all -DFLASHINFER_ENABLE_F16 -DFLASHINFER_ENABLE_BF16 -DFLASHINFER_ENABLE_FP8_E4M3 -DFLASHINFER_ENABLE_FP8_E5M2 -DNDEBUG -O3 -c /home/enginrect/llmso/.venv/lib/python3.12/site-packages/flashinfer/data/csrc/renorm.cu -o /home/enginrect/.cache/flashinfer/0.6.16.post3/89/cached_ops/sampling/csrc_renorm.cuda.o 
(EngineCore pid=13366) ERROR 08-22 20:21:50 [core.py:1349] FAILED: [code=1] /home/enginrect/.cache/flashinfer/0.6.16.post3/89/cached_ops/sampling/csrc_renorm.cuda.o 
(EngineCore pid=13366) ERROR 08-22 20:21:50 [core.py:1349]  /home/enginrect/llmso/.venv/lib/python3.12/site-packages/nvidia/cu13/bin/nvcc --generate-dependencies-with-compile -MF /home/enginrect/.cache/flashinfer/0.6.16.post3/89/cached_ops/sampling/csrc_renorm.cuda.o.d -DPy_LIMITED_API=0x03090000 -D_GLIBCXX_USE_CXX11_ABI=1 -I/home/enginrect/llmso/.venv/lib/python3.12/site-packages/flashinfer/data/cccl/cub -I/home/enginrect/llmso/.venv/lib/python3.12/site-packages/flashinfer/data/cccl/libcudacxx/include -I/home/enginrect/llmso/.venv/lib/python3.12/site-packages/flashinfer/data/cccl/thrust -isystem /usr/include/python3.12 -isystem /home/enginrect/llmso/.venv/lib/python3.12/site-packages/nvidia/cu13/include -isystem /home/enginrect/llmso/.venv/lib/python3.12/site-packages/tvm_ffi/include -isystem /home/enginrect/llmso/.venv/lib/python3.12/site-packages/tvm_ffi/include -isystem /home/enginrect/llmso/.venv/lib/python3.12/site-packages/flashinfer/data/include -isystem /home/enginrect/llmso/.venv/lib/python3.12/site-packages/flashinfer/data/csrc -isystem /home/enginrect/llmso/.venv/lib/python3.12/site-packages/flashinfer/data/cutlass/include -isystem /home/enginrect/llmso/.venv/lib/python3.12/site-packages/flashinfer/data/cutlass/tools/util/include -isystem /home/enginrect/llmso/.venv/lib/python3.12/site-packages/flashinfer/data/spdlog/include --compiler-options=-fPIC --expt-relaxed-constexpr -static-global-template-stub=false -gencode=arch=compute_89,code=sm_89 -DFLASHINFER_ENABLE_FP8_E8M0 -DFLASHINFER_ENABLE_FP4_E2M1 -std=c++17 --threads=1 -use_fast_math -Xfatbin=-compress-all -DFLASHINFER_ENABLE_F16 -DFLASHINFER_ENABLE_BF16 -DFLASHINFER_ENABLE_FP8_E4M3 -DFLASHINFER_ENABLE_FP8_E5M2 -DNDEBUG -O3 -c /home/enginrect/llmso/.venv/lib/python3.12/site-packages/flashinfer/data/csrc/renorm.cu -o /home/enginrect/.cache/flashinfer/0.6.16.post3/89/cached_ops/sampling/csrc_renorm.cuda.o 
(EngineCore pid=13366) ERROR 08-22 20:21:50 [core.py:1349] In file included from /home/enginrect/llmso/.venv/lib/python3.12/site-packages/flashinfer/data/cccl/libcudacxx/include/cuda/std/__cccl/extended_data_types.h:26,
(EngineCore pid=13366) ERROR 08-22 20:21:50 [core.py:1349]                  from /home/enginrect/llmso/.venv/lib/python3.12/site-packages/flashinfer/data/cccl/libcudacxx/include/cuda/std/__cccl/builtin.h:26,
(EngineCore pid=13366) ERROR 08-22 20:21:50 [core.py:1349]                  from /home/enginrect/llmso/.venv/lib/python3.12/site-packages/flashinfer/data/cccl/libcudacxx/include/cuda/std/__cccl/dialect.h:25,
(EngineCore pid=13366) ERROR 08-22 20:21:50 [core.py:1349]                  from /home/enginrect/llmso/.venv/lib/python3.12/site-packages/flashinfer/data/cccl/libcudacxx/include/cuda/std/__cccl/attributes.h:26,
(EngineCore pid=13366) ERROR 08-22 20:21:50 [core.py:1349]                  from /home/enginrect/llmso/.venv/lib/python3.12/site-packages/flashinfer/data/cccl/libcudacxx/include/cuda/std/__cccl/assert.h:25,
(EngineCore pid=13366) ERROR 08-22 20:21:50 [core.py:1349]                  from /home/enginrect/llmso/.venv/lib/python3.12/site-packages/flashinfer/data/cccl/libcudacxx/include/cuda/__cccl_config:15,
(EngineCore pid=13366) ERROR 08-22 20:21:50 [core.py:1349]                  from /home/enginrect/llmso/.venv/lib/python3.12/site-packages/flashinfer/data/cccl/libcudacxx/include/cuda/std/__internal/atomic.h:13,
(EngineCore pid=13366) ERROR 08-22 20:21:50 [core.py:1349]                  from /home/enginrect/llmso/.venv/lib/python3.12/site-packages/flashinfer/data/cccl/libcudacxx/include/cuda/std/detail/__config:14,
(EngineCore pid=13366) ERROR 08-22 20:21:50 [core.py:1349]                  from /home/enginrect/llmso/.venv/lib/python3.12/site-packages/flashinfer/data/cccl/libcudacxx/include/cuda/std/type_traits:14,
(EngineCore pid=13366) ERROR 08-22 20:21:50 [core.py:1349]                  from /home/enginrect/llmso/.venv/lib/python3.12/site-packages/nvidia/cu13/include/cooperative_groups/details/info.h:167,
(EngineCore pid=13366) ERROR 08-22 20:21:50 [core.py:1349]                  from /home/enginrect/llmso/.venv/lib/python3.12/site-packages/nvidia/cu13/include/cooperative_groups.h:55,
(EngineCore pid=13366) ERROR 08-22 20:21:50 [core.py:1349]                  from /home/enginrect/llmso/.venv/lib/python3.12/site-packages/flashinfer/data/include/flashinfer/air_top_p.cuh:22,
(EngineCore pid=13366) ERROR 08-22 20:21:50 [core.py:1349]                  from /home/enginrect/llmso/.venv/lib/python3.12/site-packages/flashinfer/data/csrc/renorm.cu:16:
(EngineCore pid=13366) ERROR 08-22 20:21:50 [core.py:1349] /home/enginrect/llmso/.venv/lib/python3.12/site-packages/flashinfer/data/cccl/libcudacxx/include/cuda/std/__cccl/cuda_toolkit.h:41:8: error: #error "CUDA compiler and CUDA toolkit headers are incompatible, please check your include paths"
(EngineCore pid=13366) ERROR 08-22 20:21:50 [core.py:1349]    41 | #      error "CUDA compiler and CUDA toolkit headers are incompatible, please check your include paths"
(EngineCore pid=13366) ERROR 08-22 20:21:50 [core.py:1349]       |        ^~~~~
(EngineCore pid=13366) ERROR 08-22 20:21:50 [core.py:1349] [3/4]  /home/enginrect/llmso/.venv/lib/python3.12/site-packages/nvidia/cu13/bin/nvcc --generate-dependencies-with-compile -MF /home/enginrect/.cache/flashinfer/0.6.16.post3/89/cached_ops/sampling/csrc_flashinfer_sampling_binding.cuda.o.d -DPy_LIMITED_API=0x03090000 -D_GLIBCXX_USE_CXX11_ABI=1 -I/home/enginrect/llmso/.venv/lib/python3.12/site-packages/flashinfer/data/cccl/cub -I/home/enginrect/llmso/.venv/lib/python3.12/site-packages/flashinfer/data/cccl/libcudacxx/include -I/home/enginrect/llmso/.venv/lib/python3.12/site-packages/flashinfer/data/cccl/thrust -isystem /usr/include/python3.12 -isystem /home/enginrect/llmso/.venv/lib/python3.12/site-packages/nvidia/cu13/include -isystem /home/enginrect/llmso/.venv/lib/python3.12/site-packages/tvm_ffi/include -isystem /home/enginrect/llmso/.venv/lib/python3.12/site-packages/tvm_ffi/include -isystem /home/enginrect/llmso/.venv/lib/python3.12/site-packages/flashinfer/data/include -isystem /home/enginrect/llmso/.venv/lib/python3.12/site-packages/flashinfer/data/csrc -isystem /home/enginrect/llmso/.venv/lib/python3.12/site-packages/flashinfer/data/cutlass/include -isystem /home/enginrect/llmso/.venv/lib/python3.12/site-packages/flashinfer/data/cutlass/tools/util/include -isystem /home/enginrect/llmso/.venv/lib/python3.12/site-packages/flashinfer/data/spdlog/include --compiler-options=-fPIC --expt-relaxed-constexpr -static-global-template-stub=false -gencode=arch=compute_89,code=sm_89 -DFLASHINFER_ENABLE_FP8_E8M0 -DFLASHINFER_ENABLE_FP4_E2M1 -std=c++17 --threads=1 -use_fast_math -Xfatbin=-compress-all -DFLASHINFER_ENABLE_F16 -DFLASHINFER_ENABLE_BF16 -DFLASHINFER_ENABLE_FP8_E4M3 -DFLASHINFER_ENABLE_FP8_E5M2 -DNDEBUG -O3 -c /home/enginrect/llmso/.venv/lib/python3.12/site-packages/flashinfer/data/csrc/flashinfer_sampling_binding.cu -o /home/enginrect/.cache/flashinfer/0.6.16.post3/89/cached_ops/sampling/csrc_flashinfer_sampling_binding.cuda.o 
(EngineCore pid=13366) ERROR 08-22 20:21:50 [core.py:1349] ninja: build stopped: subcommand failed.
(EngineCore pid=13366) ERROR 08-22 20:21:50 [core.py:1349] 
...
(EngineCore pid=13366) Traceback (most recent call last):
...
(EngineCore pid=13366)     raise CalledProcessError(retcode, process.args,
(EngineCore pid=13366) subprocess.CalledProcessError: Command '['ninja', '-v', '-C', '/home/enginrect/.cache/flashinfer/0.6.16.post3/89/cached_ops/sampling', '-f', '/home/enginrect/.cache/flashinfer/0.6.16.post3/89/cached_ops/sampling/build.ninja']' returned non-zero exit status 1.
...
(EngineCore pid=13366) Traceback (most recent call last):
...
(EngineCore pid=13366)     raise e
...
(EngineCore pid=13366)     raise RuntimeError(msg) from e
(EngineCore pid=13366) RuntimeError: Ninja build failed. Ninja output:
...
(EngineCore pid=13366) FAILED: [code=1] /home/enginrect/.cache/flashinfer/0.6.16.post3/89/cached_ops/sampling/csrc_sampling.cuda.o 
...
(EngineCore pid=13366) /home/enginrect/llmso/.venv/lib/python3.12/site-packages/flashinfer/data/cccl/libcudacxx/include/cuda/std/__cccl/cuda_toolkit.h:41:8: error: #error "CUDA compiler and CUDA toolkit headers are incompatible, please check your include paths"
...
(EngineCore pid=13366) FAILED: [code=1] /home/enginrect/.cache/flashinfer/0.6.16.post3/89/cached_ops/sampling/csrc_renorm.cuda.o 
...
(EngineCore pid=13366) /home/enginrect/llmso/.venv/lib/python3.12/site-packages/flashinfer/data/cccl/libcudacxx/include/cuda/std/__cccl/cuda_toolkit.h:41:8: error: #error "CUDA compiler and CUDA toolkit headers are incompatible, please check your include paths"
...
(EngineCore pid=13366) ninja: build stopped: subcommand failed.
...
(APIServer pid=13257) Traceback (most recent call last):
...
(APIServer pid=13257)   File "/home/enginrect/llmso/.venv/lib/python3.12/site-packages/vllm/v1/engine/utils.py", line 1253, in wait_for_engine_startup
(APIServer pid=13257)     raise RuntimeError(
(APIServer pid=13257) RuntimeError: Engine core initialization failed. See root cause above. Failed core proc(s): {}
```

**원인 분석**: flashinfer가 동봉한 CCCL 헤더의 버전 가드가 컴파일을 거부했다. venv의 nvidia 패키지 버전을 확인하니:

```shell
$ grep -m1 "define CUDART_VERSION" .venv/.../nvidia/cu13/include/cuda_runtime_api.h
#define CUDART_VERSION  13000
$ ls .venv/lib/python3.12/site-packages/ | grep -iE "nvidia.*dist-info"
nvidia_cublas-13.1.1.3.dist-info
nvidia_cuda_cccl-13.3.3.4.1.dist-info
nvidia_cuda_crt-13.3.73.dist-info
nvidia_cuda_cupti-13.0.85.dist-info
nvidia_cuda_nvcc-13.3.73.dist-info
nvidia_cuda_nvdisasm-13.3.73.dist-info
nvidia_cuda_nvrtc-13.0.88.dist-info
nvidia_cuda_runtime-13.0.96.dist-info
nvidia_cudnn_cu13-9.20.0.48.dist-info
nvidia_cudnn_frontend-1.27.0.dist-info
nvidia_cufft-12.0.0.61.dist-info
nvidia_cufile-1.15.1.6.dist-info
nvidia_curand-10.4.0.35.dist-info
nvidia_cusolver-12.0.4.66.dist-info
nvidia_cusparse-12.6.3.3.dist-info
nvidia_cusparselt_cu13-0.8.1.dist-info
nvidia_cutlass_dsl-4.6.0.dist-info
nvidia_cutlass_dsl_libs_base-4.6.0.dist-info
nvidia_cutlass_dsl_libs_core-4.6.0.dist-info
nvidia_cutlass_dsl_libs_cu12-4.6.0.dist-info
nvidia_cutlass_dsl_libs_cu13-4.6.0.dist-info
nvidia_ml_py-13.610.43.dist-info
nvidia_nccl_cu13-2.29.7.dist-info
nvidia_nvjitlink-13.3.33.dist-info
nvidia_nvshmem_cu13-3.4.5.dist-info
nvidia_nvtx-13.0.85.dist-info
nvidia_nvvm-13.3.73.dist-info
```

**nvcc 계열(nvcc/crt/nvvm)은 13.3.73인데 cuda-runtime(헤더 제공)은 13.0.96** — uv의 의존성 해석이 컴파일러와 헤더를 서로 다른 마이너 버전으로 풀었다. torch가 runtime을 13.0으로 핀하는 반면 nvcc는 느슨하게 풀려 최신(13.3)이 잡힌 것. CCCL 가드는 "컴파일러 13.3 + cudart 헤더 13.0" 조합을 비호환으로 거부한다.

### Step 11 — nvcc 계열을 13.0으로 다운그레이드

```shell
$ uv pip install "nvidia-cuda-nvcc==13.0.*" "nvidia-cuda-crt==13.0.*" "nvidia-nvvm==13.0.*"
Resolved 4 packages in 49ms
Downloading nvidia-cuda-nvcc (35.7MiB)
Downloading nvidia-nvvm (58.7MiB)
 Downloaded nvidia-cuda-nvcc
 Downloaded nvidia-nvvm
Prepared 3 packages in 894ms
Uninstalled 3 packages in 1ms
Installed 3 packages in 4ms
 - nvidia-cuda-crt==13.3.73
 + nvidia-cuda-crt==13.0.88
 - nvidia-cuda-nvcc==13.3.73
 + nvidia-cuda-nvcc==13.0.88
 - nvidia-nvvm==13.3.73
 + nvidia-nvvm==13.0.88

$ nvcc --version (downgrade 후)
Cuda compilation tools, release 13.0, V13.0.88
Build cuda_13.0.r13.0/compiler.36424714_0
```

- key point: nvcc/crt/nvvm 3종을 13.0.88로 핀해 cudart 13.0.96 헤더와 정렬했다. 실패한 JIT 캐시는 `rm -rf ~/.cache/flashinfer`로 비우고 재시도.

### 시도 6 — ❌ `cannot find -lcudart / -lcuda` (링크 실패)

출력 (발췌):

```shell
(APIServer pid=14001) INFO 08-22 20:23:13 [api_utils.py:345] 
(APIServer pid=14001) INFO 08-22 20:23:13 [api_utils.py:345]        █     █     █▄   ▄█
(APIServer pid=14001) INFO 08-22 20:23:13 [api_utils.py:345]  ▄▄ ▄█ █     █     █ ▀▄▀ █  version 0.27.1
(APIServer pid=14001) INFO 08-22 20:23:13 [api_utils.py:345]   █▄█▀ █     █     █     █  model   Qwen/Qwen2.5-0.5B-Instruct
...
(APIServer pid=14001) INFO 08-22 20:23:13 [api_utils.py:273] non-default args: {'model_tag': 'Qwen/Qwen2.5-0.5B-Instruct', 'model': 'Qwen/Qwen2.5-0.5B-Instruct', 'max_model_len': 2048, 'gpu_memory_utilization': 0.85}
...
(EngineCore pid=14108) INFO 08-22 20:23:26 [weight_utils.py:867] Filesystem type for checkpoints: EXT4. Checkpoint size: 0.92 GiB. Available RAM: 27.40 GiB.
...
(EngineCore pid=14108) INFO 08-22 20:23:26 [model_runner.py:329] Model loading took 0.93 GiB and 2.005452 seconds
...
(EngineCore pid=14108) INFO 08-22 20:23:27 [gpu_worker.py:563] Available KV cache memory: 19.06 GiB
(EngineCore pid=14108) INFO 08-22 20:23:27 [kv_cache_utils.py:2235] GPU KV cache size: 1,665,392 tokens
...
(EngineCore pid=14108) INFO 08-22 20:23:29 [model_runner.py:791] Graph capturing finished in 2 secs, took 0.17 GiB
(EngineCore pid=14108) INFO 08-22 20:23:29 [gpu_worker.py:789] Free memory on device (22.45/23.99 GiB) on startup. Desired GPU memory utilization is (0.85, 20.39 GiB). Actual usage is 1.25 GiB for consumed memory (weights + non-torch), 0.08 GiB for peak activation, and 0.17 GiB for CUDAGraph memory. Replace gpu_memory_utilization config with `--kv-cache-memory=20128526234` (18.75 GiB) to fit into requested memory, or `--kv-cache-memory=22339999232` (20.81 GiB) to fully utilize gpu memory. Current kv cache memory in use is 19.06 GiB.
(EngineCore pid=14108) ERROR 08-22 20:24:11 [core.py:1349] EngineCore failed to start.
(EngineCore pid=14108) ERROR 08-22 20:24:11 [core.py:1349] Traceback (most recent call last):
(EngineCore pid=14108) ERROR 08-22 20:24:11 [core.py:1349]   File "/home/enginrect/llmso/.venv/lib/python3.12/site-packages/flashinfer/jit/cpp_ext.py", line 370, in run_ninja
(EngineCore pid=14108) ERROR 08-22 20:24:11 [core.py:1349]     subprocess.run(
(EngineCore pid=14108) ERROR 08-22 20:24:11 [core.py:1349]   File "/usr/lib/python3.12/subprocess.py", line 571, in run
(EngineCore pid=14108) ERROR 08-22 20:24:11 [core.py:1349]     raise CalledProcessError(retcode, process.args,
(EngineCore pid=14108) ERROR 08-22 20:24:11 [core.py:1349] subprocess.CalledProcessError: Command '['ninja', '-v', '-C', '/home/enginrect/.cache/flashinfer/0.6.16.post3/89/cached_ops/sampling', '-f', '/home/enginrect/.cache/flashinfer/0.6.16.post3/89/cached_ops/sampling/build.ninja']' returned non-zero exit status 1.
(EngineCore pid=14108) ERROR 08-22 20:24:11 [core.py:1349] 
(EngineCore pid=14108) ERROR 08-22 20:24:11 [core.py:1349] The above exception was the direct cause of the following exception:
(EngineCore pid=14108) ERROR 08-22 20:24:11 [core.py:1349] 
(EngineCore pid=14108) ERROR 08-22 20:24:11 [core.py:1349] Traceback (most recent call last):
(EngineCore pid=14108) ERROR 08-22 20:24:11 [core.py:1349]   File "/home/enginrect/llmso/.venv/lib/python3.12/site-packages/vllm/v1/engine/core.py", line 1318, in run_engine_core
(EngineCore pid=14108) ERROR 08-22 20:24:11 [core.py:1349]     engine_core = EngineCoreProc(*args, engine_index=dp_rank, **kwargs)
(EngineCore pid=14108) ERROR 08-22 20:24:11 [core.py:1349]                   ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
(EngineCore pid=14108) ERROR 08-22 20:24:11 [core.py:1349]   File "/home/enginrect/llmso/.venv/lib/python3.12/site-packages/vllm/tracing/otel.py", line 178, in sync_wrapper
(EngineCore pid=14108) ERROR 08-22 20:24:11 [core.py:1349]     return func(*args, **kwargs)
(EngineCore pid=14108) ERROR 08-22 20:24:11 [core.py:1349]            ^^^^^^^^^^^^^^^^^^^^^
(EngineCore pid=14108) ERROR 08-22 20:24:11 [core.py:1349]   File "/home/enginrect/llmso/.venv/lib/python3.12/site-packages/vllm/v1/engine/core.py", line 1074, in __init__
(EngineCore pid=14108) ERROR 08-22 20:24:11 [core.py:1349]     super().__init__(
(EngineCore pid=14108) ERROR 08-22 20:24:11 [core.py:1349]   File "/home/enginrect/llmso/.venv/lib/python3.12/site-packages/vllm/v1/engine/core.py", line 143, in __init__
(EngineCore pid=14108) ERROR 08-22 20:24:11 [core.py:1349]     kv_cache_config = self._initialize_kv_caches(vllm_config)
(EngineCore pid=14108) ERROR 08-22 20:24:11 [core.py:1349]                       ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
(EngineCore pid=14108) ERROR 08-22 20:24:11 [core.py:1349]   File "/home/enginrect/llmso/.venv/lib/python3.12/site-packages/vllm/tracing/otel.py", line 178, in sync_wrapper
(EngineCore pid=14108) ERROR 08-22 20:24:11 [core.py:1349]     return func(*args, **kwargs)
(EngineCore pid=14108) ERROR 08-22 20:24:11 [core.py:1349]            ^^^^^^^^^^^^^^^^^^^^^
(EngineCore pid=14108) ERROR 08-22 20:24:11 [core.py:1349]   File "/home/enginrect/llmso/.venv/lib/python3.12/site-packages/vllm/v1/engine/core.py", line 332, in _initialize_kv_caches
(EngineCore pid=14108) ERROR 08-22 20:24:11 [core.py:1349]     self.model_executor.compile_or_warm_up_model()
(EngineCore pid=14108) ERROR 08-22 20:24:11 [core.py:1349]   File "/home/enginrect/llmso/.venv/lib/python3.12/site-packages/vllm/v1/executor/abstract.py", line 124, in compile_or_warm_up_model
(EngineCore pid=14108) ERROR 08-22 20:24:11 [core.py:1349]     compilation_times: list[CompilationTimes] = self.collective_rpc(
(EngineCore pid=14108) ERROR 08-22 20:24:11 [core.py:1349]                                                 ^^^^^^^^^^^^^^^^^^^^
(EngineCore pid=14108) ERROR 08-22 20:24:11 [core.py:1349]   File "/home/enginrect/llmso/.venv/lib/python3.12/site-packages/vllm/v1/executor/uniproc_executor.py", line 92, in collective_rpc
(EngineCore pid=14108) ERROR 08-22 20:24:11 [core.py:1349]     result = run_method(self.driver_worker, method, args, kwargs)
(EngineCore pid=14108) ERROR 08-22 20:24:11 [core.py:1349]              ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
(EngineCore pid=14108) ERROR 08-22 20:24:11 [core.py:1349]   File "/home/enginrect/llmso/.venv/lib/python3.12/site-packages/vllm/v1/serial_utils.py", line 510, in run_method
(EngineCore pid=14108) ERROR 08-22 20:24:11 [core.py:1349]     return func(*args, **kwargs)
(EngineCore pid=14108) ERROR 08-22 20:24:11 [core.py:1349]            ^^^^^^^^^^^^^^^^^^^^^
(EngineCore pid=14108) ERROR 08-22 20:24:11 [core.py:1349]   File "/home/enginrect/llmso/.venv/lib/python3.12/site-packages/vllm/tracing/otel.py", line 178, in sync_wrapper
(EngineCore pid=14108) ERROR 08-22 20:24:11 [core.py:1349]     return func(*args, **kwargs)
(EngineCore pid=14108) ERROR 08-22 20:24:11 [core.py:1349]            ^^^^^^^^^^^^^^^^^^^^^
(EngineCore pid=14108) ERROR 08-22 20:24:11 [core.py:1349]   File "/home/enginrect/llmso/.venv/lib/python3.12/site-packages/vllm/v1/worker/gpu_worker.py", line 795, in compile_or_warm_up_model
(EngineCore pid=14108) ERROR 08-22 20:24:11 [core.py:1349]     warmup_kernels(self.model_runner, self.execute_model, self.sample_tokens)
(EngineCore pid=14108) ERROR 08-22 20:24:11 [core.py:1349]   File "/home/enginrect/llmso/.venv/lib/python3.12/site-packages/torch/utils/_contextlib.py", line 124, in decorate_context
(EngineCore pid=14108) ERROR 08-22 20:24:11 [core.py:1349]     return func(*args, **kwargs)
(EngineCore pid=14108) ERROR 08-22 20:24:11 [core.py:1349]            ^^^^^^^^^^^^^^^^^^^^^
(EngineCore pid=14108) ERROR 08-22 20:24:11 [core.py:1349]   File "/home/enginrect/llmso/.venv/lib/python3.12/site-packages/vllm/v1/worker/gpu/warmup.py", line 296, in warmup_kernels
(EngineCore pid=14108) ERROR 08-22 20:24:11 [core.py:1349]     worker_sample_tokens(grammar_output)
(EngineCore pid=14108) ERROR 08-22 20:24:11 [core.py:1349]   File "/home/enginrect/llmso/.venv/lib/python3.12/site-packages/torch/utils/_contextlib.py", line 124, in decorate_context
(EngineCore pid=14108) ERROR 08-22 20:24:11 [core.py:1349]     return func(*args, **kwargs)
(EngineCore pid=14108) ERROR 08-22 20:24:11 [core.py:1349]            ^^^^^^^^^^^^^^^^^^^^^
(EngineCore pid=14108) ERROR 08-22 20:24:11 [core.py:1349]   File "/home/enginrect/llmso/.venv/lib/python3.12/site-packages/vllm/v1/worker/gpu_worker.py", line 1015, in sample_tokens
(EngineCore pid=14108) ERROR 08-22 20:24:11 [core.py:1349]     return self.model_runner.sample_tokens(grammar_output)
(EngineCore pid=14108) ERROR 08-22 20:24:11 [core.py:1349]            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
(EngineCore pid=14108) ERROR 08-22 20:24:11 [core.py:1349]   File "/home/enginrect/llmso/.venv/lib/python3.12/site-packages/torch/utils/_contextlib.py", line 124, in decorate_context
(EngineCore pid=14108) ERROR 08-22 20:24:11 [core.py:1349]     return func(*args, **kwargs)
(EngineCore pid=14108) ERROR 08-22 20:24:11 [core.py:1349]            ^^^^^^^^^^^^^^^^^^^^^
(EngineCore pid=14108) ERROR 08-22 20:24:11 [core.py:1349]   File "/home/enginrect/llmso/.venv/lib/python3.12/site-packages/vllm/v1/worker/gpu/eplb_utils.py", line 38, in wrapper
(EngineCore pid=14108) ERROR 08-22 20:24:11 [core.py:1349]     result = fn(self, *args, **kwargs)
(EngineCore pid=14108) ERROR 08-22 20:24:11 [core.py:1349]              ^^^^^^^^^^^^^^^^^^^^^^^^^
(EngineCore pid=14108) ERROR 08-22 20:24:11 [core.py:1349]   File "/home/enginrect/llmso/.venv/lib/python3.12/site-packages/vllm/v1/worker/gpu/model_runner.py", line 1495, in sample_tokens
(EngineCore pid=14108) ERROR 08-22 20:24:11 [core.py:1349]     sampler_output, num_sampled, num_rejected = self.sample(
(EngineCore pid=14108) ERROR 08-22 20:24:11 [core.py:1349]                                                 ^^^^^^^^^^^^
(EngineCore pid=14108) ERROR 08-22 20:24:11 [core.py:1349]   File "/home/enginrect/llmso/.venv/lib/python3.12/site-packages/vllm/v1/worker/gpu/model_runner.py", line 1163, in sample
(EngineCore pid=14108) ERROR 08-22 20:24:11 [core.py:1349]     sampler_output = self.sampler(logits, input_batch)
(EngineCore pid=14108) ERROR 08-22 20:24:11 [core.py:1349]                      ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
(EngineCore pid=14108) ERROR 08-22 20:24:11 [core.py:1349]   File "/home/enginrect/llmso/.venv/lib/python3.12/site-packages/vllm/v1/worker/gpu/sample/sampler.py", line 92, in __call__
(EngineCore pid=14108) ERROR 08-22 20:24:11 [core.py:1349]     sampled, processed_logits = self.sample(
(EngineCore pid=14108) ERROR 08-22 20:24:11 [core.py:1349]                                 ^^^^^^^^^^^^
(EngineCore pid=14108) ERROR 08-22 20:24:11 [core.py:1349]   File "/home/enginrect/llmso/.venv/lib/python3.12/site-packages/vllm/v1/worker/gpu/sample/sampler.py", line 252, in sample
(EngineCore pid=14108) ERROR 08-22 20:24:11 [core.py:1349]     sampled = flashinfer_sample(processed_logits, top_k, top_p).to(torch.int64)
(EngineCore pid=14108) ERROR 08-22 20:24:11 [core.py:1349]               ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
(EngineCore pid=14108) ERROR 08-22 20:24:11 [core.py:1349]   File "/home/enginrect/llmso/.venv/lib/python3.12/site-packages/vllm/v1/sample/ops/topk_topp_sampler.py", line 508, in flashinfer_sample
(EngineCore pid=14108) ERROR 08-22 20:24:11 [core.py:1349]     next_token_ids = flashinfer.sampling.top_k_top_p_sampling_from_logits(
(EngineCore pid=14108) ERROR 08-22 20:24:11 [core.py:1349]                      ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
(EngineCore pid=14108) ERROR 08-22 20:24:11 [core.py:1349]   File "/home/enginrect/llmso/.venv/lib/python3.12/site-packages/flashinfer/api_logging.py", line 2333, in _auto_dump_wrapper
(EngineCore pid=14108) ERROR 08-22 20:24:11 [core.py:1349]     return _inner(*args, **kwargs)
(EngineCore pid=14108) ERROR 08-22 20:24:11 [core.py:1349]            ^^^^^^^^^^^^^^^^^^^^^^^
(EngineCore pid=14108) ERROR 08-22 20:24:11 [core.py:1349]   File "/home/enginrect/llmso/.venv/lib/python3.12/site-packages/flashinfer/sampling.py", line 1547, in top_k_top_p_sampling_from_logits
(EngineCore pid=14108) ERROR 08-22 20:24:11 [core.py:1349]     masked_logits = top_k_mask_logits(logits, top_k)
(EngineCore pid=14108) ERROR 08-22 20:24:11 [core.py:1349]                     ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
(EngineCore pid=14108) ERROR 08-22 20:24:11 [core.py:1349]   File "/home/enginrect/llmso/.venv/lib/python3.12/site-packages/flashinfer/api_logging.py", line 2333, in _auto_dump_wrapper
(EngineCore pid=14108) ERROR 08-22 20:24:11 [core.py:1349]     return _inner(*args, **kwargs)
(EngineCore pid=14108) ERROR 08-22 20:24:11 [core.py:1349]            ^^^^^^^^^^^^^^^^^^^^^^^
(EngineCore pid=14108) ERROR 08-22 20:24:11 [core.py:1349]   File "/home/enginrect/llmso/.venv/lib/python3.12/site-packages/flashinfer/sampling.py", line 1974, in top_k_mask_logits
(EngineCore pid=14108) ERROR 08-22 20:24:11 [core.py:1349]     return get_sampling_module().top_k_mask_logits(
(EngineCore pid=14108) ERROR 08-22 20:24:11 [core.py:1349]            ^^^^^^^^^^^^^^^^^^^^^
(EngineCore pid=14108) ERROR 08-22 20:24:11 [core.py:1349]   File "/home/enginrect/llmso/.venv/lib/python3.12/site-packages/flashinfer/sampling.py", line 68, in get_sampling_module
(EngineCore pid=14108) ERROR 08-22 20:24:11 [core.py:1349]     module = gen_sampling_module().build_and_load()
(EngineCore pid=14108) ERROR 08-22 20:24:11 [core.py:1349]              ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
(EngineCore pid=14108) ERROR 08-22 20:24:11 [core.py:1349]   File "/home/enginrect/llmso/.venv/lib/python3.12/site-packages/flashinfer/jit/core.py", line 320, in build_and_load
(EngineCore pid=14108) ERROR 08-22 20:24:11 [core.py:1349]     self.build()
(EngineCore pid=14108) ERROR 08-22 20:24:11 [core.py:1349]   File "/home/enginrect/llmso/.venv/lib/python3.12/site-packages/flashinfer/jit/core.py", line 427, in build
(EngineCore pid=14108) ERROR 08-22 20:24:11 [core.py:1349]     run_ninja(self.build_dir, self.ninja_path, verbose)
(EngineCore pid=14108) ERROR 08-22 20:24:11 [core.py:1349]   File "/home/enginrect/llmso/.venv/lib/python3.12/site-packages/flashinfer/jit/cpp_ext.py", line 382, in run_ninja
(EngineCore pid=14108) ERROR 08-22 20:24:11 [core.py:1349]     raise RuntimeError(msg) from e
(EngineCore pid=14108) ERROR 08-22 20:24:11 [core.py:1349] RuntimeError: Ninja build failed. Ninja output:
(EngineCore pid=14108) ERROR 08-22 20:24:11 [core.py:1349] ninja: Entering directory `/home/enginrect/.cache/flashinfer/0.6.16.post3/89/cached_ops/sampling'
(EngineCore pid=14108) ERROR 08-22 20:24:11 [core.py:1349] [1/4]  /home/enginrect/llmso/.venv/lib/python3.12/site-packages/nvidia/cu13/bin/nvcc --generate-dependencies-with-compile -MF /home/enginrect/.cache/flashinfer/0.6.16.post3/89/cached_ops/sampling/csrc_flashinfer_sampling_binding.cuda.o.d -DPy_LIMITED_API=0x03090000 -D_GLIBCXX_USE_CXX11_ABI=1 -I/home/enginrect/llmso/.venv/lib/python3.12/site-packages/flashinfer/data/cccl/cub -I/home/enginrect/llmso/.venv/lib/python3.12/site-packages/flashinfer/data/cccl/libcudacxx/include -I/home/enginrect/llmso/.venv/lib/python3.12/site-packages/flashinfer/data/cccl/thrust -isystem /usr/include/python3.12 -isystem /home/enginrect/llmso/.venv/lib/python3.12/site-packages/nvidia/cu13/include -isystem /home/enginrect/llmso/.venv/lib/python3.12/site-packages/tvm_ffi/include -isystem /home/enginrect/llmso/.venv/lib/python3.12/site-packages/tvm_ffi/include -isystem /home/enginrect/llmso/.venv/lib/python3.12/site-packages/flashinfer/data/include -isystem /home/enginrect/llmso/.venv/lib/python3.12/site-packages/flashinfer/data/csrc -isystem /home/enginrect/llmso/.venv/lib/python3.12/site-packages/flashinfer/data/cutlass/include -isystem /home/enginrect/llmso/.venv/lib/python3.12/site-packages/flashinfer/data/cutlass/tools/util/include -isystem /home/enginrect/llmso/.venv/lib/python3.12/site-packages/flashinfer/data/spdlog/include --compiler-options=-fPIC --expt-relaxed-constexpr -static-global-template-stub=false -gencode=arch=compute_89,code=sm_89 -DFLASHINFER_ENABLE_FP8_E8M0 -DFLASHINFER_ENABLE_FP4_E2M1 -std=c++17 --threads=1 -use_fast_math -Xfatbin=-compress-all -DFLASHINFER_ENABLE_F16 -DFLASHINFER_ENABLE_BF16 -DFLASHINFER_ENABLE_FP8_E4M3 -DFLASHINFER_ENABLE_FP8_E5M2 -DNDEBUG -O3 -c /home/enginrect/llmso/.venv/lib/python3.12/site-packages/flashinfer/data/csrc/flashinfer_sampling_binding.cu -o /home/enginrect/.cache/flashinfer/0.6.16.post3/89/cached_ops/sampling/csrc_flashinfer_sampling_binding.cuda.o 
(EngineCore pid=14108) ERROR 08-22 20:24:11 [core.py:1349] [2/4]  /home/enginrect/llmso/.venv/lib/python3.12/site-packages/nvidia/cu13/bin/nvcc --generate-dependencies-with-compile -MF /home/enginrect/.cache/flashinfer/0.6.16.post3/89/cached_ops/sampling/csrc_renorm.cuda.o.d -DPy_LIMITED_API=0x03090000 -D_GLIBCXX_USE_CXX11_ABI=1 -I/home/enginrect/llmso/.venv/lib/python3.12/site-packages/flashinfer/data/cccl/cub -I/home/enginrect/llmso/.venv/lib/python3.12/site-packages/flashinfer/data/cccl/libcudacxx/include -I/home/enginrect/llmso/.venv/lib/python3.12/site-packages/flashinfer/data/cccl/thrust -isystem /usr/include/python3.12 -isystem /home/enginrect/llmso/.venv/lib/python3.12/site-packages/nvidia/cu13/include -isystem /home/enginrect/llmso/.venv/lib/python3.12/site-packages/tvm_ffi/include -isystem /home/enginrect/llmso/.venv/lib/python3.12/site-packages/tvm_ffi/include -isystem /home/enginrect/llmso/.venv/lib/python3.12/site-packages/flashinfer/data/include -isystem /home/enginrect/llmso/.venv/lib/python3.12/site-packages/flashinfer/data/csrc -isystem /home/enginrect/llmso/.venv/lib/python3.12/site-packages/flashinfer/data/cutlass/include -isystem /home/enginrect/llmso/.venv/lib/python3.12/site-packages/flashinfer/data/cutlass/tools/util/include -isystem /home/enginrect/llmso/.venv/lib/python3.12/site-packages/flashinfer/data/spdlog/include --compiler-options=-fPIC --expt-relaxed-constexpr -static-global-template-stub=false -gencode=arch=compute_89,code=sm_89 -DFLASHINFER_ENABLE_FP8_E8M0 -DFLASHINFER_ENABLE_FP4_E2M1 -std=c++17 --threads=1 -use_fast_math -Xfatbin=-compress-all -DFLASHINFER_ENABLE_F16 -DFLASHINFER_ENABLE_BF16 -DFLASHINFER_ENABLE_FP8_E4M3 -DFLASHINFER_ENABLE_FP8_E5M2 -DNDEBUG -O3 -c /home/enginrect/llmso/.venv/lib/python3.12/site-packages/flashinfer/data/csrc/renorm.cu -o /home/enginrect/.cache/flashinfer/0.6.16.post3/89/cached_ops/sampling/csrc_renorm.cuda.o 
(EngineCore pid=14108) ERROR 08-22 20:24:11 [core.py:1349] [3/4]  /home/enginrect/llmso/.venv/lib/python3.12/site-packages/nvidia/cu13/bin/nvcc --generate-dependencies-with-compile -MF /home/enginrect/.cache/flashinfer/0.6.16.post3/89/cached_ops/sampling/csrc_sampling.cuda.o.d -DPy_LIMITED_API=0x03090000 -D_GLIBCXX_USE_CXX11_ABI=1 -I/home/enginrect/llmso/.venv/lib/python3.12/site-packages/flashinfer/data/cccl/cub -I/home/enginrect/llmso/.venv/lib/python3.12/site-packages/flashinfer/data/cccl/libcudacxx/include -I/home/enginrect/llmso/.venv/lib/python3.12/site-packages/flashinfer/data/cccl/thrust -isystem /usr/include/python3.12 -isystem /home/enginrect/llmso/.venv/lib/python3.12/site-packages/nvidia/cu13/include -isystem /home/enginrect/llmso/.venv/lib/python3.12/site-packages/tvm_ffi/include -isystem /home/enginrect/llmso/.venv/lib/python3.12/site-packages/tvm_ffi/include -isystem /home/enginrect/llmso/.venv/lib/python3.12/site-packages/flashinfer/data/include -isystem /home/enginrect/llmso/.venv/lib/python3.12/site-packages/flashinfer/data/csrc -isystem /home/enginrect/llmso/.venv/lib/python3.12/site-packages/flashinfer/data/cutlass/include -isystem /home/enginrect/llmso/.venv/lib/python3.12/site-packages/flashinfer/data/cutlass/tools/util/include -isystem /home/enginrect/llmso/.venv/lib/python3.12/site-packages/flashinfer/data/spdlog/include --compiler-options=-fPIC --expt-relaxed-constexpr -static-global-template-stub=false -gencode=arch=compute_89,code=sm_89 -DFLASHINFER_ENABLE_FP8_E8M0 -DFLASHINFER_ENABLE_FP4_E2M1 -std=c++17 --threads=1 -use_fast_math -Xfatbin=-compress-all -DFLASHINFER_ENABLE_F16 -DFLASHINFER_ENABLE_BF16 -DFLASHINFER_ENABLE_FP8_E4M3 -DFLASHINFER_ENABLE_FP8_E5M2 -DNDEBUG -O3 -c /home/enginrect/llmso/.venv/lib/python3.12/site-packages/flashinfer/data/csrc/sampling.cu -o /home/enginrect/.cache/flashinfer/0.6.16.post3/89/cached_ops/sampling/csrc_sampling.cuda.o 
(EngineCore pid=14108) ERROR 08-22 20:24:11 [core.py:1349] [4/4] c++ /home/enginrect/.cache/flashinfer/0.6.16.post3/89/cached_ops/sampling/csrc_sampling.cuda.o /home/enginrect/.cache/flashinfer/0.6.16.post3/89/cached_ops/sampling/csrc_renorm.cuda.o /home/enginrect/.cache/flashinfer/0.6.16.post3/89/cached_ops/sampling/csrc_flashinfer_sampling_binding.cuda.o -shared -L/home/enginrect/llmso/.venv/lib/python3.12/site-packages/nvidia/cu13/lib64 -L/home/enginrect/llmso/.venv/lib/python3.12/site-packages/nvidia/cu13/lib64/stubs -lcudart -lcuda -o /home/enginrect/.cache/flashinfer/0.6.16.post3/89/cached_ops/sampling/sampling.so
(EngineCore pid=14108) ERROR 08-22 20:24:11 [core.py:1349] FAILED: [code=1] /home/enginrect/.cache/flashinfer/0.6.16.post3/89/cached_ops/sampling/sampling.so 
(EngineCore pid=14108) ERROR 08-22 20:24:11 [core.py:1349] c++ /home/enginrect/.cache/flashinfer/0.6.16.post3/89/cached_ops/sampling/csrc_sampling.cuda.o /home/enginrect/.cache/flashinfer/0.6.16.post3/89/cached_ops/sampling/csrc_renorm.cuda.o /home/enginrect/.cache/flashinfer/0.6.16.post3/89/cached_ops/sampling/csrc_flashinfer_sampling_binding.cuda.o -shared -L/home/enginrect/llmso/.venv/lib/python3.12/site-packages/nvidia/cu13/lib64 -L/home/enginrect/llmso/.venv/lib/python3.12/site-packages/nvidia/cu13/lib64/stubs -lcudart -lcuda -o /home/enginrect/.cache/flashinfer/0.6.16.post3/89/cached_ops/sampling/sampling.so
(EngineCore pid=14108) ERROR 08-22 20:24:11 [core.py:1349] /usr/bin/ld: cannot find -lcudart: No such file or directory
(EngineCore pid=14108) ERROR 08-22 20:24:11 [core.py:1349] /usr/bin/ld: cannot find -lcuda: No such file or directory
(EngineCore pid=14108) ERROR 08-22 20:24:11 [core.py:1349] collect2: error: ld returned 1 exit status
(EngineCore pid=14108) ERROR 08-22 20:24:11 [core.py:1349] ninja: build stopped: subcommand failed.
(EngineCore pid=14108) ERROR 08-22 20:24:11 [core.py:1349] 
...
(EngineCore pid=14108) Traceback (most recent call last):
...
(EngineCore pid=14108)     raise CalledProcessError(retcode, process.args,
(EngineCore pid=14108) subprocess.CalledProcessError: Command '['ninja', '-v', '-C', '/home/enginrect/.cache/flashinfer/0.6.16.post3/89/cached_ops/sampling', '-f', '/home/enginrect/.cache/flashinfer/0.6.16.post3/89/cached_ops/sampling/build.ninja']' returned non-zero exit status 1.
...
(EngineCore pid=14108) Traceback (most recent call last):
...
(EngineCore pid=14108)     raise e
...
(EngineCore pid=14108)     raise RuntimeError(msg) from e
(EngineCore pid=14108) RuntimeError: Ninja build failed. Ninja output:
...
(EngineCore pid=14108) FAILED: [code=1] /home/enginrect/.cache/flashinfer/0.6.16.post3/89/cached_ops/sampling/sampling.so 
...
(EngineCore pid=14108) collect2: error: ld returned 1 exit status
(EngineCore pid=14108) ninja: build stopped: subcommand failed.
...
(APIServer pid=14001) Traceback (most recent call last):
...
(APIServer pid=14001)   File "/home/enginrect/llmso/.venv/lib/python3.12/site-packages/vllm/v1/engine/utils.py", line 1253, in wait_for_engine_startup
(APIServer pid=14001)     raise RuntimeError(
(APIServer pid=14001) RuntimeError: Engine core initialization failed. See root cause above. Failed core proc(s): {}
```

**원인 분석**: 큰 진전 — CUDA 컴파일 3개([1/4]~[3/4])는 전부 통과했고, 마지막 **링크**([4/4])만 실패했다. 링크 명령이 `-L$CUDA_HOME/lib64 -L$CUDA_HOME/lib64/stubs -lcudart -lcuda`를 쓰는데:

```shell
$ ls $CUDA_HOME/          # lib64가 없다 (pip 레이아웃은 lib)
bin  cccl  include  lib  nvvm
$ ls $CUDA_HOME/lib/ | grep -iE "cudart|cuda\.so"   # unversioned .so가 없다
libcudart.so.13
libcudart_static.a
$ ls $CUDA_HOME/lib/stubs/   # stubs 디렉토리 자체가 없다
$ ls /usr/lib/wsl/lib/ | grep cuda   # libcuda는 WSL 전용 경로에 있다
libcuda.so
libcuda.so.1
libcuda.so.1.1
libcudadebugger.so.1
```

세 가지가 겹쳤다: ①pip 레이아웃은 `lib`(≠`lib64`) ②링커용 unversioned `libcudart.so` 심링크가 없음 ③`libcuda.so`(드라이버 라이브러리)는 WSL 특수 경로 `/usr/lib/wsl/lib`에만 있음.

### Step 12 — 심링크 3개로 레이아웃 보정

```shell
$ pip CUDA 레이아웃 보정 (lib64/libcudart.so/stubs 심링크)
lib64 -> lib
libcudart.so -> libcudart.so.13
stubs/libcuda.so -> /usr/lib/wsl/lib/libcuda.so

$ 확인
lrwxrwxrwx 1 enginrect enginrect 15 Aug 22 20:24 /home/enginrect/llmso/.venv/lib/python3.12/site-packages/nvidia/cu13/lib64/libcudart.so -> libcudart.so.13
lrwxrwxrwx 1 enginrect enginrect 27 Aug 22 20:24 /home/enginrect/llmso/.venv/lib/python3.12/site-packages/nvidia/cu13/lib64/stubs/libcuda.so -> /usr/lib/wsl/lib/libcuda.so
```

- key point: apt 패키지 추가 설치 없이 심링크 3개로 flashinfer가 기대하는 CUDA 툴킷 레이아웃을 완성했다.

### 시도 7 — ✅ 기동 성공

출력 (성공이라 로그가 짧다):

```shell
(APIServer pid=14678) INFO 08-22 20:24:59 [api_utils.py:345] 
(APIServer pid=14678) INFO 08-22 20:24:59 [api_utils.py:345]        █     █     █▄   ▄█
(APIServer pid=14678) INFO 08-22 20:24:59 [api_utils.py:345]  ▄▄ ▄█ █     █     █ ▀▄▀ █  version 0.27.1
(APIServer pid=14678) INFO 08-22 20:24:59 [api_utils.py:345]   █▄█▀ █     █     █     █  model   Qwen/Qwen2.5-0.5B-Instruct
...
(APIServer pid=14678) INFO 08-22 20:24:59 [api_utils.py:273] non-default args: {'model_tag': 'Qwen/Qwen2.5-0.5B-Instruct', 'model': 'Qwen/Qwen2.5-0.5B-Instruct', 'max_model_len': 2048, 'gpu_memory_utilization': 0.85}
...
(EngineCore pid=14790) INFO 08-22 20:25:12 [weight_utils.py:867] Filesystem type for checkpoints: EXT4. Checkpoint size: 0.92 GiB. Available RAM: 28.10 GiB.
...
(EngineCore pid=14790) INFO 08-22 20:25:12 [model_runner.py:329] Model loading took 0.93 GiB and 2.060686 seconds
...
(EngineCore pid=14790) INFO 08-22 20:25:13 [gpu_worker.py:563] Available KV cache memory: 19.06 GiB
(EngineCore pid=14790) INFO 08-22 20:25:13 [kv_cache_utils.py:2235] GPU KV cache size: 1,665,392 tokens
...
(EngineCore pid=14790) INFO 08-22 20:25:15 [model_runner.py:791] Graph capturing finished in 2 secs, took 0.28 GiB
(EngineCore pid=14790) INFO 08-22 20:25:15 [gpu_worker.py:789] Free memory on device (22.45/23.99 GiB) on startup. Desired GPU memory utilization is (0.85, 20.39 GiB). Actual usage is 1.25 GiB for consumed memory (weights + non-torch), 0.08 GiB for peak activation, and 0.28 GiB for CUDAGraph memory. Replace gpu_memory_utilization config with `--kv-cache-memory=20002824090` (18.63 GiB) to fit into requested memory, or `--kv-cache-memory=22214297088` (20.69 GiB) to fully utilize gpu memory. Current kv cache memory in use is 19.06 GiB.
...
(EngineCore pid=14790) INFO 08-22 20:25:17 [core.py:348] init engine (profile, create kv cache, warmup model) took 4.68 s (compilation: 0.11 s)
...
(APIServer pid=14678) INFO:     Started server process [14678]
...
(APIServer pid=14678) INFO:     Application startup complete.
...
(APIServer pid=14678) INFO:     127.0.0.1:41266 - "POST /v1/chat/completions HTTP/1.1" 200 OK
(APIServer pid=14678) INFO 08-22 20:25:44 [loggers.py:310] Engine 000: Avg prompt throughput: 4.4 tokens/s, Avg generation throughput: 7.6 tokens/s, Running: 0 reqs, Waiting: 0 reqs, GPU KV cache usage: 0.0%, Prefix cache hit rate: 0.0%
(APIServer pid=14678) INFO 08-22 20:25:54 [loggers.py:310] Engine 000: Avg prompt throughput: 0.0 tokens/s, Avg generation throughput: 0.0 tokens/s, Running: 0 reqs, Waiting: 0 reqs, GPU KV cache usage: 0.0%, Prefix cache hit rate: 0.0%
```

- key point: `Application startup complete.` — GPU KV cache **1,665,392 토큰**, max_model_len 2048 기준 최대 동시성 **813.18x**. 계획 문서의 "0.5B는 사실상 무제한" 추정과 부합한다.

---

## API 검증

### /v1/models

```shell
$ curl -s http://localhost:8000/v1/models | python3 -m json.tool
{
    "object": "list",
    "data": [
        {
            "id": "Qwen/Qwen2.5-0.5B-Instruct",
            "object": "model",
            "created": 1787397936,
            "owned_by": "vllm",
            "root": "Qwen/Qwen2.5-0.5B-Instruct",
            "parent": null,
            "max_model_len": 2048,
            "permission": [
                {
                    "id": "modelperm-ac498b0249add854",
                    "object": "model_permission",
                    "created": 1787397936,
                    "allow_create_engine": false,
                    "allow_sampling": true,
                    "allow_logprobs": true,
                    "allow_search_indices": false,
                    "allow_view": true,
                    "allow_fine_tuning": false,
                    "organization": "*",
                    "group": null,
                    "is_blocking": false
                }
            ]
        }
    ]
}
```

### /v1/chat/completions — 실추론

```shell
$ curl -s http://localhost:8000/v1/chat/completions -H "Content-Type: application/json" -d '{"model": "Qwen/Qwen2.5-0.5B-Instruct", "messages": [{"role": "user", "content": "KV cache가 뭔지 한 문장으로 설명해줘."}], "max_tokens": 100}' | python3 -m json.tool
{
    "id": "chatcmpl-90fc83686e04a254",
    "object": "chat.completion",
    "created": 1787397942,
    "model": "Qwen/Qwen2.5-0.5B-Instruct",
    "choices": [
        {
            "index": 0,
            "message": {
                "role": "assistant",
                "content": "Kv (Key-Value) cache\ub294 \ud074\ub77c\uc6b0\ub4dc \ucef4\ud4e8\ud305\uc5d0\uc11c \uc0ac\uc6a9\ub418\ub294 \ub370\uc774\ud130 \uc800\uc7a5 \uae30\ubc95\uc785\ub2c8\ub2e4. \uc774 \uae30\ubc95\uc740 \uc6d0\ud558\ub294 \ub370\uc774\ud130\ub97c \uc27d\uac8c \ucc3e\uc744 \uc218 \uc788\ub294 \ubc29\ubc95\uc744 \uc81c\uacf5\ud558\uba70, \uc5ec\ub7ec \uac1c\uc758 \uc11c\ubc84\uc640 \uc5f0\uacb0\ub418\uc5b4 \ub370\uc774\ud130\ub97c \uc800\uc7a5\ud558\uace0 \ucc98\ub9ac\ud569\ub2c8\ub2e4. \uc774\ub97c \ud1b5\ud574 \uc2e4\uc2dc\uac04 \ub370\uc774\ud130 \ucc98\ub9ac \ubc0f \ubd84\uc11d\uc774 \uac00\ub2a5\ud558\ub2e4\uace0 \ubcfc \uc218 \uc788\uc2b5\ub2c8\ub2e4.",
                "refusal": null,
                "annotations": null,
                "audio": null,
                "function_call": null,
                "reasoning": null
            },
            "logprobs": null,
            "finish_reason": "stop",
            "stop_reason": null,
            "token_ids": null,
            "routed_experts": null
        }
    ],
    "service_tier": null,
    "system_fingerprint": "vllm-0.27.1-106fe22d",
    "usage": {
        "prompt_tokens": 44,
        "total_tokens": 120,
        "completion_tokens": 76,
        "prompt_tokens_details": null
    },
    "prompt_logprobs": null,
    "prompt_token_ids": null,
    "prompt_text": null,
    "kv_transfer_params": null,
    "ec_transfer_params": null,
    "metrics": null
}
```

- key point: 76 completion 토큰 생성, `finish_reason: stop` 정상. (한글이 `\uXXXX`로 보이는 것은 `python3 -m json.tool`의 ensure_ascii 동작이다.) 내용을 디코드하면 — 0.5B 모델이 KV cache를 "클라우드 컴퓨팅의 데이터 저장 기법"이라고 답했다. **틀린 답이다.** 서빙 인프라 검증과 모델 품질은 별개라는 표본.

### 서빙 중 GPU 상태

```shell
$ nvidia-smi (vLLM 서버 가동 중)
Sat Aug 22 20:25:50 2026       
+-----------------------------------------------------------------------------------------+
| NVIDIA-SMI 590.57                 Driver Version: 591.86         CUDA Version: 13.1     |
+-----------------------------------------+------------------------+----------------------+
| GPU  Name                 Persistence-M | Bus-Id          Disp.A | Volatile Uncorr. ECC |
| Fan  Temp   Perf          Pwr:Usage/Cap |           Memory-Usage | GPU-Util  Compute M. |
|                                         |                        |               MIG M. |
|=========================================+========================+======================|
|   0  NVIDIA GeForce RTX 4090        On  |   00000000:01:00.0  On |                  Off |
|  0%   50C    P2             72W /  450W |   24019MiB /  24564MiB |      5%      Default |
|                                         |                        |                  N/A |
+-----------------------------------------+------------------------+----------------------+

+-----------------------------------------------------------------------------------------+
| Processes:                                                                              |
|  GPU   GI   CI              PID   Type   Process name                        GPU Memory |
|        ID   ID                                                               Usage      |
|=========================================================================================|
|    0   N/A  N/A              23      G   /Xwayland                             N/A      |
|    0   N/A  N/A           14790      C   /python3.12                           N/A      |
+-----------------------------------------------------------------------------------------+
```

- key point: VRAM 24,019MiB / 24,564MiB 사용 — `--gpu-memory-utilization 0.85`가 총 VRAM 기준으로 동작해 desktop 렌더링 사용분 위에 얹힌다. 계획 문서의 "0.95는 OOM 위험" 경고가 수치로 확인된다.

---

## 정리

1. **WSL2 + RTX 4090에서 vLLM 0.27.1 서빙이 동작한다.** Qwen2.5-0.5B-Instruct 기준 기동 성공, OpenAI 호환 API로 실추론 확인.
2. 기동 성공까지 **서버 시도 7회, 함정 6개** — ①UVA/pinned memory(WSL2 기본 비활성) ②gcc 부재 ③Python.h 부재 ④CUDA 툴킷 부재 ⑤nvcc(13.3)↔cudart 헤더(13.0) 버전 불일치 ⑥pip CUDA 레이아웃에 lib64/unversioned .so/stubs 없음. 전부 계획 문서에 없던 것들이다.
3. vLLM 0.27.1의 **V2 Model Runner는 pinned memory 필수**다. WSL2에서는 `VLLM_WSL2_ENABLE_PIN_MEMORY=1` opt-in 없이는 기동 자체가 안 된다 (커널 ≥ 4.19.121 필요, 이 머신은 5.15.90).
4. **flashinfer sampling 커널은 기동 시 JIT 컴파일된다** — gcc, Python.h, CUDA 툴킷(nvcc+헤더+링크 라이브러리)이 전부 런타임 요구사항이다. 시스템 CUDA 툴킷 없이 pip `nvidia-cuda-nvcc`(venv 내 `nvidia/cu13`)를 `CUDA_HOME`으로 지정해 해결했고, 버전 정렬(13.0.88)과 심링크 3개가 추가로 필요했다.
5. uv/pip의 의존성 해석이 **nvcc 계열과 cuda-runtime을 다른 마이너 버전으로 풀 수 있다**(13.3 vs 13.0). JIT가 없으면 조용히 지나가지만, JIT를 쓰는 순간 CCCL 헤더 가드에서 터진다.
6. 최종 영속 설정(`~/.bashrc`): uv PATH, npm prefix(`~/.npm-global`), `VLLM_WSL2_ENABLE_PIN_MEMORY=1`, `CUDA_HOME=$HOME/llmso/.venv/lib/python3.12/site-packages/nvidia/cu13`.

## Self-Check

- [x] WSL2에서 `nvidia-smi`로 4090 인식 (Step 0)
- [x] torch 2.13.0+cu130이 4090 / CC (8,9) 인식 (Step 5)
- [x] `vllm serve` 기동 + `/v1/models` 응답 (시도 7, API 검증)
- [x] chat completion 실추론 (API 검증)
- [x] 재부팅 후에도 유효하도록 환경변수 영속화 (Step 9, 10)
- [x] Docker (이 문서 작성 시점엔 미진행이었으나, 서빙 엔진 실습 Lab 6에서 Docker Desktop 대신 **docker.io를 WSL2에 직접 설치**로 해결)
- [ ] HF 토큰 로그인 (미진행 — 게이트 모델 필요 시)
- [ ] `.wslconfig` 메모리 상향 (미진행 — 현재 31GiB 할당, 7B 로딩 시 검토)

