# LLMSO Study Notes

**Hands-On LLM Serving and Optimization Study (LLMSO)** 참여 중 작성한 학습 정리.

- 기간: 2026-08-02 ~ 2026-09-13 (매주 일요일 20:30~22:30)
- 교재: *Hands-On LLM Serving and Optimization* (2026.4)
- 운영: CloudNet@ / 서종호(gasida)

## 노트

**스터디 노트**는 개념 정리, **실습 기록**은 로컬 GPU(RTX 4090, WSL2)에서 직접 실행한 명령·출력·측정값이다.

| 날짜 | 주제 | 스터디 노트 | 실습 기록 |
|---|---|---|---|
| 2026-08-02 | 모델 서빙 입문, LLM 실행 원리 | [노트](model-serving-and-llm-basics.md) | [실습](labs/model-serving-and-llm-basics.md) |
| 2026-08-09 | 서빙 시스템 설계, 프로덕션 모범 사례 | [노트](serving-engine-and-production-architecture.md) | [실습](labs/serving-engine-and-production-architecture.md) |
| 2026-08-16 | 서빙 병목과 필수 최적화 기법 | [노트](serving-bottlenecks-and-optimization.md) | [실습](labs/serving-bottlenecks-and-optimization.md) |
| 2026-08-23 | 고급 최적화, 서빙 프레임워크 | [노트](advanced-optimization-and-serving-frameworks.md) | [실습](labs/advanced-optimization-and-serving-frameworks.md) |
| 2026-08-30 | 실제 적용, 새로운 방향 | 작성 예정 | — |
| 2026-09-06 | AWS Workshop: Generative AI on Amazon EKS | 작성 예정 | — |
| 2026-09-13 | llm-d + KServe | 작성 예정 | — |

실습 환경 구축 과정(WSL2 + RTX 4090, 함정 6개와 해결)은 [environment-setup](labs/environment-setup.md)에 별도로 기록했다.

## 참고 링크

- [llm-d 공식 문서](https://llm-d.ai/docs)
- [KServe 공식 문서](https://kserve.github.io/website/)
- AWS Workshop: Generative AI on Amazon EKS (진행 시 링크 추가)
