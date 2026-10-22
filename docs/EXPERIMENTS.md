# 실험 설계와 결과

## 자원 제약과 실험 순서

대회 기간은 6일이고 팀의 Colab GPU 사용량은 제한되어 있었습니다. 부분 데이터와 train holdout 검증으로 모델 후보와 학습 조건을 비교하고, 가치가 확인된 설정을 full training에 적용했습니다. 초기 baseline은 Qwen2.5-VL-3B, 최종 선택은 Qwen3-VL-32B 4bit 계열입니다.

팀장으로 모델 검증 코드, 실험 우선순위와 최종 모델 선정, GPU 사용·일정·제출을 관리했습니다. 팀은 모델 실험과 데이터 수집·전처리·증강으로 작업을 나눴습니다.

## 공개 노트북의 기본 설정

아래는 `final_training.ipynb`의 코드 기본값입니다. 당시 모든 실험 또는 최종 제출의 설정을 의미하지 않습니다.

| 설정 | 코드 기본값 |
| --- | --- |
| 모델 | `unsloth/Qwen3-VL-32B-Instruct-bnb-4bit` |
| 양자화 | 4bit |
| LoRA rank / alpha / dropout | 32 / 32 / 0.0 |
| Epoch | 2 |
| Learning rate | `1e-4` |
| Train batch / gradient accumulation | 1 / 8 |
| Seed | 42 |
| Holdout 비율 | 0.10 · 정답 기준 stratified split |
| 전체 데이터 학습 | `train_on_full_data=True` |
| 추론 모드 | `score` |
| 저장 주기 | optimizer step 50회 |
| 최근 체크포인트 유지 | 3개 |
| 학습 재개 | `resume_from="latest"` |
| 데이터 증강 | 기본 비활성화 |
| 반복 full training / voting | `DO_FULL_TRAIN_ROUNDS=False` |
| Grounding DINO crop 생성 | 선택 기능 · 기본 비활성화 |

모델 registry에는 32B·8B 변형이 함께 있으며, 설치 셀에 라이브러리 버전 조합이 기록되어 있습니다. 설치 셀의 주석을 해제하기 전에 해당 GPU·CUDA 환경과의 호환성을 확인해야 합니다.

## 체크포인트와 실행 안정성

Colab 연결이 끊겨 장시간 학습을 처음부터 다시 수행하는 비용을 줄이기 위해 모델·optimizer·scheduler·scaler, epoch·step/global step을 저장합니다. 최근 resume 체크포인트와 best 체크포인트를 관리하며, 종료 후 추론 결과를 `id,answer` CSV로 생성합니다.

## 앙상블

하이퍼파라미터를 다르게 설정해 얻은 세 추론 결과를 다수결로 결합했습니다. 더 복잡한 score/soft ensemble은 추가 추론 비용과 마감 시간을 고려해 최종 전략으로 선택하지 않았습니다.

체크포인트 저장·학습 재개는 실행 안정성을 위한 별도 기능입니다.

공개 코드에는 여러 submission 파일을 결합하는 함수와 반복 full training 후 voting하는 선택 경로가 포함되어 있습니다. 이 선택 경로는 기본 비활성화 상태이며, 실제 최종 제출에 사용한 세 설정을 그대로 재현하는 절차가 아닙니다. 세 설정의 정확한 값이나 조합은 공개 자료에서 확정하지 않습니다.

## 성과 기록과 해석

| 기록 | 범위 |
| --- | --- |
| Local validation 최고 0.9055 | 참가자 확인 기록 · holdout 실험 |
| Full-train 1epoch 제출 0.93811 | 참가자 확인 기록 · 대회 제출 점수 |
| 2epoch + LR `7e-5` | 참가자 기록에서 단일 모델 후보 중 가장 좋았던 조건 |
| 최종 결과 | Qwen3-VL-32B 계열 + Hard Voting · 192팀 중 7위 |

각 수치는 데이터·학습·평가 조건이 다르므로 동일 조건의 연속적인 개선율로 계산하지 않습니다. 공개본에는 원본 리더보드와 모든 실험 로그가 포함되어 있지 않습니다. 학습과 추론을 공개 준비 과정에서 다시 실행하지 않았습니다.

## Pseudo label 실험의 경계

개발 데이터의 다수 응답을 이용한 pseudo label 데이터가 실험 자료에 존재합니다. 다만 최종 7위 제출 모델이 해당 데이터를 실제 사용했다는 연결 근거는 부족하므로 최종 성과의 원인으로 단정하지 않습니다. 공개본은 원본·파생 데이터셋을 포함하지 않습니다.
