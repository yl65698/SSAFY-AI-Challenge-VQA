# 데이터 준비

대회에서 정식으로 제공받은 CSV와 이미지를 별도로 준비합니다. 이 공개 저장소에는 데이터셋과 가중치를 포함하지 않습니다.

## 데이터 구성

| 파일 | 행 수 | 내용 |
| --- | ---: | --- |
| `train.csv` | 5,073 | 이미지 경로, 질문, 선택지, 정답 |
| `dev.csv` | 4,413 | 이미지 경로, 질문, 선택지, 교육생 응답 5개 · 공식 정답 없음 |
| `test.csv` | 5,074 | 이미지 경로, 질문, 선택지 · 정답 없음 |

`train.csv`의 필수 컬럼은 `id,path,question,a,b,c,d,answer`이며 `test.csv`는 `answer`를 제외합니다. `answer`는 `a / b / c / d` 한 글자입니다. CSV의 `path`는 데이터 루트 아래 이미지 파일과 연결되어야 합니다.

```text
<DATA_ROOT>/
├── train.csv
├── dev.csv
├── test.csv
├── train/   # train.csv의 path와 일치하는 이미지
├── dev/
└── test/
```

실제 압축 해제 구조가 다르면 노트북 `CFG.data_root`와 이미지 경로를 함께 확인하세요. `CFG.data_zip_path`는 본인 Drive의 데이터 압축 파일, `CFG.output_root`는 체크포인트·실험 로그·submission 저장 경로로 설정합니다.

## 제출 형식

```csv
id,answer
test_0001,a
test_0002,b
```

위 두 행은 형식 설명용 예시입니다. 실제 제출은 원본 `test.csv`의 ID와 행 순서에 대응하는 예측으로 생성합니다.
