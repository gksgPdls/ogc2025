# OGC 2025 차량 운송선 적재 · 하역 최적화

[Optimization Grand Challenge (OGC)](https://optichallenge.com) 2025 공모전의 차량 운송선(RoRo) 적재 문제를 푸는 Python 알고리즘입니다.
선박 내부를 그래프로 보고, 여러 항구에서 차량을 싣고 내리는 순서와 위치를 정해 **고정비 + 이동 거리 비용**의 합을 줄입니다.

## 접근

| 단계 | 내용 |
|---|---|
| 그래프 분석 | 선박 칸을 노드, 통로를 간선으로 놓고 출입구에서의 거리 · 중심성으로 노드 우선순위를 매김 (`priority.py`) |
| 경로를 고려한 배치 | 먼저 내릴 차량을 출입구 가까이, 나중에 내릴 차량을 안쪽에 두어 다른 차를 옮기는 일을 줄임 |
| 클러스터 배치 | 같은 항구에서 내릴 차량을 묶어 한 구역에 배치. 수요량에 따라 클러스터를 나눔 |
| 미래 하역 고려 · 말단 노드 활용 | 다음 항구들의 하역 순서를 미리 보고, 막다른 칸을 늦게 내릴 차량에 씀 |
| 여러 버전 동시 실행 | 제한 시간 안에서 여러 휴리스틱을 돌려 비용이 가장 낮은 해를 고름 (`myalgorithm.py`) |

## 코드

| 파일 | 내용 |
|---|---|
| `src/myalgorithm.py` | 입구 `algorithm(prob_info, timelimit=60)`. 아래 버전들을 차례로 돌려 가장 좋은 해를 반환 |
| `src/vertical_v3_next_v1_multitry_v2.py` | 경로 인식 클러스터 배치 + 다중 실행 |
| `src/vertical_v5.py` | + 미래 하역 고려, 말단 노드 활용 |
| `src/vertical_v7.py` | 네트워크 구조에 맞춰 가중치를 조정하는 버전 |
| `src/vertical_v8.py` | + 수요량 기반 클러스터 분할 |
| `src/multi_cluster_v1.py` | 여러 클러스터 배치 실험 |
| `src/priority.py` | NetworkX 중심성 기반 노드 우선순위 |
| `src/util.py` | 해의 실행 가능성 검사 · 비용 계산 (대회 제공) |
| `baseline/` | 대회에서 준 베이스라인 코드 (비교용) |
| `examples/` | 대회에서 준 연습 문제 10개 (`prob1.json` ~ `prob10.json`) |

## 실행

```bash
pip install -r requirements.txt
cd src
python -c "import json, myalgorithm as m; p = json.load(open('../examples/prob1.json')); print(m.algorithm(p, timelimit=60) is not None)"
```

문제 정의 · 베이스라인 · 연습 문제 · `util.py` 는 [OGC 2025](https://optichallenge.com) 주최 측이 제공했고, `src/` 의 배치 알고리즘은 베이스라인을 바탕으로 직접 개발했습니다.
