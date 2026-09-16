# Signal Validation 학습 가이드

> 목적: 이 프로젝트를 내가 직접 설계·구현한 것으로서 인터뷰에서 막힘없이 설명할 수 있게 되기

---

## 1. 한 문장 정의

**Signal validation = "이 트레이딩 모델을 믿고 돈을 태워도 되는가?"를 모델을 만든 사람이 아닌 제3자가 검증하는 일**

- 은행에서는 이걸 **MRM (Model Risk Management)** 이라고 부름
- 미국 감독 지침 **SR 11-7 / OCC 2011-12** 가 표준
- 핵심 원칙: 모델을 만든 사람(1선, developer)과 검증하는 사람(2선, validator)은 달라야 하고, validator는 모델을 **고치지 않는다**. 승인 / 조건부 승인 / 불승인만 한다.
- 이 레포는 내가 예전에 만든 모델 3개(GBM, CLAM, momentum)를 "남의 모델 보듯이" 검증하는 구조
- `config.py` 주석: *"this repo is the judge, not the modeller"*

---

## 2. 왜 필요한가 — 백테스트가 거짓말하는 3가지 방식

| 문제 | 실제 사례 | 이 레포에서의 질문 | 모듈 |
|---|---|---|---|
| 과적합 — 수십 개 시도 중 제일 잘 나온 걸 보고서에 올림 | 파라미터 100개 돌려보면 하나는 Sharpe 2 나옴 | **Q1** 백테스트 성과가 진짜인가? | `sv/validation/overfit.py` |
| 드리프트 — 만들 땐 맞았는데 지금은 시장이 바뀜 | 2020 이전 모델이 2022 금리 상승기에 죽음 | **Q2** 지금도 유효한가? 언제 깨졌는지 어떻게 아나? | `sv/validation/monitoring.py` |
| 꺼야 할 때 못 끔 — 손실 나는데 규칙이 없어서 계속 돌림 | 드로다운 -40%까지 방치 | **Q3** 언제 꺼야 하고, 그 규칙이 실제로 도움이 되나? | `sv/validation/gate.py` |

---

## 3. 핵심 결과 (외워야 할 숫자)

| 모델 | Q1 과적합 테스트 | Q2 모니터링 (최신) | Q3 게이트 효과 (OOT) | 결정 |
|---|---|---|---|---|
| `momentum` | 2/3 통과 (DSR 0.68 실패) | AUC 0.506, PSI 0.25 (watch) | max DD −29.7% → −22.0%, CAGR 35.5% → 23.7% | 벤치마크로만 조건부 승인 |
| `gbm_expected` | 0/3 | AUC 0.500, calibration slope −0.03 | 63% 주간 off | 불승인 — 스킬 증거 없음 |
| `gbm` (배포) | 0/3 | AUC 0.502 | 92% 주간 off | 불승인 — 구현 결함 (F1) |
| `clam_orig` | — | — | — | 불승인 — 검증 불가 (F3, F4) |
| `clam_2021` | — | — | — | 보류 — 재학습 twin 미채점 |

- 후보 5개 중 승인 0개. untuned momentum이 "미래 모델이 넘어야 할 벤치마크"로 유지됨
- 가장 가치 있는 산출물은 통계가 아니라 **코드를 읽고 재현하다 발견한 Finding F1~F5**
- Full-sample backtest (2015-01 ~ 2026-09): momentum CAGR 27.0% / Sharpe 0.88 / max DD −39.9% / 회전율 15%, gbm 회전율 **90%**
- DSR: 후보 3개뿐이어도 노이즈 전략의 기대 최대 Sharpe가 0.51 → momentum 0.65가 95% 신뢰로 못 넘김
- Permutation test는 **gross** 수익으로 평가 (net이면 셔플 신호 회전율 ~100% → 비용 편향으로 null Sharpe ≈ −1.9)
- F5: 1일 실행 지연이 momentum CAGR을 절반으로

---

## 4. 파일 읽기 순서 (전체 파일, 파이프라인 순서)

### 0단계 — 뼈대 (먼저 훑고 마지막에 다시 읽기)

| # | 파일 | 역할 | 인터뷰에서 답해야 할 것 |
|---|---|---|---|
| 1 | `README.md` | 프로젝트 목적, Q1/Q2/Q3 ↔ 모듈 매핑, 후보 모델 5개, 재현 명령 | 왜 validation이지 development가 아닌가 / 왜 untuned 벤치마크가 필요한가 |
| 2 | `config.py` | 모든 모델링 가정이 한 곳에 (유니버스 3000, 비용 10bps, top-50, 12-1 momentum, DSR N=3, bootstrap block 13주, gate 임계값 4개) | 숫자 하나하나 "왜 이 값인가". 특히 `COST_BPS=10`, `BOOTSTRAP_BLOCK=13`, `GATE_RULES`, `OOT_START=2022-01-03` |

### 1단계 — 데이터 (DB 레이어)

| # | 파일 | 역할 | 인터뷰 포인트 |
|---|---|---|---|
| 3 | `sql/schema.sql` | DuckDB 테이블: universe → prices → panel → signals → portfolio_returns → monitoring / gate_decisions / validation_results | 테이블 흐름을 화이트보드에 그릴 수 있어야. 각 테이블 PK |
| 4 | `sv/db.py` | DuckDB 얇은 래퍼 (`connect`, `run_sql_file`, `read`, `write_df`) | 왜 DuckDB인가 (파일 하나, SQL로 리뷰 가능, window function 빠름) |
| 5 | `sv/universe.py` | Yahoo 스크리너로 시총 상위 3000 스냅샷 | 생존편향 — 왜 생기고, 어떻게 완화했나(유니버스 대비 초과수익), 왜 완전 해결은 안 되나 |
| 6 | `sv/ingest.py` | 일별 OHLCV 2013~ 다운로드, upsert | 왜 2013부터인가 (백테스트 2015 시작 + 2년 워밍업). adj_close vs close |
| 7 | `sql/queries/build_panel.sql` | 주간(금요일) 리밸런스 패널: 유동성 필터, 다음 주 수익률 | window function 직접 설명. 유동성 필터(가격≥$5, ADV≥$5M)가 알파가 아니라 tradability인 이유 |
| 8 | `sv/features.py` | 위 SQL 실행 + 후보 모델용 wide 가격 행렬 | 얇은 파일 |
| 9 | `sql/queries/universe_coverage.sql` | 연도별 가격 있는 종목 수 → fig1 | 생존편향을 수치로 증명 (2015년 2,006개 → 2026년 3,001개) |
| 10 | `notebooks/00_data_quality.ipynb` | workpaper: 커버리지, 음수 adj_close 3종목(CBIO, NFE, DEC), 부분 주 | 데이터에서 뭘 발견했고 어떻게 처리했나 |

### 2단계 — 검증 대상 모델 (후보)

| # | 파일 | 역할 | 인터뷰 포인트 |
|---|---|---|---|
| 11 | `sv/candidates/momentum.py` | 12-1 momentum (`P[t−21]/P[t−252] − 1`), 16줄 | 파라미터를 데이터 보고 정하지 않았다는 게 왜 중요한가 |
| 12 | `sv/candidates/gbm.py` | 배포된 GBM 시뮬레이션 verbatim 포팅 + `gbm_expected` 파생 | **F1 핵심**: 단일 MC path는 노이즈, `E[score] = mean_i exp(μ·t_i) − 1` 은 μ만의 함수 → 결국 trailing mean return 순위. 수식 유도 가능해야 |
| 13 | `sv/candidates/clam.py` | CNN-LSTM-Attention 추론, 배포 코드 그대로 | 왜 원본 모델을 재학습하면 안 되는가(검증 대상이니까). 왜 `clam_orig`는 검증 불가인가(2025-08까지 학습 → OOT 없음) |
| 14 | `scripts/run_candidates.py` | momentum + gbm 점수를 `signals` 테이블에 기록 | 얇음 |
| 15 | `scripts/retrain_clam.py` | 원본 학습 코드로 2021-12-31 컷 → `clam_2021` | "새 모델이 아니라 같은 방법론을 이전 정보집합으로 재실행한 것" |
| 16 | `scripts/gpu_env.sh` | TF GPU 환경 | 실행용, 인터뷰 무관 |
| 17 | (형제 레포) `Long_Term_Trading/stats_model_process.py`, `clam_inference.py` | 원본 배포 코드 | validator가 원본을 읽고 재현했다는 근거. F1이 원본 어디서 나오는지 |

### 3단계 — 공통 백테스트 엔진

| # | 파일 | 역할 | 인터뷰 포인트 |
|---|---|---|---|
| 18 | `sql/queries/realised_vol.sql` | 252일 실현 변동성 (모니터링 input 변수) | 왜 vol을 regime variable로 골랐나 |
| 19 | `sv/backtest.py` | 주간 top-50 동일가중, 비용, `_lag1` 변형, 유니버스/SPY 벤치마크 | 회전율 기반 비용 계산. F5 (1일 지연 → CAGR 절반) 왜? 모든 후보에 같은 엔진을 쓰는 이유 |

### 4단계 — Q1 과적합

| # | 파일 | 역할 | 인터뷰 포인트 |
|---|---|---|---|
| 20 | `sv/validation/overfit.py` | stationary block bootstrap (5,000회, block 13주), DSR (N=3), cross-sectional permutation (500회) | 세 테스트가 각각 어떤 오류를 막는지. DSR 수식 (expected max SR of N noise trials, skew/kurtosis 보정). 왜 permutation을 gross로 바꿨는지 — 리뷰 중 스스로 고친 스토리 |
| 21 | `notebooks/01_overfit.ipynb` | 결과: momentum 2/3 (DSR 0.68 실패), gbm 0/3 | "N=3인데도 노이즈 최대 Sharpe 기대가 0.51" 해석 |

| 테스트 | 막는 것 | 통계량 | 통과 기준 |
|---|---|---|---|
| Block bootstrap Sharpe CI | Sharpe 점추정치 하나만 믿는 것 | 연율화 active Sharpe 95% CI | CI가 0 제외 |
| Deflated Sharpe Ratio | 여러 시도 중 최고만 고르는 것, 비정규 수익 | P(true SR > 노이즈 N개 중 기대 최대 SR) | ≥ 0.95 |
| Permutation null | 신호–수익 관계가 우연인 것 | 주별 점수 셔플 후 Sharpe ≥ 관측치 비율 (gross) | p < 0.05 |

### 5단계 — Q2 모니터링

| # | 파일 | 역할 | 인터뷰 포인트 |
|---|---|---|---|
| 22 | `sql/queries/psi_score.sql` | PSI를 SQL로 (개발표본 첫 104주, 10분위) | PSI 공식 `Σ (actual − expected) · ln(actual / expected)`, 0.1 / 0.25 관행 |
| 23 | `sql/queries/auc_by_year.sql` | RANK 기반 AUC | AUC = Mann-Whitney U 원리. SQL로 짤 수 있다는 것 |
| 24 | `sv/validation/monitoring.py` | 52주 롤링 PSI(score/input)/KS/AUC(1w, 13w)/calibration slope/rolling Sharpe/active DD | look-ahead 방지: 13주 horizon 지표를 13주 lag한 이유. 각 지표가 score 드리프트 / input 드리프트 / 판별력 / 보정 중 무엇을 보는지 |
| 25 | `notebooks/02_monitoring.ipynb` | fig2, 최신 판독 | "AUC 0.5 근처인데 왜 수익이 나나" 대비 |

### 6단계 — Q3 킬스위치

| # | 파일 | 역할 | 인터뷰 포인트 |
|---|---|---|---|
| 26 | `sv/validation/gate.py` | 규칙 4개 OR → 다음 주 현금, 포트폴리오 지표 1주 lag | 임계값을 OOT 보기 전에 고정한 이유. 게이트 자체가 과적합되는 경로 |
| 27 | `sql/queries/champion_challenger.sql` | 게이트 O/X 성과 비교 (2022-01-03 ~, 246주) | champion(ungated) / challenger(gated) 용어 |
| 28 | `notebooks/03_gate.ipynb` | fig3, DD −29.7 → −22.0 vs CAGR 35.5 → 23.7 | "게이트가 Sharpe를 낮췄는데 왜 유의미한가" — trade-off로 정직하게 |

| 규칙 | 임계값 | 근거 |
|---|---|---|
| `psi_score` | > 0.25 | "모집단이 바뀜"의 업계 관행 |
| `rolling_sharpe` | < 0 | 최근 1년 active 수익 음수 |
| `auc_1w` | < 0.50 | 최근 1년 판별력 없음 |
| `active_drawdown` | < −15% | 상대 드로다운 허용 초과 |

### 7단계 — 결론

| # | 파일 | 역할 | 인터뷰 포인트 |
|---|---|---|---|
| 29 | `notebooks/04_findings.ipynb` | F1~F5 정리 | 다섯 finding을 외워서 |
| 30 | `reports/validation_report.md` | SR 11-7 구조: 목적 → 데이터 → 방법론 → 결과 → Findings → 한계 → 모니터링 계획 → 사용 조건 → 결정 | 섹션 순서 자체, "5개 중 승인 0" 결론을 자신 있게 |
| 31 | `reports/validation_report.{html,pdf}`, `reports/figures/fig1~3.png` | 산출물 | 그림 3개 각각 한 문장 |
| 32 | `requirements.txt`, `.gitignore`, `sv/__init__.py`, `sv/candidates/__init__.py`, `sv/validation/__init__.py`, `sv/report/__init__.py` | 설정 / 빈 파일 | 읽을 것 없음. `sv/report/`는 리포트 자동화 자리 (미구현) |

- `data/*.log`, `data/raw/screener_raw.json`, `data/signal_validation.duckdb` 는 실행 산출물 → 읽을 필요 없음. DB는 DuckDB CLI로 직접 쿼리해보면 SQL 연습에 좋음

---

## 5. 배경 자료

- SR 11-7 / OCC 2011-12 원문 (~20페이지)
- Bailey & López de Prado (2014), *The Deflated Sharpe Ratio*
- López de Prado, *Advances in Financial Machine Learning* 11~14장 (백테스트 과적합)
- Siddiqi, *Credit Risk Scorecards* (PSI / KS / AUC 모니터링)
- Politis & Romano (1994), *The Stationary Bootstrap*

---

## 6. 학습 방식

- 단계별로 **파일 읽기 → 내가 설명해보기 → 막히는 부분 질문**
- 총 코드 ~1,100줄
- 시간을 제일 많이 쓸 곳: `sv/validation/overfit.py` (DSR), `sv/candidates/gbm.py` (F1 유도)

## 7. 진행 체크리스트

- [ ] 0단계 README / config
- [ ] 1단계 데이터
- [ ] 2단계 후보 모델
- [ ] 3단계 백테스트
- [ ] 4단계 Q1 과적합
- [ ] 5단계 Q2 모니터링
- [ ] 6단계 Q3 게이트
- [ ] 7단계 결론 / Findings F1~F5 암기
