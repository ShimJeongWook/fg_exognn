# FG_ExoGNN

CauAir 벤치마크(`Exo_Benchtool/CauAir`)에서 **FG_ExoGNN** 모델만 따로 분리해 단독 실행이 가능하도록 추출한 패키지입니다. 학습/평가에 필요한 코드, 데이터로더, 유틸, 그리고 3개 데이터셋만 포함합니다.

FG_ExoGNN(`ExoGNN`)은 외생변수(exogenous variable)를 명시적으로 분리해 처리하는 그래프 시계열 예측 모델입니다.
Patch + TimeXer 스타일 인코더와 Chebyshev 그래프 디코더를 결합해, 대기질(공기질) 다변량 시계열을 노드 그래프 위에서 예측합니다.

- 입력: `[B, T, N, F]` (batch, time, node, feature)
- 출력: `[B, H, N, O]` (batch, horizon, node, output_dim)

---

## 1. 디렉터리 구조

```
FG_ExoGNN/
├── experiments/
│   └── fg_exognn/
│       └── main.py              # 학습/평가 진입점
├── src/
│   ├── base/
│   │   ├── engine.py            # 학습 루프 베이스 (early-stop, ckpt, 로깅)
│   │   └── model.py             # BaseModel
│   ├── engines/
│   │   └── deepair_engine.py    # FG_ExoGNN이 쓰는 엔진 (AMP/grad clip 등)
│   ├── models/
│   │   └── fg_exognn.py         # ExoGNN 모델 본체
│   └── utils/
│       ├── args.py              # 공통 argparse 설정
│       ├── dataloader_deepair.py# 데이터 로딩 + get_dataset_info
│       ├── logging.py           # 로거
│       ├── metrics.py           # masked MAE/RMSE/MAPE 등
│       └── project.py           # 경로(루트/데이터/출력) 정의
├── data_airkorea/               # AirKorea 데이터셋 (306 노드, 12 변수)
│   ├── 24_24/all/{his.npz, idx_train/val/test.npy}
│   └── adj_mx.npy
├── data_knowair/                # KnowAir 데이터셋 (184 노드)
│   ├── 24_24/all/...
│   └── adj_mx.npy
├── datagagnn/                   # GAGNN 데이터셋 (209 노드)
│   ├── 24_24/all/...
│   └── adj_mx.npy
├── requirements.txt
└── README.md
```

> 실행 시 결과(로그, 체크포인트)는 `outputs/experiments/fg_exognn/<dataset>/` 아래에 생성됩니다.
> 호환을 위해 `experiments/fg_exognn/<dataset>` 경로로 심볼릭 링크도 자동 생성됩니다.

---

## 2. 설치

Python 3.10+ / CUDA GPU 환경 기준입니다.

```bash
pip install -r requirements.txt
```

FG_ExoGNN 실행에 **실제로 필요한 최소 패키지**는 다음과 같습니다(나머지는 원본 벤치마크에서 상속된 항목):

- `torch`
- `numpy`
- `tqdm`

```bash
pip install torch numpy tqdm
```

---

## 3. 데이터셋

`src/utils/dataloader_deepair.py`의 `get_dataset_info()`에 정의된 키로 데이터셋을 선택합니다.
이 패키지에 **번들된** 데이터셋은 다음 3개입니다.

| `--dataset` 키 | 데이터셋 | 노드 수 | 변수 수 | 경로 |
|---|---|---|---|---|
| `24_24_AK` | AirKorea | 306 | 12 | `data_airkorea/` |
| `24_24_KA` | KnowAir  | 184 | 13 | `data_knowair/` |
| `24_24_G`  | GAGNN    | 209 | -  | `datagagnn/` |

> 참고: 원본 CauAir의 기본 키 `24_24`(노드 1341)는 별도 대용량 데이터가 필요하며 **이 패키지에는 포함되어 있지 않습니다.**
> 사용하려면 해당 데이터를 `data/24_24/all/`, `data/adj_mx.npy`에 직접 배치하세요.
> (원본의 `KA_sub1/2/3` 키는 외부 절대경로를 참조하던 항목이라 추출 시 제거했습니다.)

각 데이터셋 폴더 구조:
```
<dataset_root>/
├── adj_mx.npy                       # 인접행렬 (N x N)
└── 24_24/
    └── all/
        ├── his.npz                  # 전체 시계열 텐서
        ├── idx_train.npy
        ├── idx_val.npy
        └── idx_test.npy
```

---

## 4. 실행

프로젝트 루트(`FG_ExoGNN/`)에서 실행합니다.

### 학습

```bash
python experiments/fg_exognn/main.py \
  --model_name fg_exognn \
  --dataset 24_24_AK \
  --device cuda:0 \
  --mode train
```

### 평가 (test / val)

저장된 체크포인트를 이용해 평가합니다.

```bash
python experiments/fg_exognn/main.py \
  --model_name fg_exognn \
  --dataset 24_24_AK \
  --device cuda:0 \
  --mode test
```

---

## 5. 주요 하이퍼파라미터

`experiments/fg_exognn/main.py` + `src/utils/args.py` 에서 정의됩니다.

**공통 (args.py)**

| 인자 | 기본값 | 설명 |
|---|---|---|
| `--dataset` | `24_24` | 데이터셋 키 (위 표 참조) |
| `--device` | `''` | 예: `cuda:0`, `cpu` |
| `--seed` | `2025` | 랜덤 시드 |
| `--bs` | `64` | 배치 크기 |
| `--seq_len` | `24` | 입력 시퀀스 길이 |
| `--horizon` | `24` | 예측 길이 |
| `--input_dim` | `8` | 입력 변수 수 |
| `--output_dim` | `1` | 출력(타깃) 변수 수 |
| `--max_epochs` | `100` | 최대 에폭 |
| `--patience` | `30` | early-stop patience |
| `--mode` | `train` | `train` / `test` / `val` |

**모델 전용 (main.py)**

| 인자 | 기본값 | 설명 |
|---|---|---|
| `--d_model` | `512` | 모델 차원 |
| `--d_ff` | `1024` | FFN 차원 |
| `--n_heads` | `8` | 어텐션 헤드 수 |
| `--e_layers` | `1` | 인코더 레이어 수 |
| `--dropout` | `0.1` | 드롭아웃 |
| `--patch_len` / `--patch_stride` | `12` / `12` | 패치 길이/스트라이드 |
| `--cheb_k` | `3` | Chebyshev 차수 |
| `--target_var_indices` | `None` | 타깃 변수 인덱스 (예: `"0"`) |
| `--exogenous_var_indices` | `None` | 외생 변수 인덱스 (예: `"1,2,3"`) |
| `--use_spatial` | `1` | 공간(그래프) 모듈 사용 |
| `--use_exo_attn` | `1` | 외생변수 어텐션 |
| `--use_cross` | `1` | cross-attention |
| `--use_future_exo` | `1` | 미래 외생변수 활용 |
| `--use_exo_head_context` | `1` | 외생 head context |
| `--use_future_step_context` | `1` | future-step context |
| `--restrict_static_graph` | `0` | 정적 그래프 제한 |
| `--use_amp` | `1` | 혼합정밀(AMP) 학습 |
| `--lrate` | `3e-4` | 학습률 |
| `--wdecay` | `1e-4` | weight decay |
| `--step_size` / `--gamma` | `10` / `0.95` | StepLR 스케줄러 |
| `--clip_grad_value` | `5` | gradient clipping |
| `--loss` | `mae` | `mae` 또는 `mae_rmse_mape` |

---

## 6. 평가 지표

`src/utils/metrics.py` — masked **MAE / RMSE / MAPE / R2 / IOA** 5개 지표를 horizon별 + 평균으로 출력합니다.
(R2 = 결정계수, IOA = Willmott index of agreement. 모두 다른 지표와 동일한 마스킹 규칙 적용.)
학습이 끝나면 test 평가에서 다음 형식으로 로그가 남습니다.

```
Horizon 1, Test MAE: ..., Test RMSE: ..., Test MAPE: ..., Test R2: ..., Test IOA: ...
...
Average Test MAE: ..., Test RMSE: ..., Test MAPE: ..., Test R2: ..., Test IOA: ...
```

학습/검증 로그와 horizon별 테스트 결과가 `outputs/experiments/fg_exognn/<dataset>/record_s<seed>.log`에 기록됩니다.

---

## 7. 추출 시 변경 사항 (원본 대비)

분리 과정에서 다음만 수정했고, 모델/엔진 로직은 원본과 동일합니다.

- `src/utils/project.py` : `AIRKOREA_ROOT = PROJECT_ROOT / "data_airkorea"` 추가.
- `src/utils/dataloader_deepair.py` : `get_dataset_info()`에 AirKorea용 `24_24_AK`(306노드) 키 추가, 외부 절대경로를 참조하던 `KA_sub1/2/3` 키 제거.
  - (원본에서는 AirKorea 데이터가 `get_dataset_info`에 등록돼 있지 않았습니다.)
- `src/utils/metrics.py` : `masked_r2`, `masked_ioa` 추가, `compute_all_metrics`가 `(MAE, MAPE, RMSE, R2, IOA)` 5개를 반환하도록 변경. F1-score 관련 코드(`masked_f1_score`)와 `torcheval` 의존성 제거.
- `src/engines/deepair_engine.py`, `src/base/engine.py` : test 평가 출력에 R2/IOA 추가, F1-score 출력 제거. deepair_engine에 인라인으로 박혀 있던 기존 R2/IOA·PM2.5-GNN-style(CSI/POD/FAR) 블록 제거.
