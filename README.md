# Thai Stock Close Forecasting: KBANK & ADVANC

**Predicting two closing prices a day for two SET stocks, with locked selection rules, leakage-safe features and an honest answer to "does deep learning help?" (it didn't).**

![Course](https://img.shields.io/badge/CN340-Machine_Learning-B71C1C)
![Role](https://img.shields.io/badge/Role-Price_regression_lead-blue)
![Python](https://img.shields.io/badge/Python-3.11-3776AB?logo=python&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.9-F7931E?logo=scikitlearn&logoColor=white)

> **Context:** CN340 Machine Learning, Thammasat University (team of 4, Aug – Nov 2026). This repo is the **price-regression task (Task 3)**, which I led. It is a public snapshot of the code and reports. Raw market data and fitted models are not included (see [Data](#data)).

---

## The problem

Every trading day, submit **4 predictions**: the morning close and the official close of KBANK and ADVANC.

| Round | Predict | Deadline | Baseline to beat |
|---|---|---|---|
| Morning | Last price of the morning session | before 09:30 | yesterday's close (dividend-adjusted) |
| Afternoon | Official close (closing auction 16:35–16:40) | before 13:00 | the morning close |

These baselines are hard to beat: most days the price moves only 1–3 ticks. The real question is whether a model can add **any** reliable signal over "the price won't move". It also has to do that using only data available before each deadline.

## Approach

1. **Lock the rules first.** The protocol ([`02_PROTOCOL.md`](src/task3/versions9/docs/02_PROTOCOL.md)) fixed the selection period (2019–2022), the confirmation period (2023 → Sep 2026) and the pass criteria **before** any model saw the confirmation data. The chosen models were tagged `v9-prereg` and never changed afterwards.
2. **Leakage-safe features.** Every row is cut at the deadline with one shared rule. Features cover overnight foreign markets (S&P 500, Nikkei, Hang Seng, USD/THB), the previous day's closing-auction bounce, volume, and the stock's own morning move ([`03_FEATURES.md`](src/task3/versions9/docs/03_FEATURES.md)).
3. **Walk-forward evaluation** from 2017, scored as *skill* = % reduction in MSE against the no-change baseline, with paired standard errors.
4. **Sanity checks as code** ([`checks/`](src/task3/versions9/checks/)): data integrity, auction calendar, a positive control that injects a fake 2% signal to confirm the pipeline can find one, and an overnight-effect test.

## Results

Confirmation period 2023-01 → 2026-09, **~890 trading days never used for selection**:

| Round | Model | Skill vs baseline | 95% CI | RMSE KBANK (baseline → model) | RMSE ADVANC |
|---|---|---|---|---|---|
| Morning | Huber regression, per stock | **+6.43%** | +3.5 → +9.5 | 1.755 → 1.696 THB | 3.102 → 3.003 THB |
| Afternoon | HistGradientBoosting, pooled | **+4.12%** | +2.2 → +6.1 | 1.003 → 0.981 THB | 1.906 → 1.869 THB |

**What this means in practice:** the gain is statistically real but small, about 2–10 satang per day, which is less than one price tick. A trader should read it as a slightly better anchor, not as an edge you can trade on by itself.

### Did bigger models help? ([`versions9_1`](src/task3/versions9_1/docs/INTERIM_REPORT.md))

After locking v9, I benchmarked **~40 alternatives** on exactly the same splits:

| Family | Models tried | Best result vs v9 |
|---|---|---|
| Tabular (14) | Lasso, SVR, RF, XGBoost, LightGBM, CatBoost, TabPFN, … | within ±1 SE, no clear winner |
| Sequence (13) | ARIMAX, LSTM, GRU, TCN, Transformer, PatchTST, N-HiTS, TFT, … | worse or tied |
| Foundation (4) | Chronos-Bolt, TimesFM 2.5, Moirai 2.0, Kronos | ≈ 0% skill, worse by 2–8% |
| CNN (9) | 1D-CNN, CNN-LSTM, WaveNet, TimesNet, chart images, spectrograms | worse or tied |

**Takeaway:** with ~480 rows per stock per year, the simple tabular models won. Bigger models were the worst in their families. The morning round needs the overnight foreign-market data, and models that only saw the stock's own price history scored about zero.

## Deployment

[`src/task3/deployment/v9/`](src/task3/deployment/v9/) turns the locked model into a one-command daily predictor:

```bash
cd src/task3/deployment/v9
pip install -r requirements.txt                      # Python 3.11, pinned versions
python fetch_yahoo.py --out INPUT_DIR                # free daily / 1-minute / foreign-market data
python live.py --inputs INPUT_DIR --round morning --date 2026-11-10 --log predictions.jsonl
```

- `live.py` recomputes features using a **byte-identical copy** of the training feature code, which `--selftest` verifies. It rounds the forecast to a valid SET tick and logs it with a timestamp as proof it was made before the cutoff.
- If a feature is missing, `predict.py` falls back to a smaller backup model instead of failing.
- `replay_check.py` re-ran the live path over 133 past days (532 runs). The features matched training to within 4e-17.

## Repository map

```
src/task3/versions9/     locked research code (features, model ladder, selection, confirmation) + docs/
src/task3/versions9_1/   post-lock benchmark of ~40 alternative models
src/task3/deployment/v9/ live predictor, refit and replay tools, reports
```

Most docs are in Thai. The one-page summary is [`FINAL_REPORT_UNOFFICIAL.md`](src/task3/versions9/docs/FINAL_REPORT_UNOFFICIAL.md).

## Data

The price data came from licensed sources (SETSMART, a broker MT5 export) that can't be redistributed, so **no raw data or fitted model is included**. `fetch_yahoo.py` rebuilds the free part of the inputs. `build_data.py` and `fit.py` show exactly how the training set and models are built.

## Credits

Task 3 (price regression) by **Athiphat Sunsit**. The team of 4 also covered Task 1 (even/odd classification) and Task 2 (up/down classification), which are not part of this snapshot. Coursework for CN340, Department of Computer Engineering, Thammasat University.
