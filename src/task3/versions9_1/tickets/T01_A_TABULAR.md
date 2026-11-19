# T01 — กลุ่ม A: โมเดลข้อมูลตาราง (A01–A14)

- Status: ร่าง 2026-10-29 · ขึ้นกับ [T00](T00_COMMON.md) · โค้ด: `versions9_1/models_tab.py`
- คำถามที่ตอบ: โมเดลที่ review นับว่านิยมที่สุด (SVM, RF, ANN, boosting) และชุดของ Gu-Kelly-Xiu ชนะ Huber/HGB ของ v9 ไหม
  เมื่อได้ feature, หน้าต่าง และงบจูนเท่ากัน
- ที่คาดไว้ (เขียนก่อนรัน): ต้นไม้และ boosting ≈ HGB · SVR/kNN แพ้ ridge · NN3/NN5 แพ้ HGB (Grinsztajn 2022) · TabPFN ไม่รู้
  เพราะยังไม่มีงานไหนลองกับหุ้น · ไม่มีตัวไหนชนะ R3 เกิน 1 SE ในรอบเช้า

## 1. ข้อมูลเข้าและการเตรียม (ตายตัวทุกตัวในกลุ่ม)

| เรื่อง | ค่า |
|---|---|
| คอลัมน์ | A01–A06 **รายหุ้น** ใช้ `cols(rnd, s, "own")` (= ridge v9) · A07–A14 **รวมสองหุ้น** ใช้ `cols(rnd, None, "pooled")` (= HGB v9 + `is_kbank`) |
| เหตุผลที่แบ่ง | ทำตาม v9: linear อยู่รายหุ้น ต้นไม้/NN รวมสองหุ้น · เทียบกับ R2/R3 ได้ตรง |
| clip | 1%/99% ของหน้าต่างเทรน ทุกตัว |
| ปรับสเกล X | ค่าเฉลี่ย/SD ของหน้าต่างเทรน: A01–A06, A12–A13 · ไม่ปรับ: A07–A11 (ต้นไม้), A14 (TabPFN ปรับเอง) |
| ปรับสเกล y | fit บน `y / sd(y)` แล้วคูณกลับ: A01–A06, A12–A14 · ต้นไม้ใช้ `y` ดิบ (ไม่ขึ้นกับสเกล) |
| แถวที่หาย | ไม่ใช้ (ไม่เติม) |
| fit ใหม่ | **ทุกสัปดาห์** ยกเว้น A12–A13 **รายเดือน** (โปรโตคอล v9 อนุญาตให้ NN เป็นรายเดือน) |
| เลือก hyperparameter | `ladder.select` ปีละครั้ง · grid ไม่เกิน 3 ค่า |

## 2. รายตัว

| ID | โมเดล | ค่าตายตัว | grid (เลือกปีละครั้ง) | หมายเหตุ |
|---|---|---|---|---|
| A01 | `sklearn.linear_model.Lasso` | `max_iter=10000`, `tol=1e-6`, `selection="cyclic"` | α ∈ {0.01, 0.03, 0.1} | loss ของ Lasso หารด้วย n (ต่างจาก Ridge) · α ที่ทำให้ coefficient เป็น 0 หมดคือ `max|Xᵀy|/n` ≈ ค่า corr สูงสุด ≈ 0.1–0.4 · grid ครอบ "เกือบไม่ shrink" ถึง "ตัดเกือบหมด" |
| A02 | `ElasticNet` | `l1_ratio=0.5` (GKX ตั้งไว้ 0.5), `max_iter=10000` | α ∈ {0.01, 0.03, 0.1} | |
| A03 | `cross_decomposition.PLSRegression` | `scale=False` (ปรับสเกลแล้ว) | component ∈ {1, 2, 3} | GKX |
| A04 | PCR = `PCA` → OLS | `svd_solver="full"` | component ∈ {1, 3, 5} | GKX · PCA fit บนหน้าต่างเทรนเท่านั้น |
| A05 | `svm.SVR` | `kernel="rbf"`, `gamma="scale"`, `epsilon=0.1` (หน่วย SD ของ y) , `cache_size=500` | C ∈ {0.1, 1, 10} | กลุ่มที่ review นับว่านิยมที่สุด · ~2,000 แถวต่อหุ้น เร็วพอ |
| A06 | `neighbors.KNeighborsRegressor` | `weights="uniform"`, ระยะ Euclid บน X ที่ปรับสเกล | k ∈ {20, 50, 100} | ตัวเทียบไม่มีพารามิเตอร์ |
| A07 | `ensemble.RandomForestRegressor` | `n_estimators=500`, `max_features=0.33`, `bootstrap=True`, `random_state=340`, `n_jobs=1` | `min_samples_leaf` ∈ {50, 100, 200} | ใบ ≥ 50 แถวเท่ากับ HGB v9 · GKX พบต้นไม้ที่ดีตื้นมาก |
| A08 | `ensemble.ExtraTreesRegressor` | เหมือน A07 (`bootstrap=False`) | `min_samples_leaf` ∈ {50, 100, 200} | ดีที่สุดรองจาก ridge ใน v5 |
| A09 | `xgboost.XGBRegressor` | `learning_rate=0.05`, `n_estimators≤500`, `min_child_weight=50`, `reg_lambda=1`, `subsample=0.8`, `colsample_bytree=0.8`, `tree_method="hist"`, `nthread=1`, `random_state=340` | `max_depth` ∈ {2, 3, 4} | จำนวนรอบ: early stop แบบ v9 HGB (patience 20 บนเส้น validation ของ 3 ช่วง → รอบเฉลี่ย) · ดึงเส้น validation ด้วย `iteration_range` |
| A10 | `lightgbm.LGBMRegressor` | `learning_rate=0.05`, `n_estimators≤500`, `min_child_samples=50`, `reg_lambda=1`, `subsample=0.8`, `subsample_freq=1`, `colsample_bytree=0.8`, `num_threads=1`, `deterministic=True`, `random_state=340`, `verbose=-1` | `num_leaves` ∈ {4, 8, 16} | รอบเหมือน A09 · เส้น validation ด้วย `predict(num_iteration=k)` |
| A11 | `catboost.CatBoostRegressor` | `learning_rate=0.05`, `iterations≤500`, `l2_leaf_reg=3`, `thread_count=1`, `random_seed=340`, `verbose=0` | `depth` ∈ {2, 3, 4} | รอบเหมือน A09 · เส้น validation ด้วย `staged_predict` |
| A12 | NN3 (torch) | 32-16-8, ReLU, BatchNorm หลังทุกชั้นซ่อน, Adam lr 10⁻³, batch 64, ≤ 200 epoch, patience 20, init Kaiming, 5 seed (340–344) เฉลี่ย | weight decay ∈ {10⁻⁴, 10⁻³, 10⁻²} | GKX ตัวที่ดีที่สุด · GKX ใช้ BN + early stop + เฉลี่ยหลายรอบ |
| A13 | NN5 (torch) | 32-16-8-4-2 นอกนั้นเหมือน A12 | เหมือน A12 | ดูว่าลึกขึ้นแย่ลงไหม (GKX: NN5 < NN3) |
| A14 | TabPFN v2 (`tabpfn.TabPFNRegressor.create_default_for_version(ModelVersion.V2)`) | `n_estimators=8` (ค่าเริ่มต้น), `random_state=340`, ค่าทาย = ค่าเฉลี่ย (`output_type="mean"`), device `mps` ถ้าใช้ได้ ไม่งั้น `cpu` | ไม่มี | ไม่ต้องเทรน: ทุกสัปดาห์ใส่หน้าต่างเทรนเป็น context ใหม่ |

### 2.1 A14 TabPFN — เรื่องที่ต้องตัดสินก่อนรัน

- **license/weight:** ใช้ weight **v2** (Prior Labs License = Apache 2.0 + ให้เครดิต) · ไม่ใช้ v2.5+ (ใช้เพื่อการค้าไม่ได้ และต้องมี token)
  · ถ้า v2 ต้อง login หรือขอ token เมื่อไหร่ ให้หยุดแล้วถามเจ้าของ
- **ขนาด:** เอกสาร TabPFN บอกว่า CPU ใช้ได้กับข้อมูล ≲ 1,000 แถว · หน้าต่างรวมสองหุ้นของเรายาวถึงราว 4,000 แถว
- **กฎสำรองที่ตั้งก่อนเห็นผล (วัดแค่เวลา):** จับเวลา 1 สัปดาห์ (fit + ทาย) ที่ปลายหน้าต่าง (2026-08) บน MPS
  · ≤ 60 วินาที → ใช้หน้าต่างเต็ม · > 60 วินาที → ใช้ **1,000 แถวล่าสุด** ของหน้าต่างเทรนทุกสัปดาห์ · จดค่าที่ใช้ลง manifest
- MPS ไม่ทำซ้ำได้ทุกบิต → รัน A14 สองครั้งบน 4 สัปดาห์แรก รายงานส่วนต่างสูงสุด · ยอมรับถ้า < 10⁻⁶ ของ SD ของ y

## 3. test ของกลุ่มนี้ (เพิ่มใน `tests/test_task3_v9_1.py`)

1. `train("A01", ...)` ที่ α เล็กมากได้ค่าทายใกล้ OLS ของ v9 (`atol` 1e-6 บน y/sd(y))
2. A04 ที่ component = จำนวน feature ได้ค่าทายเท่า OLS
3. A09/A10/A11: เส้น validation ที่ดึงทีละรอบ ที่รอบสุดท้ายเท่ากับ `predict` ของโมเดลเต็ม
4. A12: สองครั้งได้ค่าเท่ากันทุกบิต (CPU, deterministic)
5. ทุกตัว: `predict` ใช้ clip/สเกลของหน้าต่างเทรน ไม่ใช่ของแถวที่ทาย (ใส่แถวทายค่าสุดโต่ง 1 แถว แล้วค่าทายแถวอื่นไม่เปลี่ยน)

## 4. positive control

- ใส่ `x` (T00 §7) เป็นคอลัมน์ที่ 1 ของ X ทุกตัว · A14 ด้วย · ให้คะแนน 2021–2022 · ผ่าน = skill เพิ่ม ≥ 1%
- ที่คาด: linear ทุกตัวผ่าน · ต้นไม้ผ่านเพราะ `x` เป็นเส้นตรงแต่ใช้ใบหยาบ อาจได้แค่บางส่วน · kNN อาจไม่ผ่าน (`x` เป็น 1 ใน 20+ มิติ)

## 5. การรันและเวลา

```
OMP_NUM_THREADS=1 .venv/bin/python -W ignore -m src.task3.versions9_1.zoo morning --group A
OMP_NUM_THREADS=1 .venv/bin/python -W ignore -m src.task3.versions9_1.zoo afternoon --group A
```

| ตัว | ประมาณต่อรอบทาย (2019-01 → 2026-09) |
|---|---|
| A01–A06 | 1–10 นาที (SVR ช้าสุด) |
| A07–A08 | 10–30 นาที (500 ต้น × ~400 สัปดาห์ · `n_jobs=1`) |
| A09–A11 | 10–20 นาที |
| A12–A13 | 20–40 นาที |
| A14 | วัดตาม §2.1 ก่อน · ถ้าเกิน 2 ชั่วโมงแจ้งเจ้าของ |

- ต้องติดตั้ง: `tabpfn` (ขออนุญาต) · ตัวอื่นมีใน `.venv` แล้ว

## 6. เสร็จเมื่อ

- [ ] test §3 ผ่าน · smoke ทั้ง 14 ตัว ทั้งสองรอบ
- [ ] positive control ของกลุ่ม A รันแล้ว
- [ ] run ทั้งสองรอบเขียน `artifacts/task3/v9_1/v91-<round>-A-*` พร้อม `hp.json` ที่บอก % ชนขอบ grid ของแต่ละตัว
- [ ] จดผลใน EXPERIMENT_LOG ของ v9.1 (ตาราง skill, SE, ผลต่างกับ R3, รายปี, ธงยุบ/หยุดเร็ว)
