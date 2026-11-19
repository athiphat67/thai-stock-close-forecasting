# Task 3 v9.1 — ข้อเสนอ: เทียบโมเดลดังจากงานวิจัย ภายใต้กติกา v9 ที่ล็อกแล้ว

- Status: ร่าง 2026-10-29 · **ยังไม่ได้รันอะไร** · เจ้าของตอบ §7 ตามข้อเสนอทุกข้อ และเพิ่มกลุ่ม CNN (§3.4) 2026-10-29 · รายละเอียดที่ล็อกจริงอยู่ใน [tickets/](../tickets/T00_COMMON.md) · ถ้าขัดกับไฟล์นี้ ticket ชนะ
- v9.1 คือ **การทดลองรอง** ไม่เปลี่ยนโมเดลที่ส่งสอบ (เช้า Huber, บ่าย HGB ตาม tag `v9-prereg`)
  ถ้ามีตัวไหนชนะ จะเป็นผู้สมัครของ v10 แล้วต้องพิสูจน์บนเดือนใหม่ที่ยังไม่เคยเห็น
- ไม่แก้ไฟล์ใดใน `versions9/` · โค้ด v9.1 อยู่ใน `versions9_1/` และเรียก feature store ของ v9 อย่างเดียว

## 1. งานวิจัยนิยมใช้โมเดลกลุ่มไหน

### 1.1 จาก literature review (นับว่าเปเปอร์ส่วนใหญ่ใช้อะไร)

| งาน | ขอบเขต | กลุ่มที่ใช้บ่อยที่สุด |
|---|---|---|
| Lin & Marques (2024) review ของ review | 10 systematic review | **SVM, LSTM, ANN** |
| Htun et al. (2023, Financial Innovation) | งานปี 2011–2022 | **RF, SVM, LSTM** |
| Kumbure et al. (2022, ESWA 197) | 138 บทความ 2000–2019 | **NN, SVM** · deep learning โตเร็ว |
| Sezer, Gudelek, Ozbayoglu (2020, Applied Soft Computing 90) | deep learning การเงิน 2005–2019 | **RNN (LSTM/GRU)** มากสุด รองลงมา CNN, DBN |
| Sonkavde et al. (2023, IJFS) | review + ทดลอง | RF, XGBoost, LSTM และ ensemble ของสามตัว |

สรุป: กลุ่มที่ "ดัง" ตามจำนวนเปเปอร์คือ **SVM, RF, ANN/MLP, LSTM/GRU** แล้วตามด้วย boosting (XGBoost/LightGBM)
และ CNN · ข้อควรระวัง: review พวกนี้นับความนิยม ไม่ได้บอกว่าตัวไหนชนะ random walk และหลายงานวัดด้วย accuracy ของทิศ ไม่ใช่ RMSE

### 1.2 จากงานที่เทียบโมเดลอย่างเข้มงวด (ใครชนะจริง)

| งาน | ผลที่เกี่ยวกับเรา |
|---|---|
| Gu, Kelly, Xiu (2020, RFS) | ชุดมาตรฐานของสายการเงิน: OLS, ENet, PCR, PLS, GLM, RF, GBRT, NN1–NN5 · NN3 ดีสุด แต่ใช้ข้อมูลหลายล้านแถว |
| Krauss, Do, Huck (2017, EJOR) | DNN, GBT, RF บนหุ้น S&P 500 รายวัน · RF ดีสุด ensemble ดีกว่าตัวเดี่ยว |
| Fischer & Krauss (2018, EJOR) | LSTM ชนะ RF/DNN/logistic บน S&P 500 รายวัน (ทายทิศ ไม่ใช่ขนาด) · ผลหายไปหลังปี 2010 |
| Zeng et al. (2023, AAAI) | DLinear ชนะ Transformer 5 แบบ · ชุดอัตราแลกเปลี่ยน "ทายค่าเดิม" เท่ากับตัวที่ดีที่สุด |
| Grinsztajn et al. (2022) | ข้อมูลตาราง ~10K แถว ต้นไม้ชนะ NN |
| Saly-Kaufmann, Wood, Peter-Calliess, Zohren (2026, arXiv 2603.01820) | futures รายวันหลายตลาด · **VSN+LSTM** (ลูกผสมแบบ TFT) Sharpe สูงสุด · xLSTM, PatchTST ดี · linear ไม่เสถียร · วัดด้วย Sharpe ไม่ใช่ RMSE |
| FinTSBridge (2025, arXiv 2503.06928) | PatchTST, iTransformer, DLinear, TimesNet บนข้อมูลการเงิน · ผลปนกัน ไม่มีผู้ชนะชัด |
| Noguer i Alonso & Pereira Franklin (2026, arXiv 2606.27100) | foundation model (TimesFM-2.5, Moirai-2.0, Chronos-2, TimeGPT) บนหุ้น US 5 ตัว · ชนะ random walk น้อยและไม่สม่ำเสมอ |
| Re(Visiting) TSFM in Finance (2025, arXiv 2511.18578) | Chronos zero-shot R² −1.27% · fine-tune แล้ว −0.59% · ฝึกใหม่บนข้อมูลการเงินดีกว่า |
| Kronos (AAAI 2026) | foundation model เฉพาะแท่งเทียน (K-line) จาก 45 ตลาด · วัดด้วย RankIC ไม่ใช่ RMSE |
| Hollmann et al. (2025, Nature 637) TabPFN v2 | foundation model สำหรับข้อมูลตาราง ≤ 10K แถว · ขนาดข้อมูลเราพอดี · ยังไม่เจองานที่ใช้กับหุ้น |

### 1.3 หลักฐานของเราเอง

- **v5 M4** (เป้าเดิม = ราคาปิดวันถัดไป): ลอง Ridge, DT, ExtraTrees, RF, XGBoost, LightGBM, CatBoost, ANN1, ANN2, hurdle,
  flat-MLP, GRU, TCN · Ridge ดีสุด · CatBoost ใกล้ ๆ · GRU แย่กว่าทายค่าเดิม 24–27% · TCN แย่กว่า 54–63%
  ([13_M4_MODEL_COMPARISON_REPORT](../../versions5/13_M4_MODEL_COMPARISON_REPORT.md))
- **v8:** CNN บนภาพ = HAR (หยุดที่ epoch 0) · **v9 M2:** MLP และ HGB แพ้ Huber ในรอบเช้า
- v9 ตัด LSTM/Transformer, XGBoost/LightGBM, ARIMA/GARCH ออกด้วยเหตุผลใน [04_REFERENCES](../../versions9/docs/04_REFERENCES.md) §8.3
  → **v9.1 เอากลับมาครบ** เพื่อให้รายงานตอบได้ว่า "ลองแล้ว ไม่ใช่แค่อ้างเปเปอร์"

## 2. สิ่งที่ล็อก (ใช้ของ v9 ทั้งหมด)

| เรื่อง | ค่าที่ใช้ | ที่มา |
|---|---|---|
| เป้า | `y_m = log(P_d / (C_{d−1} − D_d))` · `y_a = log(C_d / P_d)` | v9 02 §3 |
| เวลาตัด | 09:30 (เช้า) · 13:00 (บ่าย ใช้ข้อมูลถึง 12:30) | v9 00 |
| feature | `features.feature_cols(รอบ, หุ้น)` ไม่เพิ่มไม่ลด · clip 1%/99% ของหน้าต่างเทรน · แถวที่หายไม่ใช้ | v9 03 |
| การ fit | หน้าต่างขยาย · fit ใหม่รายสัปดาห์ · NN และ sequence model รายเดือน (จดไว้ก่อน) | v9 02 §1 |
| hyperparameter | เลือกปีละครั้งบน 3 holdout ติดกัน (30% ท้าย) · **grid ไม่เกิน 3 ค่าต่อโมเดล** เพื่อไม่จูนหนักกว่า Huber/HGB · โมเดลที่ early stop ใช้ epoch เฉลี่ย | v9 T1–T5 |
| โมเดลรวมสองหุ้น | + ตัวบอกหุ้น · แบ่งเทรน/validation ตามวันที่ | v9 T4 |
| seed | 340–344 (NN เฉลี่ย 5 seed) · ต้นไม้ seed 340 | v9 |
| คะแนน | skill เทียบ `ŷ = 0` · ผลต่างแบบจับคู่กับตัวที่ v9 เลือก (SE) · Clark–West · รายปี · วันที่ทุกตัวมีค่าทายเท่านั้น | v9 §4–§5 |
| `OMP_NUM_THREADS=1` | torch, LightGBM, XGBoost, CatBoost | memory: segfault/deadlock |

**สิ่งเดียวที่เปลี่ยนคือตัวโมเดล** · และ sequence model ต้องรับ "ประวัติย้อนหลัง" ซึ่ง v9 ไม่มี (§3.2)

## 3. รายชื่อโมเดลที่เสนอ (ครอบคลุมทุกกลุ่มที่นิยม)

ตัวเทียบที่รันซ้ำทุกครั้ง (ไม่นับ): `ŷ = 0` · ค่าเฉลี่ย · **v9 Huber (เช้า) / v9 HGB (บ่าย)** ด้วยโค้ดเดิม

### 3.1 กลุ่ม A — ข้อมูลตาราง (feature เวกเตอร์เดียวกับขั้น 2 ของ v9)

| ID | โมเดล | งานอ้างอิง | ทำไมใส่ | grid (≤ 3) |
|---|---|---|---|---|
| A01 | Lasso | Tibshirani 1996 | linear ที่เลือก feature เอง | α 3 ค่า |
| A02 | Elastic Net | Zou & Hastie 2005 · GKX | ชุดมาตรฐาน GKX | α 3 ค่า · l1_ratio 0.5 |
| A03 | PLS | GKX | ลดมิติแบบดูเป้า | 1/2/3 component |
| A04 | PCR | GKX | ลดมิติไม่ดูเป้า | 1/2/3 component |
| A05 | **SVR (RBF)** | Drucker 1997 · review อันดับต้น | นิยมสุดกลุ่มหนึ่งในทุก review | C 3 ค่า · ε = 0.1·sd(y) |
| A06 | kNN regression | review | baseline ไม่ใช้พารามิเตอร์ ที่ review เจอบ่อย | k ∈ {20, 50, 100} |
| A07 | **Random Forest** | Breiman 2001 · Krauss 2017 · GKX | นิยมอันดับต้น · ชนะใน Krauss | min_leaf ∈ {50, 100, 200} |
| A08 | ExtraTrees | Geurts 2006 | ใกล้ ridge ใน v5 | min_leaf 3 ค่า |
| A09 | **XGBoost** | Chen & Guestrin 2016 | ดังที่สุดของ boosting | depth ∈ {2, 3, 4} |
| A10 | **LightGBM** | Ke 2017 | GBRT ของ GKX | leaves ∈ {4, 8, 16} |
| A11 | CatBoost | Prokhorenkova 2018 | ดีสุดของต้นไม้ใน v5 | depth ∈ {2, 3, 4} |
| A12 | NN3 (32-16-8) | GKX | ตัวที่ดีที่สุดใน GKX | weight decay 3 ค่า |
| A13 | NN5 (32-16-8-4-2) | GKX | ดูว่าลึกขึ้นแย่ลงไหม | weight decay 3 ค่า |
| A14 | **TabPFN v2** | Hollmann 2025 (Nature) | foundation model ข้อมูลตาราง ≤ 10K แถว · ไม่ต้องจูน | ไม่มี |

### 3.2 กลุ่ม B — ลำดับเวลา (ดูย้อนหลัง L วัน)

รูปแบบข้อมูลเข้า (ตั้งก่อนรัน): **feature + y ของ 20 วันทำการล่าสุด** (ทุกค่ารู้ก่อนเวลาตัด) ต่อกับ feature ของวันนี้
· รอบบ่ายมีแบบที่สองคือ **แท่ง 1m ภาคเช้าวันนี้** เหมือนขั้น 6 ของ v9 (ใช้กับ LSTM และ PatchTST เท่านั้น)

| ID | โมเดล | งานอ้างอิง | ทำไมใส่ |
|---|---|---|---|
| B01 | ARIMAX (เลือก p, q ≤ 2 ด้วย AIC) | Box–Jenkins · review | baseline สถิติคลาสสิก · exog = feature ขั้น 2 |
| B02 | **LSTM** | Hochreiter 1997 · Fischer & Krauss 2018 | DL ที่นิยมที่สุดในทุก review |
| B03 | GRU | Cho 2014 | คู่ของ LSTM · v5 เคยแย่ |
| B04 | TCN | Bai, Kolter, Koltun 2018 | conv แบบ causal · v5 เคยแย่มาก |
| B05 | Transformer encoder | Vaswani 2017 | ตัวเทียบแบบ vanilla |
| B06 | **PatchTST** | Nie 2023 (ICLR) | Transformer ที่ดีที่สุดใน benchmark ส่วนใหญ่ |
| B07 | iTransformer | Liu 2024 (ICLR) | Transformer แบบสลับแกน เหมาะกับหลายตัวแปร |
| B08 | **DLinear** | Zeng 2023 (AAAI) | linear ที่ชนะ Transformer |
| B09 | N-HiTS | Challu 2023 (AAAI) | สาย MLP ของ N-BEATS |
| B10 | TFT แบบย่อ (VSN + LSTM) | Lim 2021 · Oxford 2026 | ตัวที่ดีสุดใน benchmark การเงินล่าสุด |
| B11 | xLSTM | Beck 2024 (NeurIPS) | ทางเลือก · ตัด ถ้าเวลาไม่พอ |

ขนาดทุกตัว: hidden 16–32 · 1–2 ชั้น · dropout 0.1 · AdamW 10⁻³ · early stop 10 epoch · เฉลี่ย 5 seed · grid = weight decay 3 ค่า

### 3.3 กลุ่ม C — foundation model ไม่ต้องเทรน (zero-shot)

| ID | โมเดล | ข้อมูลเข้า | หมายเหตุ |
|---|---|---|---|
| C01 | Chronos-Bolt small (Amazon) | อนุกรม y ย้อนหลัง | ใช้ feature ของเราไม่ได้ |
| C02 | TimesFM 2.5 (Google) | อนุกรม y ย้อนหลัง | อันดับดีใน 2606.27100 |
| C03 | Moirai 2.0 (Salesforce) | อนุกรม y ย้อนหลัง | อันดับดีใน 2606.27100 |
| C04 | Kronos small | แท่งเทียนครึ่งวัน (เช้า/บ่าย) ย้อนหลัง 256 วัน | ตัวเดียวที่ฝึกมาจากข้อมูลการเงิน |

### 3.4 กลุ่ม E — CNN (เพิ่มตามเจ้าของ 2026-10-29)

TCN (B04) เป็น CNN อยู่แล้ว และ 1D-CNN ขั้น 6 ของ v9 รันซ้ำเป็นตัวเทียบรอบบ่าย กลุ่มนี้เพิ่มแบบที่งานวิจัยนิยม:

| ID | โมเดล | ข้อมูลเข้า (เช้า / บ่าย) | งานอ้างอิง | ทำไมใส่ |
|---|---|---|---|---|
| E01 | 1D-CNN | หน้าต่าง 20 วัน / แท่ง 1m ภาคเช้า | Sezer 2020: CNN เป็นกลุ่ม DL ที่ใช้บ่อยรองจาก RNN | แบบพื้นฐาน |
| E02 | **CNN-LSTM** | เหมือน E01 | Livieris 2020 · review 2024: ลูกผสมนี้ชนะ CNN เดี่ยวบ่อยที่สุด | ลูกผสมที่นิยมที่สุด |
| E03 | WaveNet (dilated causal conv) | เหมือน E01 | Borovykh, Bohte, Oosterlee 2017 · DeepVol 2022 | ชนะ LSTM ในงานต้นฉบับ |
| E04 | TimesNet | หน้าต่าง 20 วัน / แท่ง 1m | Wu 2023 (ICLR) | พับอนุกรมเป็น 2D แล้วใช้ conv |
| E05 | ModernTCN | หน้าต่าง 20 วัน / แท่ง 1m | Luo & Wang 2024 (ICLR) | อันดับเฉลี่ยดีสุดใน benchmark การเงิน 2026 (preprint) |
| E06 | **2D-CNN บนภาพกราฟ OHLC** | กราฟ 20 วัน / กราฟ 1m ภาคเช้า | Jiang, Kelly, Xiu 2023 (JF 78(6)) | CNN บนภาพที่ดังที่สุดในสายการเงิน |
| E07 | spectrogram → CNN แบบแก้แล้ว | แท่ง 1m ภาคเช้า (บ่ายเท่านั้น) | งานของเราเอง v6–v8 | ปิดคำถามเดิมด้วยโครงที่แก้ข้อบกพร่องแล้ว · ทางเลือก |

**กฎออกแบบที่ใช้กับทุกตัวในกลุ่ม E** (บทเรียนจาก v6–v8):
- `ŷ = ridge(feature ขั้น 2) + g(CNN)` ให้ CNN ทำนายเฉพาะส่วนที่ ridge ทายไม่ได้ · มีแบบ CNN ล้วนคู่ด้วยเฉพาะ E02 และ E06
- ไม่ pool จนแกนเวลาหายหมด (v7: ภาพ 5×21 ถูก pool เหลือ 1×5) · ทิศของ return เป็นช่องแยก ไม่ปนกับขนาด
- รายงาน SD ของ ŷ เทียบกับ SD ของ y ทุกตัว · ถ้าต่ำกว่า 5% ถือว่ายุบไปทายค่าเฉลี่ย (v7: 0.0005 เทียบ 0.0124)
- รายงานว่าหยุดที่ epoch 0 กี่ % ของรอบ fit (v8: CNN = HAR เพราะหยุดที่ epoch 0)
- **E06 fit ใหม่ปีละครั้ง** เหมือน JKX ที่เทรนครั้งเดียว · v8 วัดได้ราว 11 นาทีต่อการ fit หนึ่งครั้งต่อ seed บน Air · 3 seed
- ขนาดภาพ จำนวนชั้น และ kernel ของ E06 เอาตามส่วนสถาปัตยกรรมของ JKX · **อ่านส่วนนั้นตอน Z0 ก่อนล็อก**

### 3.5 กลุ่ม D — รวมโมเดล

| ID | โมเดล | งานอ้างอิง |
|---|---|---|
| D01 | ค่าเฉลี่ยเท่ากันของตัวที่ดีที่สุดในแต่ละกลุ่ม A/B/C/E (เลือกตัวแทนกลุ่มบนช่วงเลือกเท่านั้น) | Rapach 2010 · combination puzzle |

**รวม รอบเช้า 38 · รอบบ่าย 41 สเปก** (A 14 · B 11 + แบบ 1m อีก 2 ในรอบบ่าย · C 4 · E 6 + ตัวล้วน E02p/E06p 2 · E07 บ่าย 1 · D 1 · ทะเบียนจริงใน T00 §2) · ถ้าต้องลดเหลือชุดหลัก เสนอตัวที่**ตัวหนา** + A02, A12, B01 = 13 ตัว

## 4. วัดผลและกันผลจากดวง

1. **คะแนนหลัก:** skill RMSE เทียบ `ŷ = 0` และผลต่างแบบจับคู่กับตัวที่ v9 เลือก ± SE · รายปี · แยกวัน XD
2. **Model Confidence Set** (Hansen, Lunde, Nason 2011, Econometrica) ที่ระดับ 10% บน squared error รายวัน
   → ตอบว่า "โมเดลไหนแยกไม่ออกจากตัวที่ดีที่สุด" แทนการดูอันดับ · 30 กว่าสเปกที่สหสัมพันธ์สูง อันดับ 1 ชนะเพราะดวงได้ง่าย
3. **การหักผลจากการลองหลายแบบ:** N = จำนวนสเปกจริง (v9 + v9.1) ตามสูตร Bailey–López de Prado · รายงานอย่างเดียว
4. **positive control:** ใส่สัญญาณปลอมที่รู้ขนาดแล้วดูว่า LSTM / PatchTST / TabPFN จับได้ไหม · ถ้าจับไม่ได้ แปลว่าแพ้เพราะ
   ฝึกไม่ขึ้น ไม่ใช่เพราะไม่มีสัญญาณ (ต่างจาก v8 ที่ CNN หยุดที่ epoch 0 โดยไม่มีใครรู้จนหลังจบ)
5. **กฎว่าใครเป็นผู้สมัคร v10:** อยู่ใน MCS · ชนะตัวที่ v9 เลือกเกิน 1 SE · บวกทุกช่วงปี · แล้วต้องชนะบนเดือนใหม่หลัง 2026-11-20

## 5. เวลาและเครื่อง (Air M2 16 GB ไม่มีพัดลม)

| กลุ่ม | ประมาณเวลาต่อรอบทาย ต่อช่วงข้อมูล | หมายเหตุ |
|---|---|---|
| A01–A11 | วินาที–นาที | sklearn, xgboost, lightgbm, catboost มีใน `.venv` แล้ว |
| A12–A13 | ราว 10–20 นาที | แบบเดียวกับ MLP ของ v9 |
| A14 TabPFN | ราว 30–60 นาที บน CPU | ต้องติดตั้ง + โหลด weight |
| B01 ARIMAX | ราว 10–30 นาที | ต้องติดตั้ง `statsmodels` |
| B02–B11 | รวมหลายชั่วโมง → **รันข้ามคืน** | fit รายเดือน · 5 seed · 3 grid · torch มีแล้ว |
| E01–E05, E07 | รวมหลายชั่วโมง → ข้ามคืน (คืนเดียวกับกลุ่ม B ไม่ได้) | fit รายเดือน · โครงเล็ก |
| E06 ภาพ OHLC | ราว 4 ชั่วโมงต่อรอบทาย ต่อช่วงข้อมูล | fit ปีละครั้ง · 3 seed · ช้าที่สุด แจ้งก่อนรัน |
| C01–C04 | ราว 10–30 นาทีต่อตัว | inference อย่างเดียว · ต้องโหลด weight (รวมราว 1–2 GB) |

- **ห้ามแย่งเวลา M5:** การซ้อมสอบของ v9 ต้องเสร็จก่อน 11-02 · v9.1 ไม่มีผลกับคะแนนสอบ
- **ห้ามแตะช่วงปิด F11 (2026-09-05 → 11-20):** ข้อมูลของ v9.1 จบที่ 2026-09-04

## 6. ลำดับงาน

| ขั้น | งาน | เสร็จเมื่อ |
|---|---|---|
| Z0 | เจ้าของตอบ §7 · ล็อกรายชื่อ + grid ในไฟล์นี้ · tag `v9.1-prereg` ก่อนรันโมเดลจริงตัวแรก | tag push |
| Z1 | ตัวสร้างหน้าต่างย้อนหลัง 20 วัน + test ว่าไม่เกินเวลาตัด (test เดียวกับ v9) · positive control | test ผ่าน |
| Z2 | กลุ่ม A + C (เร็ว) | ตาราง skill |
| Z3 | กลุ่ม B และ E (ข้ามคืน 2–3 คืน · แจ้งก่อนรันทุกคืน) | ตาราง skill |
| Z4 | D01 + MCS + รายงาน · ให้ Opus ตรวจ leakage แบบที่ทำกับ v9 | รายงานไทย + สไลด์ 1 หน้า |

## 7. สิ่งที่เจ้าของตัดสินแล้ว (2026-10-29: "โอเคหมด" = ทุกข้อตามคอลัมน์ขวา)

| ID | คำถาม | ผมเสนอ |
|---|---|---|
| Q1 | ช่วงข้อมูลที่ให้คะแนน | ช่วงเลือก 2019-07 → 2022-12 **และ** ช่วงยืนยัน 2023 → 2026-09-04 · ติดป้าย "หลังล็อก" ทั้งคู่ · ผลไม่เปลี่ยนตัวส่งสอบ |
| Q2 | shadow ในสัปดาห์สอบ 11-09 → 11-13 | ทำ ถ้าล็อกรายชื่อก่อน 11-09 · รายงานเป็นข้อมูลประกอบ (5 วันไม่มีพลังทางสถิติ) |
| Q3 | ขอบเขต | ชุดเต็ม รวมกลุ่ม CNN (เช้า 38 · บ่าย 41) · ถ้าเวลาไม่พอ ลดเป็นชุดหลัก 13 ตัวใน §3 |
| Q4 | ติดตั้งแพ็กเกจ + โหลด weight (statsmodels, tabpfn, chronos-forecasting, timesfm, uni2ts, Kronos) | ขออนุญาตทีละตัวตอนจะใช้ |
| Q5 | เริ่มเมื่อไร | Z0–Z2 ทำคู่กับการรอข้อมูล M5 ได้ (ไม่ใช้ข้อมูลใหม่) · Z3 ข้ามคืนหลัง M5 เสร็จ |

## แหล่งที่มา

- Lin & Marques (2024) review ของ review — https://research.usj.edu.mo/bibliography/ECHMTQA3
- Htun et al. (2023) Financial Innovation — https://doaj.org/article/5fec26d2e196475ca5110ca1a65f7c70
- Kumbure et al. (2022) ESWA 197:116659 — https://doi.org/10.1016/j.eswa.2022.116659
- Sezer, Gudelek, Ozbayoglu (2020) — https://arxiv.org/abs/1911.13288
- Sonkavde et al. (2023) IJFS — https://doaj.org/article/89cdc271aa224212a03e38b8a6758948
- Saly-Kaufmann et al. (2026) — https://arxiv.org/abs/2603.01820
- FinTSBridge (2025) — https://arxiv.org/abs/2503.06928
- Noguer i Alonso & Pereira Franklin (2026) — https://arxiv.org/abs/2606.27100
- Re(Visiting) TSFM in Finance (2025) — https://arxiv.org/abs/2511.18578
- Kronos (AAAI 2026) — https://ojs.aaai.org/index.php/AAAI/article/view/39730
- TabPFN v2 (Nature 2025) — https://pubmed.ncbi.nlm.nih.gov/39780007/
- Jiang, Kelly, Xiu (2023) JF 78(6):3193–3249 — https://economics.yale.edu/research/re-imagining-price-trends
- Borovykh, Bohte, Oosterlee (2017) WaveNet ทางการเงิน — https://arxiv.org/abs/1703.04691
- review ลูกผสม CNN-LSTM — https://arxiv.org/abs/2004.13408
- GKX, Zeng, Grinsztajn, Rapach และงานอื่นที่อ่านแล้ว — [v9 04_REFERENCES](../../versions9/docs/04_REFERENCES.md)
- ยังไม่ได้อ่านฉบับเต็ม: ทุกงานในตาราง §1.1 และ 2603.01820, 2606.27100, 2511.18578 (อ่านแค่บทคัดย่อ/สรุป) · Krauss 2017 และ
  Fischer & Krauss 2018 อ้างจากความรู้เดิม ยังไม่ได้เปิดยืนยันในรอบนี้
