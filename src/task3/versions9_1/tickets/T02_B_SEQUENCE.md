# T02 — กลุ่ม B: โมเดลลำดับเวลา (B01–B11, B02m, B06m)

- Status: ร่าง 2026-10-29 · ขึ้นกับ [T00](T00_COMMON.md) · โค้ด: `versions9_1/models_seq.py` (ใช้ร่วมกับ T04)
- คำถามที่ตอบ: ถ้าให้โมเดลเห็น**ประวัติ 20 วัน** (ไม่ใช่แค่ feature ของวันนี้) โมเดลที่ review ว่านิยมที่สุด (LSTM/GRU)
  และโมเดลรุ่นใหม่ (PatchTST, iTransformer, TFT) ทำได้ดีกว่า v9 ไหม
- ที่คาดไว้ (เขียนก่อนรัน): DLinear ≈ ridge · LSTM/GRU/Transformer แพ้ R3 (v5: GRU −27%, TCN −63% เทียบทายค่าเดิม)
  · TFT-lite ดีสุดในกลุ่ม (Oxford 2026) แต่ไม่ชนะ R3 · ARIMAX ≈ ridge เพราะส่วน AR ของ return รายวันเกือบ 0

## 1. ข้อมูลเข้า (ตายตัวทุกตัวที่เป็น torch)

| ส่วน | รูปร่าง | ที่มา |
|---|---|---|
| หน้าต่าง `W` | `(L=20, p+2)` | `inputs.windows(t, rnd, cols(rnd, None, "pooled"))` = feature รวมสองหุ้นของ 20 วันก่อนหน้า + `y_m` + `y_a` |
| วันนี้ `x` | `(p,)` | `cols(rnd, None, "pooled")` ของวัน d (มี `is_kbank`) |
| แท่ง 1m `S` (B02m, B06m เท่านั้น) | `(2, 155)` | `inputs.minute_seq` ของ v9 (return และปริมาณ ทุกนาที 09:56 → 12:30) |

- clip 1%/99% และปรับสเกลรายช่องจากหน้าต่างเทรน (ช่องของ `W` ใช้สถิติเดียวกับคอลัมน์เดียวกันของ `x`)
- **รูปแบบหัวโมเดลเดียวกันทุกตัว:** `encoder(W) → h (16 มิติ)` แล้ว `ŷ = Linear([h, x])` ·
  ทำให้ทุกตัวเห็นข้อมูลชุดเดียวกัน ต่างกันแค่ encoder · ถ้าไม่มี `x` โมเดลลำดับเวลาจะเสียเปรียบตั้งแต่ต้น (feature ต่างประเทศ
  ของวันนี้คือแหล่งสัญญาณหลักของรอบเช้า)
- รวมสองหุ้น · แถวเทรนและแถว validation แบ่งตามวันที่ (T4 ของ v9)

## 2. การเทรน (ตายตัวทุกตัว torch ในกลุ่ม B)

| เรื่อง | ค่า |
|---|---|
| loss | MSE บน `y / sd(y)` |
| optimizer | AdamW lr 10⁻³ · batch 64 |
| epoch | ≤ 100 · patience 10 บน validation (เหมือน 1D-CNN ของ v9) |
| dropout | 0.1 (หลัง encoder) |
| seed | 340–344 · เฉลี่ยค่าทาย 5 seed |
| fit ใหม่ | **รายเดือน** (จดไว้ก่อนแล้วตามโปรโตคอล v9) |
| grid | weight decay ∈ {10⁻⁴, 10⁻³, 10⁻²} · ปีละครั้ง บน 3 ช่วง · fit ใหม่ด้วย epoch เฉลี่ย |
| init | ค่าเริ่มต้นของ PyTorch |
| ความทำซ้ำได้ | CPU, `torch.use_deterministic_algorithms(True)`, `set_num_threads(1)` |

## 3. รายตัว

| ID | encoder | ขนาด | อ้างอิง | หมายเหตุการเขียน |
|---|---|---|---|---|
| B01 | **ARIMAX** (statsmodels `SARIMAX`) รายหุ้น · exog = `cols(rnd, s, "own")` ที่ปรับสเกล | `order=(p,0,q)`, `trend="c"` | Box–Jenkins | **ไม่ใช้หัวร่วม** (ไม่ใช่ torch) · grid `(p,q)` ∈ {(1,0), (0,1), (1,1)} เลือกปีละครั้งบน 3 ช่วงเหมือนตัวอื่น · fit ใหม่ทุกสัปดาห์ · ระหว่างสัปดาห์ใช้ `results.append(y, exog, refit=False)` หลังรู้ y ของวันนั้นเพื่อทายวันถัดไป · ไม่ fit ใหม่ระหว่างสัปดาห์ · `y` ที่ใช้คือ log return ของรอบนั้น (ไม่ใช่ `y/sd`) |
| B02 | LSTM 1 ชั้น | hidden 32 → Linear → 16 | Hochreiter 1997 · Fischer & Krauss 2018 | ใช้ hidden ของขั้นสุดท้าย |
| B03 | GRU 1 ชั้น | hidden 32 → 16 | Cho 2014 | เหมือน B02 |
| B04 | TCN | 4 residual block, kernel 3, dilation 1/2/4/8 (receptive field 31 ≥ 20), 16 ช่อง, causal padding | Bai, Kolter, Koltun 2018 | ใช้ค่าที่ขั้นสุดท้าย · ไม่มี weight norm (ไม่ช่วยเมื่อโมเดลเล็ก) |
| B05 | Transformer encoder | d_model 32, 2 head, 1 ชั้น, FF 64, positional embedding แบบเรียนรู้ | Vaswani 2017 | token ละวัน · ใช้ token สุดท้าย → 16 |
| B06 | PatchTST | patch 5 stride 5 → 4 patch ต่อช่อง · แยกช่อง (channel-independent) ใช้ encoder ร่วม · d_model 16, 2 head, 1 ชั้น | Nie 2023 | flatten (ช่อง × patch × d) → Linear → 16 · ไม่ใช้ RevIN (ปรับสเกลแล้ว) |
| B07 | iTransformer | แต่ละช่อง (อนุกรม 20 วัน) เป็น 1 token · Linear 20 → 32 · attention ข้ามช่อง 2 head, 1 ชั้น | Liu 2024 | mean ข้าม token → 16 |
| B08 | DLinear | แยก trend ด้วย moving average kernel 5 + ส่วนที่เหลือ · Linear 20 → 1 ต่อส่วน ต่อช่อง (น้ำหนักร่วมทุกช่อง) | Zeng 2023 | ได้ `(p+2)×2` ค่า → Linear → 16 |
| B09 | N-HiTS | 3 block · max-pool ตามเวลา kernel {4, 2, 1} · MLP 2×64 ต่อ block · backcast/forecast แบบ residual | Challu 2023 | input ของ block = หน้าต่างที่ pool แล้ว flatten · ผลรวม forecast ของทุก block → 16 |
| B10 | TFT-lite (VSN + LSTM) | Variable Selection Network (GRN ต่อตัวแปร, softmax น้ำหนัก) ทุกขั้นเวลา → LSTM 32 · `x` ผ่าน VSN ของตัวเอง | Lim 2021 · Oxford 2026 (VSN+LSTM ดีสุด) | **ไม่มี** interpretable multi-head attention และ quantile loss (ตัดตาม benchmark Oxford) · บันทึกน้ำหนัก VSN เฉลี่ยรายปีไว้ใน `diag.json` |
| B11 | xLSTM | 2 block (mLSTM + sLSTM) embedding 32 | Beck 2024 | ใช้แพ็กเกจ `xlstm` แบบ backend torch ล้วน · **ถ้าติดตั้งไม่ได้หรือบังคับใช้ CUDA → จด "ไม่ได้รัน" ไม่เขียนเอง** |
| B02m | LSTM บน `S` | hidden 32 · 155 ขั้น | | บ่ายเท่านั้น · หัว `ŷ = Linear([h, x])` · `W` ไม่ใช้ |
| B06m | PatchTST บน `S` | patch 15 stride 15 → 10 patch · 2 ช่อง | | บ่ายเท่านั้น · เหมือน B06 นอกนั้น |

## 4. ข้อที่ต้องระวัง

1. **เส้นแบ่งเวลาในหน้าต่าง:** แถวสุดท้ายของ `W` คือ d−1 เสมอ · ห้ามมีแถว d · test 2 ของ T00 ต้องครอบคลุมทุกตัว
2. **วันหยุดยาว:** หน้าต่างนับวันทำการของหุ้นนั้น · ไม่เติมวันที่ไม่มีการซื้อขาย · `gap_days` มีอยู่ใน feature แล้ว
3. **หุ้นสองตัวในหน้าต่าง:** หน้าต่างของ KBANK มีแค่แถวของ KBANK · ไม่ผสม
4. **feature ที่เริ่มทีหลัง:** รอบเช้า `fut_post` เริ่ม 2019 → หน้าต่างแรกที่ครบคือ ~2019-02 · เทรนจริงจาก 2019 (เท่ากับ HGB v9 รอบเช้า)
   · รอบบ่ายใช้ชุด `pooled_early` ของ v9 (ไม่มี `LATE`)
5. **ARIMAX:** ถ้า fit ไม่ converge ในสัปดาห์ใด ให้ใช้ค่าทายของสัปดาห์ก่อน (คงพารามิเตอร์เดิม `append` ต่อ) และนับจำนวนครั้งใน `diag.json`
6. **ยุบเป็นค่าเฉลี่ย:** รายงาน `sd(ŷ)/sd(y)` ทุกตัว · LSTM/Transformer บนข้อมูลน้อยมักยุบ (v7)

## 5. test เพิ่ม

1. แต่ละ encoder: เปลี่ยนค่าในขั้นเวลาสุดท้ายของ `W` แล้ว `h` ต้องเปลี่ยน (โมเดลมองเห็นวันล่าสุดจริง) · สำหรับ B04 ตรวจว่า
   ขั้นเวลาแรกก็มีผล (receptive field ครอบ 20)
2. B01: `append` ทีละวันได้ค่าทายเท่ากับ fit ใหม่ด้วยพารามิเตอร์เดิมบนข้อมูลที่ยาวขึ้น (`atol` 1e-8)
3. B08 ที่น้ำหนักเป็นค่าเฉลี่ยเท่ากันได้ moving average ที่คำนวณมือ

## 6. positive control

- `x` ของ T00 §7 ใส่ใน `x` ของวันนี้ (ไม่อยู่ใน `W`) · ทุกตัวในกลุ่ม · B01 ใส่เป็น exog
- ที่คาด: ทุกตัวผ่าน เพราะหัวเป็น Linear บน `[h, x]` · **ถ้าตัวไหนไม่ผ่าน แปลว่าการเทรนพัง** (ไม่ใช่ข้อมูล) → เป็นบั๊ก แก้ได้ตาม T00 §1.4

## 7. การรันและเวลา

```
OMP_NUM_THREADS=1 .venv/bin/python -W ignore -m src.task3.versions9_1.zoo morning --group B
OMP_NUM_THREADS=1 .venv/bin/python -W ignore -m src.task3.versions9_1.zoo afternoon --group B
```

- ประมาณ: fit รายเดือน ~92 เดือน × 5 seed × โมเดลเล็กบน ~4,000 หน้าต่าง · เลือกรายปี 8 ปี × 3 ค่า × 3 ช่วง × 5 seed
  → ราว 10–40 นาทีต่อตัวต่อรอบ · ทั้งกลุ่มสองรอบ **ราว 6–12 ชั่วโมง = 1 คืน** · ARIMAX ราว 30 นาที
- วัดเวลา B05 (ช้าสุด) ด้วย `--smoke` ก่อน แล้วแจ้งเจ้าของพร้อมตัวเลขประมาณก่อนรันจริง
- ต้องติดตั้ง: `statsmodels`, `xlstm` (ขออนุญาต)

## 8. เสร็จเมื่อ

- [ ] test §5 + test 2 ของ T00 ครอบทุกตัว · smoke ทุกตัวทั้งสองรอบ
- [ ] positive control ผ่านทุกตัว (หรือแก้บั๊กแล้วจด)
- [ ] run ทั้งสองรอบ · `diag.json` มี `sd(ŷ)/sd(y)`, % หยุดที่ epoch 1, น้ำหนัก VSN ของ B10, จำนวน ARIMAX ที่ไม่ converge
- [ ] จดผลใน EXPERIMENT_LOG
