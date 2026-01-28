## Multimodal-For_SusDev — Baseline VQA cho ảnh san hô (CoralVQA)

### Highlights (con số chính)

- **Dữ liệu chuẩn (theo paper/dataset card)**: **12,805 ảnh / 277,653 QA**
- **Dữ liệu JSONL hiện có trong repo**:
  - **Train**: **226,726** QA pairs, **10,536** ảnh (unique)
  - **Test**: **27,984** QA pairs, **1,274** ảnh (unique)
  - **Số lớp câu trả lời** (unique answers từ train): **1,344** (khớp `num_classes=1344`)
- **Baseline 1 (chính)**: **ResNet50 (image) + BiLSTM (question) + Soft-Attention fusion → 1,344-way classification**
  - **Input image**: resize **224×224**
  - **Max question length**: **32 tokens**
- **Training (mặc định)**: **10 epochs**, **batch_size=64**, **lr=1e-4**, **AMP**, `num_workers=8`, `prefetch_factor=4`
- **Artifacts có sẵn**: checkpoint `epoch_1..epoch_10.pth` + `loss_plt.png`


---

## 1) Cấu trúc thư mục

- `src/train.py`: train baseline CNN+LSTM (classification)
- `src/model/cnn_lstm_based/vqa_model.py`: model baseline (ResNet50 + BiLSTM + soft attention)
- `src/model/cnn_lstm_based/checkpoints/checkpoints/`: checkpoints + `loss_plt.png`
- `src/data/dataset.py`: dataset loader theo LLAVA-format JSONL
- `src/benchmark_latency.py`: benchmark latency/throughput inference (không cần dataset)
- `src/predict_cnn_lstm.py`: suy luận 1 mẫu (ảnh + câu hỏi → câu trả lời)
- `data/CoralVQA_{train,test}.jsonl`: dữ liệu QA dạng JSONL
- `data/Code/`: code tham khảo/đính kèm cho các LVLM lớn (BLIP3, Qwen2.5-VL, InternVL, ...)

---

## 2) Setup môi trường

```bash
pip install -r requirement.txt
```

## 3) Baseline model (CNN + LSTM + Soft Attention)

### Kiến trúc

- **Image encoder**: ResNet50 → vector **1024**
- **Text encoder**: Embedding + **BiLSTM(512)** → question embedding **1024**
- **Fusion**: soft-attention (trên image features) + concat
- **Answer head**: MLP → logits **1,344 classes**

### Training (reproduce)

Chạy theo dạng module để tránh lỗi import:

```bash
python -m src.train --config configs.yaml
```

Mặc định script sẽ:
- split train/val = **85% / 15%**
- in ra **Accuracy theo epoch** (Top-1 trên val)
- lưu checkpoint theo epoch và lưu `loss_plt.png`


> - Windows (PowerShell): `python -m src.train --config configs.yaml > train.log`
> - Linux/macOS: `python -m src.train --config configs.yaml | tee train.log`

---

## 4) Inference (1 mẫu)

Script dưới đây load checkpoint CNN+LSTM và trả về câu trả lời dạng string (xây vocab answer từ train JSONL, đúng logic `sorted(unique_answers)`):

```bash
python -m src.predict_cnn_lstm --checkpoint src/model/cnn_lstm_based/checkpoints/checkpoints/epoch_10.pth --image "<path/to/image.jpg>" --question "Is there algae near the coral?"
```

---

## 5) Latency / Throughput benchmark 

Chạy benchmark inference trên input ngẫu nhiên (không cần ảnh/dataset):

```bash
python -m src.benchmark_latency --device cpu --batch-size 1 --warmup 30 --steps 200
```


## 6) Kết quả hiện có trong repo

### Loss curve (từ checkpoint folder)

File: `src/model/cnn_lstm_based/checkpoints/checkpoints/loss_plt.png`

- **Train loss** giảm mạnh khoảng **~1.15 → ~0.36** sau **10 epochs**
- **Val loss** ổn định quanh **~0.63–0.67** (dao động nhẹ về cuối)

### Accuracy / Time

Repo hiện **chưa commit file log** để cố định các con số **accuracy/time**. Khi bạn chạy lại `src/train.py` và redirect log (như mục 3), bạn sẽ có thể điền vào bảng sau:

| Model | Task framing | #classes | Epochs | Batch | Val Acc@1 | Train time/epoch | Inference p50 (ms) | Throughput (img/s) |
|------|--------------|----------|--------|-------|-----------|------------------|--------------------|--------------------|
| ResNet50 + BiLSTM + SoftAttn | 1,344-way classification | 1,344 | 10 | 64 | (fill from `train.log`) | (fill) | (fill from benchmark) | (fill from benchmark) |

---

## 7) Notes / Known gotchas

- `configs.yaml` dùng đường dẫn tương đối để portable; bạn có thể chỉnh `data_dir`/`image_dir` theo máy.
- Nếu chạy offline và backbone muốn tải weights, hãy dùng benchmark/predict (đã tắt tải weights mặc định trong benchmark) hoặc cache weights trước.

---

