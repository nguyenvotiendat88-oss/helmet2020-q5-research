# HELMET2020 — G/Q5 research workspace

**Kho nghiên cứu nội bộ, private.** Đây là báo cáo thực nghiệm đang phát triển, **không phải công bố đã được hội nghị/tạp chí chấp nhận**.

Nhóm: Nguyễn Võ Tiến Đạt · Phan Anh Đức · Nguyễn Trường Khang — Đại học FPT.

Snapshot trạng thái: **23:47 ngày 29/10/2026, UTC+7**. Repo không tự cập nhật theo máy GPU.

## Kết quả chính đã hoàn tất

G/Q5 gốc, checkpoint epoch 24 theo lịch **12+12 có khởi động lại learning rate**, seed 0/1/2; detector seed 0 cố định. Cùng 14.600 frame VAL, 30 lớp có ground truth. Không phải TEST.

| Mô hình | Macro AP50 mean ± SD (%) | Weighted AP50 mean ± SD (%) |
|---|---:|---:|
| G | 28.5300 ± 1.7529 | 75.5066 ± 0.2997 |
| Q5 | 27.9341 ± 1.7661 | 74.2800 ± 0.0923 |

**Q5 gốc chưa vượt G ổn định.** Không chọn riêng seed/metric tốt nhất để tuyên bố thắng.

| Thí nghiệm | Kết quả / trạng thái |
|---|---|
| Loss trên cấu hình hợp lệ, epoch 4 / seed 0 | Q5 macro 26,0287 → 28,0267%, vẫn dưới G gốc 28,8617%; mức tăng nhạy với lớp chỉ có 1 GT |
| GQ5 residual, epoch 4 / seed 0 | Macro 28,4408%, weighted 75,6854%; chưa đạt tiêu chí sàng lọc so với G gốc |
| Ensemble 3 seed + calibration | Trên 9.400 frame / 24 lớp ngoài clip fit: G 33,3914 / 80,6108%; Q5 32,1445 / 80,4994% (macro / weighted) |
| Logit adjustment | Cả G/Q5 chọn τ = 0 trong grid đã đặt; không được chọn thay cho calibration đơn thuần |
| Selective role prediction | Ứng viên đồng thuận chưa vượt baseline xác suất biên; không đạt gate mở rộng |
| Long-tail loss, 3 seed | **Đang chạy** tại snapshot: G, seed 0, epoch 4; chưa có tổng kết mới |

## Đọc tài liệu

- [Bảng đo đầy đủ](docs/RESULTS.md): phân biệt population/epoch/seed và kết luận.
- [Mô hình và các phần kế thừa](docs/METHODS.md).
- [Tái lập và giới hạn của gói](docs/REPRODUCIBILITY.md).
- [Nguồn, quyền sử dụng và trích dẫn](docs/PROVENANCE.md).
- [AP theo lớp](results/per_class_3seed.csv).
- [Bằng chứng JSON](evidence/) và [source thí nghiệm](experiments/).

## Giới hạn cần giữ khi viết bài

- Các kết quả trên là **VAL development đã xem nhiều lần**. TEST chưa dùng trong những run này; không phải xác nhận độc lập.
- Split hiện tại khác split công bố gốc; các site xuất hiện ở cả TRAIN/VAL. Không gọi là unseen-camera generalization.
- Chưa xác minh đầy đủ parity của scorer với MATLAB chính thức. Không so trực tiếp số AP với số khác metric/protocol trong paper.
- Mean ± sample SD mô tả 3 classifier seed trên detector cố định; không phải khoảng tin cậy hoặc độ ổn định của toàn pipeline.
- Calibration, conditional likelihood, ensemble và residual attention có tiền lệ. Code mới không tự chứng minh thuật toán mới.
- Chưa có kết quả test Việt Nam. Cấu hình G+/Q5shift/Q10 trong kit không đồng nghĩa đã train.

## Nội dung được đưa lên

Source Python/shell, protocol, metadata split/ontology, metric/hash receipts, báo cáo và hình vẽ số đo. **Không có ảnh dataset, checkpoint, private SSH key, token, file .env, log terminal thô hoặc endpoint máy thuê.** Đây là snapshot nghiên cứu; không phải bản sao lưu đầy đủ của máy GPU.

Kiểm tra gói offline bằng Python chuẩn:

```bash
python3 tools/verify_snapshot.py
```
