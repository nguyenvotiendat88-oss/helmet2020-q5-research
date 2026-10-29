# Tái lập và giới hạn snapshot

## Có trong repo

Core model/loss/decoder/scorer; source từng run; protocol; split recipe và ontology; metrics và hash receipts; hình đánh giá.

## Không có trong repo

JPEG dataset, manifest train/val/test từng frame, pretrained weights, checkpoints, proposal streams, pixel hash maps, full predictions và cached logits/probabilities. Các file đó cần được cung cấp từ artifact gốc trước khi rerun. Đây **không phải gói tái lập toàn bộ chỉ bằng git clone**.

Các script lịch sử tham chiếu `/workspace/HELMET2020_training_kit_v1`, `/workspace/helmet_images` và `/workspace/runs/...`. Cần giữ bản sao artifact và cấu trúc đường dẫn gốc, hoặc viết adapter cho một run mới. Không sửa source rồi resume một study đã khóa.

## Môi trường đã dùng

Python 3.12.15, PyTorch 2.8.0+cu128, torchvision 0.23.0+cu128, numpy 2.2.6, Pillow 11.3.0, PyYAML 6.0.3. Xem [runtime](../environment/observed_training_runtime.json).

Requirements của kit chỉ là một phần môi trường, không tự cài NVIDIA driver. Đọc báo cáo không yêu cầu cài môi trường GPU.

## Kiểm tra snapshot offline

```bash
python3 tools/verify_snapshot.py
```

Lệnh này kiểm tra hash các file export, cú pháp Python, JSON và tính nhất quán của bảng số. Không import model, không download, không train và không đọc TEST.

## Hash và provenance

`provenance/export_manifest.json` ghi SHA256 nguồn trước export và SHA256 file trong repo. Bản export chuẩn hóa xuống dòng LF và bỏ dòng trống dư ở cuối file. Đường dẫn Mac cá nhân được thay bằng `/local/helmet-research` hoặc `/local/user` khi export. Numeric metrics và hash artifact gốc không đổi.

Hash trỏ tới một checkpoint không được kèm theo repo là receipt cho artifact bên ngoài; không có nghĩa checkpoint đã được sao lưu ở đây. Source hash trong study binding thuộc lần chạy gốc. Bản export có thể khác hash vì đổi đường dẫn, được đánh dấu `modified_for_export`; không dùng bản đã đổi để resume trực tiếp study gốc.

## Giao thức đánh giá

TRAIN: 58.200 frame / 582 clip / 181.176 crop. VAL: 14.600 frame / 146 clip, 30 lớp có ground truth. Metadata giữ tên file gốc nên tên folder ảnh không quyết định split mới.

TEST chưa được dùng trong các run đã báo. Các site xuất hiện ở cả TRAIN/VAL, nên không tuyên bố khả năng tổng quát hóa sang camera chưa thấy. Scorer nội bộ chưa được xác minh đầy đủ với MATLAB chính thức. Không so trực tiếp chỉ số khác protocol với paper 2020. Chưa có đánh giá trên dữ liệu Việt Nam.

Protocol cũ trong `protocols/` là lịch sử. Đọc README và RESULTS để biết trạng thái tại thời điểm snapshot.

## Không tự động chạy trên GitHub

Repo không chứa GitHub Actions và không tự train, deploy hoặc công khai dữ liệu. Đây là snapshot private, không thay thế việc theo dõi GPU.
