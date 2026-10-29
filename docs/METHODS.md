# Mô hình và thay đổi đã kiểm tra

## Pipeline

Khi train classifier, crop được cắt từ box nhãn TRAIN. Khi đánh giá, crop được cắt từ proposals của RetinaNet. RetinaNet ResNet50-FPN đã được fine-tune trên split dự án, backbone bắt đầu từ ImageNet. Các run classifier dùng lại detector seed 0 cố định; đây không phải checkpoint HELMET gốc được tải về dùng nguyên.

| Module | Cấu trúc |
|---|---|
| G | ResNet18 → global mean pooling → Linear 512→10; head 5.130 tham số |
| Q5 | ResNet18 → feature 512×7×7 → projection 128 → 5 query cross-attention, 4 heads → FFN 128/256/128 → 2 output/query; head 200.074 tham số |
| GQ5 residual | Một ResNet18 chung; logits G + 5 hệ số tanh học được × logits Q5; head 205.209 tham số |
| Capacity control | Global MLP 512→392→10; head 205.026 tham số |

Thứ tự đầu ra: presence D/P1/P2/P3/P0, rồi helmet cho 5 vai. Mỗi query Q5 chia sẻ biểu diễn cho presence và helmet của cùng vai. Không có box đầu riêng hoặc giám sát vị trí bằng nhãn riêng cho từng vai; attention map không tự chứng minh định vị đúng.

BCE gốc tính presence cho cả 5 vai, helmet chỉ tại vai có người theo ground truth, rồi chia 5. Decoder chuẩn hóa xác suất kết hợp trên 36 cấu hình hợp lệ. Score cuối bằng max detector score của các box trùng tọa độ nhân xác suất lớp crop, rồi NMS và giới hạn số detection. Không gọi detector score này là objectness đã hiệu chỉnh.

## Các biến thể

- **Omit constant presence:** bỏ thành phần presence có giá trị hằng trong ontology, không bỏ vai khỏi decoder.
- **Conditional NLL:** likelihood trên 36 cấu hình hợp lệ, tính ổn định trong log-space; đây là nguyên lý đã biết.
- **GQ5 residual:** mô hình lai có tiền lệ về global/spatial residual attention; không báo như Q5 gốc.
- **Calibration:** bản áp lên 10 native logits và bản áp lên log của trung bình 36 class probabilities có không gian khác nhau, được báo riêng.
- **Long-tail:** asymmetric focusing chỉ cho presence; giữ nguyên helmet mask, sampler và kiến trúc.

## Code

Core nằm trong [training kit](../training_kit/helmetlab/). Các runner và biến thể thí nghiệm được giữ riêng trong [experiments](../experiments/).

ResNet, RetinaNet, các lớp Transformer và những phương pháp kế thừa không được trình bày như phát minh mới của nhóm.
