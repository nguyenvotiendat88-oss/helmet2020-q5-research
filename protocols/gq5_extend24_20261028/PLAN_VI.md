# Lượt mở rộng G và Q5 từ epoch 13 đến 24

Ngày 28/10/2026. Người dùng đã đồng ý mốc mở rộng cố định thay vì chạy tới khi Q5 thắng.

- Tiếp tục G và Q5, seed 0, từ checkpoint epoch 12 của từng mô hình.
- Giữ nguyên checkpoint/run cũ. Không ghi lại hash để giả vờ cùng nghiên cứu.
- Giữ optimizer moments, số update và RNG/shuffle của từng nhánh; augmentation dùng chỉ số epoch toàn cục.
- Đặt lại learning rate của cả hai nhánh thành 0.000025, cosine 12 epoch bổ sung về 0.
- Đây là lịch 12 epoch cũ cộng 12 epoch khởi động lại learning rate; không phải cosine 24 epoch liên tục.
- Batch 256, FP32, 8 worker, cùng detector/proposal đã khóa và toàn bộ TRAIN.
- Lưu checkpoint mỗi epoch. VAL đầy đủ 14.600 frame tại epoch 18 và 24.
- So sánh chính là epoch 24 cố định. Lựa chọn phụ chọn macro VAL cao nhất trong 18/24, bằng nhau chọn sớm hơn.
- Dừng train sau epoch 24 cho cả G/Q5, bất kể kết quả. Không tự thêm seed, arm, thời gian hoặc TEST.
- Lượt mở rộng được chọn sau khi đã xem VAL của screening và run12; phải báo đây là phân tích phát triển, không phải xác nhận độc lập.

Run /workspace/runs/gq5_extend24_20261028
Log /workspace/runs/gq5_extend24_20261028.console.log
tmux q5extend24
Script /workspace/gq5_extend24_20261028/run_gq5_extend24.py

Các kiểm tra trước khởi động gồm: parent checkpoint hash, epoch12, scheduler và optimizer exposure; giữ nguyên moments khi đặt lịch mới; kết quả bước tiếp theo giống nhau sau save/resume; lịch mới kết thúc đúng 12 epoch; snapshot milestone giữ nguyên sau khi last.pt được thay.

Ước tính khoảng 4–6 giờ cho train và 4 lần VAL theo thời gian đo trước đó, không phải cam kết thời gian hoàn thành. GPU instance không tự Stop hoặc Destroy khi xong.
