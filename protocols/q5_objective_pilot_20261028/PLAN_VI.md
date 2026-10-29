# Ưu tiên kiểm chứng đóng góp — 28/10/2026

Ứng viên: huấn luyện role-query phù hợp với không gian cấu hình hợp lệ. Đây là ứng dụng conditional likelihood đã biết, chưa có bằng chứng novelty thuật toán hoặc cải thiện mô hình. Phát hiện riêng cần kiểm chứng là tác động của nhiệm vụ giám sát không ảnh hưởng trực tiếp đến decoder trong pipeline hiện tại.

Trong36 lớp, D luôn có mặt. Trong xác suất điều kiện chuẩn hóa lý tưởng, xác suất presence_D là thừa số chung nên triệt tiêu. BCE vẫn tạo gradient từ nó; query D dùng chung cho presence/helmet. Loại bỏ nhiệm vụ này có thể giúp, không giúp hoặc làm mất regularization hữu ích.

## Pilot cố định

- G và Q5 với2 objective: bỏ loss presence có giá trị hằng theo ontology; conditional NLL trên36 cấu hình.
- Tổng4 fit mới, mỗi fit4epoch từ ImageNet seed0, batch256,FP32,8worker,cosine12 đoạn đầu, lr1e-4 như baseline.
- So với G/Q5 epoch4 đã chạy, cóhashkhóa. Phải xác minh cùng sourcekit,split,pixels,ImageNet,detector,modelinitialization và sampleorder. Nếu khác thì dừng, không so lẫn.
- Cùng toàn bộ VAL 14.600 frame, decoder cũ, threshold0.001, NMS0.5, cap300; không calibration trong phép so sánh loss này.
- Cả hai loss giữ phép chia5 như loss gốc. Conditional NLL dùng log-space để ổn định; decoder cũ có floor1e-12 được giữ khi đánh giá, nên chưa thay đổi đồng thời loss và xử lý số học.
- Chỉ chọn mốc4 định trước. Gate sàng lọc: Q5 tăng macro ít nhất1 điểm phần trăm và weighted giảm không quá0.5 điểm so với Q5 gốc epoch4. Đây là ngưỡng nội bộ để quyết định nghiên cứu tiếp, không phải tiêu chuẩn nhận bài hội nghị.
- Lưu model/optimizer/scheduler/RNG sau mỗi epoch trước VAL. Lỗi, source drift hoặc dung lượng dưới8GiB thì dừng có log; không tự đổi recipe hoặc kéo dài đến khi có điểm thuận lợi.
- Không đọc TEST, không chạy dữ liệu Việt Nam, không tự mở rộng phương pháp sau pilot.

## Thứ tự thực thi dự kiến

1. Hoàn tất lượt full-VAL calibration đang chạy.
2. Pilot bốn lượt fit kiểm tra cơ chế: G/Q5 bỏ loss hằng, G/Q5 conditional NLL.
3. Tiếp tục các lượt lặp G/Q5 seed1,2 đã lên kế hoạch, không đổi recipe của chúng.

Chỉ thay hàng đợi seed khi xác minh nó còn chờ và chưa có tiến trình train con. Không ngắt epoch đang học. File cũ, checkpoint và kết quả đã có được giữ.

Source mới: /workspace/q5_objective_pilot_20261028.
Kết quả mới: /workspace/runs/q5_objective_pilot_20261028.
Trạng thái điều phối: queue_status.json; mỗi objective có status.json và progress.json.

## Đóng góp kỳ vọng và giới hạn

Đóng góp có thể kiểm chứng gồm: phân tích tác động của giám sát dư thừa trong pipeline này; đối chứng loss khớp cách giải mã trên cả global head và role-query head; và đánh giá lặp lại có kiểm soát. Chưa có bằng chứng để tuyên bố phương pháp mới trên thế giới, Q5 vượt baseline, hoặc khả năng nhận bài cao.

Conditional likelihood và học với ràng buộc đầu ra đều có tiền lệ. Cần đối chiếu Lin et al. ICIP2021, Query2Label, semantic loss, và các loss cân bằng lớp trước khi viết novelty. Nếu chỉ bỏ một thành phần loss đem lại cải thiện nhỏ, kết quả phù hợp với phân tích thực nghiệm, chưa đủ cho tuyên bố thuật toán mới.

Mẫu gradient đã đo ở chế độ eval trên query vectors cho thấy xung đột có thể xảy ra; không chứng minh cập nhật AdamW thực tế làm giảm khả năng tổng quát hóa. Gradient từ negative presence cũng không luôn xung đột, nên không được quy nguyên nhân đơn giản cho số lượng mẫu âm.

Muốn bài mạnh hơn cần: ablation giải thích cơ chế, các đối chứng hiện có phù hợp, nhiều seed, TEST khóa sau khi chọn bằng VAL, khoảng tin cậy theo clip và báo chi phí. Không có xác suất nhận bài được xác lập từ các kiểm tra phần mềm hoặc một seed.
