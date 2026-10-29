# Xác nhận G/Q5 và chuẩn bị đối chứng — 28/10/2026

Người dùng yêu cầu hướng đi tạo bài nghiên cứu và xác nhận GPU đang thuê, ưu tiên đủ bằng chứng. Thực thi trước mắt: thêm seed1 và2 của G/Q5 sau khi lượt full-VAL calibration hiện tại hoàn tất. Đây là4 lượt fit mới, mỗi lượt24 epoch theo lịch12+12 với LR restart, cùng recipe số0 đã chạy. Không chạy tới khi Q5 thắng.

## Phạm vi được xếp hàng

- Seed1:G,Q5 rồi seed2:G,Q5; mỗi seed chạy phase12 cho hai arm, sau đó phase24 cho hai arm.
- Mỗi arm khởi tạo ImageNet gốc; seed kiểm soát head, shuffle, augmentation và RNG. Detector/proposal vẫn giữ seed0 đã xác minh. Độ biến thiên đo được là của classifier trên detector cố định.
- AdamW1e-4 cosine12 rồi restart2.5e-5 cosine12; giữ optimizer moments, RNG, thứ tự data. Batch256, FP32,8 worker spawn, crop224, margin0.05, loss không đổi.
- VAL4/8/12/18/24 như lượt trước, so sánh chính ởepoch24. Không early stopping, không nới budget theo điểm.
- Lưu checkpoint model/optimizer/scheduler/RNG sau mỗi epoch, giữ milestone, kiểm tra hash; nếu ngắt trong VAL phải hoàn tất VAL trước khi train tiếp.
- Không chạy song song GPU với full-VAL; dùng khóa thực thi. Queue chỉ đợi dependency hiện tại rồi chạy4 fit hữu hạn.
- Lỗi dependency, source drift hoặc dung lượng dưới8GiB thì dừng có trạng thái; không lặp lại vô hạn.
- Không TEST, không tự hủy/stop instance, không sửa recipe/kit hay kết quả cũ.

Sau khi xong có replication_summary.json thống kê epoch24 của seed0/1/2, mean±sample SD. Seed0 đã tham gia phát triển recipe nên phải công bố lịch sử này. Calibration cho các seed mới sẽ là bước riêng, chưa được xếp hàng trong script này.

## Đối chứng đã chuẩn bị thiết kế, chưa xếp hàng

G+ có số tham số head gần Q5: kiểm tra tác động capacity. Q5shift giữ5 query và dung lượng, đổi pairing presence/helmet: kiểm tra grouping. Dùng cùng recipe và paired seeds khi thực hiện; không nhập chúng vào cột G hoặc Q5. Engine hỗ trợ cấu hình các arm này, nhưng CLI và queue lần này chỉ chạyG/Q5 seed1/2.

## Đường dẫn

Source:/workspace/gq5_confirmation_20261028
Outputs:/workspace/runs/gq5_confirmation_20261028
Queue log:/workspace/runs/gq5_confirmation_20261028.console.log
Queue state:queue_status.json
Mỗi seed:seedN/phase12 vàseedN/phase24, với status.json,progress.json vàcheckpoint từng arm.

Chưa ước tính tổng thời gian trước khi có thêm tốc độ thực tế của lượt hiện tại. Không coi thời gian thuê GPU là ngân sách chứng minh giả thuyết.
