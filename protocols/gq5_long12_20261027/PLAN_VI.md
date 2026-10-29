# G và Q5 12 epoch ngày 27 tháng 10 năm 2026

Yêu cầu đã xác nhận: chạy hai nhánh G và Q5, seed0, mỗi nhánh12epoch, chấm VAL ở4/8/12 trên GPU Vast.ai đang thuê. Không chạy các arm khác, seed khác hoặc TEST trong lượt này.

Mục tiêu: theo dõi đường học dài hơn sau screening2epoch; không giả định Q5 sẽ cải thiện. Lượt này là nghiên cứu phát triển bằng VAL đã sử dụng trước đó, không phải một tập xác nhận mới.

## Cấu hình và dữ liệu

- 58.200 frame TRAIN, 181.176 crop; 14.600 frame VAL.
- Cả G và Q5 khởi tạo ImageNet gốc, seed0; không dùng checkpoint đã học của G hoặc Q5 screening để warm-start.
- Dùng lại detector seed0 đã hoàn thành và proposal VAL có hash xác minh; không train lại RetinaNet.
- Batch256, 8 worker spawn, worker mới mỗi epoch, FP32, AMP/TF32 tắt.
- AdamW lr0.0001, weight_decay0.0001, cosine12epoch ngay từ đầu.
- Crop224, margin5%; augmentation theo seed/epoch/index như bộ kit.
- Chấm toàn VAL ởepoch4/8/12, batch inference64, NMS0.5, threshold0.001, cap300; không cắt proposal để tăng điểm/tốc độ.
- Chọn mỗi nhánh theo VAL macro AP50 cao nhất trong3mốc; bằng nhau chọnepoch sớm hơn. Báo cảweighted vàper-class.
- Không dừng sớm dựa trên điểm; cảhai nhánh đều học đủ12epoch.
- Không giới hạn giờ ứng dụng. Vẫn dừng khi lỗi hoặc dung lượng trống dưới8GiB.
- Instance không tựStop/Destroy.

## Lưu trữ và tái lập

Run: /workspace/runs/gq5_long12_20261027
Log: /workspace/runs/gq5_long12_20261027.console.log
Script: /workspace/gq5_long12_20261027/run_gq5_12.py
tmux: q5long12

Lưu last.pt nguyên model,optimizer,scheduler,RNG sau mỗi epoch trướcVAL. Giữ epoch04.pt,epoch08.pt,epoch12.pt bằng hardlink immutable qua cơ chế atomic replace. Nếu bị ngắt trongVAL, đọc checkpoint đã lưu và chấm lại VAL trước khi train tiếp. Giữ nguyên RNG trước và sauVAL.

study.json khóa code nguồn,plan,hashTrain/VAL,ImageNet,detector/proposals,runtime. Tách riêng tất cả kết quả khỏi screening2epoch và research_v1_fp32.

Kết thúc tạo comparison_result.json, comparison_val.csv và selection.json cóhashcheckpoint của mỗi nhánh. selection này là của nghiên cứu riêng, không tự mở khóa TEST của bộkit cũ.

## Kiểm tra trước chạy

Đã kiểm tra logic mốcVAL4/8/12, chọnmacro/tieearliest, restorecosine12, checkpointmilestone không bị thay khi lưu epoch mới, và phát hiện chỉnh sửa outputVAL bằnghash. Bảnpreprocessing được giữ từ screening đã chạy thành công.

Ước tính7–9giờ dựa trên6phút train/epoch và42–46phút inferenceVAL trước đó. Thời gian thực tế phụ thuộc máy vàdata. Đây không phải timeout/hứa thời điểm hoàn thành.
