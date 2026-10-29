# Phép thử tiếp theo: GQ5 residual và đối chứng dung lượng

Ngày29/10/2026. Yêu cầu: tìm giải pháp sau khi Q5 gốc không vượt G ổn định qua3seed. Chỉ chạy một pilot hữu hạn trước khi mở rộng.

## Căn cứ chọn hướng

- Q5 gốc trung bình thấp hơn G0,5958 điểm macro và1,2266 điểm weighted ởepoch24, trên3seed.
- Đọc AP từnglớp cho thấy Q5 thua ở các lớp phổbiến qua cả3seed, nên vấnđề khôngchỉ là vaihiếm.
- Pilot conditional NLL tăng gần2điểm macro soQ5 gốc nhưng riêng lớp chỉ1GT góp khoảng2,963điểm. Đây làphân tích độnhạy, không xóa lớp khỏi metric chính.
- Backup đồngthuận cấu hình không vượt đối chứng selective prediction đơn giản; không train tiếp hướng đó trong lượt này.

Do đó dùng đường global củaG làm thành phần dự đoán trực tiếp, đểnhánhquery học phần bổsung. Không hứa môhình lai sẽtốt hơn; không gọi nó làQ5 gốc.

## Hai lượt fit định trước

1. GQ5_residual: logits=G_logits+tanh(alpha_role)*Q5_logits,5hệsốkhởi tạo0, mộtbackbone.
2. G_capacity392: global MLP512→392→10 đểkiểm soátdung lượng.

Head205.209 vs205.026 tham số, sai khác~0,089%. Cảhai từImageNet,seed0,4epoch đoạnđầu cosine12 nhưG/Q5 gốc; batch256,AdamW1e-4,weight_decay1e-4,FP32,8worker. GiữBCE10logit gốc,224px,margin5%,augmentation,proposal,threshold0.001,NMS0.5,cap300,scorer khôngđổi. Khôngtrộn conditionalNLL hoặc calibration vào phépso sánhkiếntrúc.

VALtoànbộ14.600frame ởepoch4. Giữendpoint4 cốđịnh; không chọn riêngcheckpoint thuậnlợi. So vớiG/Q5gốc epoch4 đãxácminhhash vàG_capacity392 vừatrain. Khóa data/source/initialization/sampleorder. Lưucheckpoint model/optimizer/scheduler/RNG sau mỗiepoch,giữmilestone; resume nếu ngắt.

Tiêuchí sànglọc nội bộ: residual macro hơn cảGgốc vàcapacitycontrol ít nhất1điểm,weighted khôngkém quá0,5điểm; đồngthời báo support từnglớp và đónggóp lớp≤5GT. Passchưa chứngminhổnđịnh; cần giữcấuhình và kiểmchứng seed1/2,TEST sau quyếtđịnhkhoa học. Nếu fail,báođúng; khôngtự mở hàngloạt cấu hình đểđạt điểm.

Tổng8train-epoch và2lượtVAL. Ướctính banđầu2–3giờ theo~6phút/epoch và20–40phút/VAL; cậpnhật bằngsốđo mới. Đây khôngphảigiớihạn cưỡngbức. Diskreserve8GiB; lỗi phải dừngcólog, khôngtựthử vôhạn.

KhôngtruycậpTEST,khôngtrainRetinaNet,khôngsửdụngViệtNam,chưa bảođảmnovelty. Global/spatial residual attention đãcótiềnlệ CSRA(ICCV2021): https://arxiv.org/abs/2108.02456 .

## Artifacts

Source:/workspace/gq5_residual_pilot_20261029
Run:/workspace/runs/gq5_residual_pilot_20261029
Log:/workspace/runs/gq5_residual_pilot_20261029.console.log
tmux:gq5residual20261029
status.json/progress.json: trạngthái vàtiếnđộ
comparison_result.json: chỉtạo khi cảhaiarm hoànthành

Kếtthúc không tựStop/Destroy GPU. Tấtcảcheckpoint/kếtquảcũ giữnguyên.
