# Pilot dự đoán có quyền từ chối theo vai — 29/10/2026

Đăng ký kế hoạch trước khi tính kết quả của phương pháp mới. Pilot giới hạn một cấu hình ứng viên và bốn đối chứng, không train lại mô hình hoặc đọc TEST.

## Nguồn và ranh giới

Dùng logits GT crop của G/Q5 seed0 epoch24 đã lưu. Chọn đúng mẫu4.096 crop ngẫu nhiên ban đầu, bỏ phần bổ sung lớp hiếm của lần chẩn đoán cũ. 1.398 crop thuộc52 clip fit nhiệt độ được dùng để đặt ngưỡng coverage; 2.698 crop thuộc94 clip khác dùng đánh giá phát triển. Hai nhiệt độ đãfit được giữ nguyên, áp dụng giốngnhau cho mọiphươngpháp của cùngmodel. VAL đã được xem ởcác nghiên cứu trước, nên không gọi đây là xác nhận độc lập.

Đối chiếu hash logits, danh sách indices, lớp, manifest và checkpoint. Xuất cả support mỗi vai/trạng thái và từngclip. Không suy rộng sang detector, localization, crop proposal hoặc dữliệuViệtNam.

## Phương pháp

- crop_confidence: dự đoán cấu hình top1; dùng max xác suất cấu hình để trả lời/từchối toàncrop.
- native_role_confidence: dự đoán trạng thái absent/helmet/nohelmet của từngvai từ10logits gốc; confidence làmax xác suất3trạngthái.
- role_marginal: cộng xác suất36cấu hình theo trạngthái mỗi vai; lấy trạngthái cóxácsuất cao nhất và confidence tương ứng.
- topclass_role_marginal: giữ trạngthái từcấu hình top1, xếp độtin cậy theo xácsuất biên của trạngthái đó. Đối chứng này tách thayđổi confidence khỏi thayđổi nhãn.
- set_consensus: sắpxếp cấu hình theo xácsuất, giữ prefix đến khi đủ mass. Chỉ trả lời vai màcác cấu hình trong prefix thốngnhất; còn lại unknown=-1. Confidence mộtvai làtổng mass prefix trước cấu hình đầu tiên bấtđồng vềvai đó. Mọikếtluận đãtrảlời đều mởrộng được thànhcấu hình top1, nhưng không bảođảm làđúng.

Đây làứngviên hậu xửlý kếthừa ýtưởng selective prediction/prediction sets; khôngphải thuậttoánnovel đãxácminh. Không đặt tênconformal hoặc tuyênbố coverage lýthuyết: crop/video phụthuộc vàVAL đãthamgia pháttriển.

## So sánh và metric cố định

Mứccoverage chính70% mỗi vai; phụ50%,90%; risk-coverage ở10,20,...100%.

1. Matched-coverage diagnostic: trên evaluation, xếp confidence khôngdùngnhãn rồi giữ đúng floor(coverage*N) của mỗi vai. Dùng cùnghash sample để pháhòa. Đây làđường risk-coverage hồi cứu, khôngphải ngưỡng triểnkhai đãfit.
2. Ngưỡng triểnkhai: lấyquantile chỉtrên calibration để nhắmmục tiêu coverage; áp sangevaluation vàbáo coverage thựcđạt. Không dùngGT evaluation đểchọn ngưỡng. Coverage mới khôngđược bảođảm bằngtarget.

Risk chính: số lỗi trạngthái / số câu trảlời. Ngoài ra bắtbuộc báo risk trên cácvai GTpresent, mỗi role/state, tỷlệ trảlời từngrole/state, false nohelmet trênGT cómũ, false nohelmet khiGT absent, missed nohelmet và unknown. Unknown tuyệtđối không tính làabsent hoặc đúng.

So ứngviên với cảbốn đốichứng cho G vàQ5; không chỉchọn baseline yếu. Bootstrap cặp1.000 lần theo clip (seed20261029) với mask acceptance đãcốđịnh để ước tính bấtđịnh cóđiều kiện; khôngcoi crop liềnkề làđộclập. CI khônggồm độbiếnthiên train, temperature fit hoặc chọnngưỡng.

## Quyết định sau pilot

Chỉcân nhắc mởrộng ứngviên khi risk ởcoverage70% giảm ít nhất1điểm phầntrăm so vớimọi baseline trêncùngmodel, risk ởGTpresent khôngtăng quá1điểm, vàbootstrap95% upper của delta risk so baseline mạnhnhất<0. Đây làgate sànglọc nội bộ, khôngphải tiêu chuẩn hội nghị.

Nếu role_marginal hoặc baseline khác tốtngang/hơn, ghi nhận đầyđủ; không thêm hyperparameter rồi chấm lại đểđạtgate. Khôngcó guarantee Q5 thắngG. Kếtquả mộtseed/cropsubset chỉquyếtđịnh hướngnghiêncứutiếp.

## Tài liệu nền

- SelectiveNet, ICML2019: https://proceedings.mlr.press/v97/geifman19a.html
- Knowing what You Know, JMLR2021: https://www.jmlr.org/papers/v22/20-753.html

Cácnguồn trên đãcó selective prediction vàprediction sets. Đónggóp còn phảiđượcphânbiệt qua phân tíchtừngvai/cấutrúc, đốichứng vàđánhgiá ngoài.
