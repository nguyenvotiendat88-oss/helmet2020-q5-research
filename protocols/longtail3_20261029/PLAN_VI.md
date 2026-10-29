# Phép thử loss cho dữ liệu lệch lớp — 29/10/2026

Người dùng yêu cầu sau ensemble: nếu macro vẫn thấp thì thử balanced sampling hoặc loss long-tail với3seed, rồi tiếp tục yêu cầu tìm giải pháp. Ensemble+temperature hoàn tất: G33,3914/Q532,1445% macro trên9.400frame khôngfitT; cảhai chọntau0. Điều kiện xét bước tiếp đã xảyra. Chọn duy nhất simplified asymmetric focusing để tách tác dụng loss; không ghép sampler mới, kiếntrúc mới hoặc hiệu chỉnh mới vào run.

## Giả thuyết

Các vai P2/P3/P0 phần lớn absent, nên phần presence có nhiều negative dễ. Giảm trọng số negative dễ có thể giúp học vaihiếm. Đây chưa phải nguyên nhân đã chứngminh của gap G/Q5; cần đo falsepositive vàrecall, không chỉmacro.

Presence loss: positive BCE giữ nguyên; negative BCE nhân sigmoid(z)^1. Helmet loss vẫn chỉ tính khi cóngười theoGT. Mean trên batch×5 giữ nguyên. Gamma+=0,gamma−=1; khôngprobabilityshift, khôngdetach hệsốfocus, khôngclassprior hoặcsampler.

Phương pháp kếthừa simplified asymmetric loss trong Query2Label section3.2:
https://arxiv.org/html/2107.10834v1
Đây làadaptation, khôngnovelty đã xácnhận.

## Thiết kế cố định

- Model G vàQ5 gốc, seeds0/1/2;6fit, mỗi12epoch từImageNet với cùngseed khởi tạo nhưBCEbaseline.
- AdamW1e-4/weight_decay1e-4/cosine12, khôngrestart; batch256,8worker,FP32,AMP/TF32tắt; augmentation,crop224,margin0.05 khôngđổi.
- TRAIN58.200frame/181.176crop;VAL14.600frame/30lớpcóGT. Detector/proposal RetinaNet seed0 cốđịnh.
- Endpoint12 địnhtrước. Chỉ chạyVALởepoch12 để tiếtkiệm các lượt đánhgiá trunggian; các baseline đãlưu cóVAL4/8/12 nhưng các lượt đó bảo toànRNG vàkhôngđổi tham số/khôngearlystop.
- So sánh với chínhBCEbaseline ởepoch12 cùngseed; khôngso chéoepoch24 hoặcensemble.
- 72train-epoch và6VALfull; mỗiepoch~6phút là sốđo gần đây,VAL20–40phút. Ướctính8–12giờ toànrun, phụthuộc tốcđộ. Khôngdừng dựa trênđiểm vàkhôngnới sốseed đểtìm lượtthắng.
- Checkpoint model/optimizer/scheduler/RNG sau mỗiepoch;verifydata/source/init/order vàhashmetrics. Giữcheckpoint/recipecũ. Diskreserve8GiB, lock mộtGPUjob.

## Đọc kết quả

BáoG-ASL,Q5-ASL vàG/Q5-BCE theo3seed; APtừnglớp/support vàmean±SD. Nội bộ xétQ5 hơnG-ASL tấtcảseed,mean gain≥1điểm,weighted khôngkémquá0.5điểm; đâykhôngphảitiêuchuẩnhộinghị hoặcCI. Khôngđủ nếu gainchỉtừlớp1GT.

TấtcảVALđãđược xemnhiềulần, kếtquảlàdevelopment. TEST giữkhóa. Khôngtựchọnthêmgamma,sampler,epoch,kiếntrúc hoặcensemble mới. Nếu khôngcải thiện, báođúng vàđánhgiálại hướngbài, không hứaQ5 phải thắng.

## Theo dõi

Source:/workspace/longtail3_20261029
Run:/workspace/runs/longtail3_20261029
Log:/workspace/runs/longtail3_20261029.console.log
tmux:longtail3_20261029
Globalstatus:status.json; từngseed:seedN/progress.json.
Hoànthành6fit cósummary.json. KhôngautoStop/Destroy instance.
