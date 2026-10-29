# Ensemble3seed + calibration/logit adjustment — 29/10/2026

Theo yêu cầu người dùng: ưu tiên phân tích lớp và tận dụng checkpoint có sẵn, sau đó mới cân nhắc retrain long-tail. Không thay đổi các thí nghiệm đã chốt hoặc dùng TEST.

## Đầu vào cố định

- G vàQ5 gốc, mỗi model dùngcheckpoint epoch24 của seed0/1/2 đã hoàn tất. Giữ lịch12+12 trướcđó; không chọn seed/epoch theo kếtquả ensemble.
- Cùng frozen RetinaNet seed0 proposals và14600frame VAL. TRAIN chỉđọcnhãn để tínhprior; khôngcậpnhậtweight.
- Native logits cho toànproposals chưađược cache. Chạy inference mộtlần/architecture với3classifier, giữpreprocessing224px/margin0.05/batch64. Backbone vàdetector khôngtrain.
- Ensemble là trungbình xácsuất36cấu hình của3seed, trướcscorethreshold/NMS. Không lấytrungbình AP vàkhông ghép JSON hậuNMS đểgọi đó làensemblelogit.
- Cache trungbình probabilities float64, indexproposal khớp exactbox dedup. Checkhashcheckpoint/source/manifest/proposal/cache; ghi checkpoint tiếnđộcache mỗi100frame đểresume.

## Calibration và lựa chọn

Giữsplit clipsha256 calibration_v1:clip_id mod3==0:52clip/5200frame calibration,94clip/9400frame evaluation. Đây vẫn làVALdevelopment đãxem, khôngTESTđộclập.

1. Dùngmẫu4096GTcrop đãđịnhtrước, chỉlấy1398crop thuộcclipcalibration đểfitT củaensemble. Fit NLLcategorical, T∈[0.5,10],64vònggolden-section. Nhiệtđộáp lênlog(mean probability36), khác vớiTnative10 trướcđó.
2. Logit adjustment theoTRAINpriorπc=(Nc+1)/(N+36), scores=log(qc)/T−tau·logπc. τ cốđịnhtrong{0,0.25,0.5,1}; khôngmởrộnggrid.
3. Chọnτ trênAP50 của5200framecalibration: macrocaonhất trongnhómweighted khônggiảmquá0.5điểm soτ0; tie chọnτnhỏ.
4. KhóaT/τ thànhfile trước khiđánhgiá9400frame. Báo ensemblegốc,ensemble+T,ensemble+T+τcho cảG/Q5; cùngđiềukiện.
5. Giữthreshold0.001,NMS0.5,cap300 vàscorer nội bộ. BáoAPtừnglớp/support; calibration vàevaluation cósốlớpcóGTkhácnhau.

Không gọi posthoc adjustment lànovel; kếthừa Menon etal.,ICLR2021 https://arxiv.org/abs/2007.07314 . Tscale kếthừaGuo etal.,ICML2017 https://proceedings.mlr.press/v70/guo17a.html .

## Tiêu chí bước long-tail

Nếu ensembleQ5 sauposthoc vẫn khônghơnensembleG cùngxửlý, hoặcweighted kémquá0.5điểm, ưu tiên mộtrecipe long-tail duy nhất trêncảG/Q5 và3seed. Kếtquả source/variance cầnđọctrước khi khởichạy: không lặplại retrain vôhạn hoặc bỏseed thua. Giaiđoạn này chỉđềxuất decision.json, khôngtựtrain thêm.

Đốiứngviênlong-tail dựkiến: simplified asymmetric focusing cho presence (gamma+=0,gamma−=1;khôngprobabilityshift), conditionalhelmetBCE giữGTmask; soG/Q5 cùngrecipe3seed. Đây làadaptation khôngnovel. Chưa áp balanced sampler đồngthời đểtránh lẫn tácdụng.

## Tài nguyên và thứ tự

Run sau khi GQ5residual vàcapacitycontrol kếtthúc. CùngGPU executionlock. Inferencekhôngcậpnhậttrọngsố; dùngGPU choforward,CPU choAP/NMS/posthoc. ChỉđọcTRAIN/VAL, khôngTEST.

Paths:/workspace/ensemble3_20261029 ; /workspace/runs/ensemble3_20261029.
Log:/workspace/runs/ensemble3_20261029.console.log.
KhôngtựStop/Destroy instance; khôngxóa checkpointcũ.

## Kiểm tra số học trước chạy

Golden-section được chặn T chính xác vào[0.5,10] để tránh exp(log10)=10.000000000000002 do làm tròn. Đây là sửa triểnkhai trước inference, khôngthay grid hoặc đọcmetric evaluation.
