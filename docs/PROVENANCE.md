# Nguồn và trạng thái nghiên cứu

Repo private theo yêu cầu chủ tài khoản. Không tuyên bố công trình đã được phản biện/chấp nhận, hoặc thuật toán mới chỉ vì có code mới. Khi nộp bài, nhóm cần kiểm tra trích dẫn, license từng nguồn và vai trò tác giả.

## Các phần kế thừa

- [torchvision ResNet/RetinaNet và các PyTorch layers](https://github.com/pytorch/vision).
- [Lin et al., positional encoding cho helmet classification, ICIP 2021](https://doi.org/10.1109/ICIP42928.2021.9506178).
- [Query2Label](https://arxiv.org/abs/2107.10834).
- [CSRA / Residual Attention](https://arxiv.org/abs/2108.02456).
- [Temperature scaling](https://proceedings.mlr.press/v70/guo17a.html).
- [Logit adjustment](https://arxiv.org/abs/2007.07314).
- [Semantic Loss](https://proceedings.mlr.press/v80/xu18h.html).
- [SelectiveNet](https://proceedings.mlr.press/v97/geifman19a.html).
- [Confidence sets](https://www.jmlr.org/papers/v22/20-753.html).

Đây là các nguồn liên quan, không có nghĩa repo tái hiện toàn bộ từng paper. Code G/Q5 core đã có trong training kit dự án; các thư mục experiments chứa runner, biến thể và phân tích riêng. Các thư viện ngoài được import từ dependencies, không được chép toàn bộ vào repo.

## License

Chưa gán license mở cho repo nội bộ này. Không tự gán MIT cho toàn bộ kit, dataset hoặc nguồn chưa được kiểm tra license. Không phân phối lại JPEG hoặc trọng số. Quyền của các bên cung cấp dataset và mã kế thừa vẫn được giữ.

## Dữ liệu và thông tin riêng

Không đưa token, khóa SSH, IP máy GPU, cấu hình bí mật, ảnh chụp màn hình hoặc log terminal thô lên repo. Danh sách clip và support là metadata nghiên cứu để giải thích split.

Không đưa điểm synthetic/smoke vào bảng AP thực. Các file lịch sử có thể nhắc kế hoạch hoặc trạng thái cũ; báo cáo tổng hợp luôn phân biệt kết quả hoàn tất và công việc đang chạy.
