"""Render measured calibration evidence, keeping sample and evaluation scope explicit."""
import os
from pathlib import Path
ROOT=Path("/local/helmet-research")
os.environ["MPLCONFIGDIR"]=str(ROOT/".codex-build/q5_reports_20261028/mpl-cache")
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT=ROOT/"output"
d=json.loads((OUT/"q5_calibration_reliability_20261028.json").read_text())
plt.rcParams.update({"font.family":"DejaVu Sans","font.size":10,"axes.spines.top":False,
                     "axes.spines.right":False,"pdf.fonttype":42,"svg.fonttype":"none"})
fig,axes=plt.subplots(1,2,figsize=(10.8,4.7),sharex=True,sharey=True)
for ax,arm in zip(axes,("G","Q5")):
    ax.plot([0,1],[0,1],ls="--",color="#999999",lw=1,label="Perfect calibration")
    for stage,color,label in (("before","#C45B43","Original"),("after","#176B87","Temperature-scaled")):
        m=d["arms"][arm][stage]
        bins=[b for b in m["top_label_reliability"]["bins"] if b["n"]]
        x=[b["mean_confidence"] for b in bins];y=[b["accuracy"] for b in bins]
        sizes=[14+4*(b["n"]**.5) for b in bins]
        ax.plot(x,y,color=color,lw=1,alpha=.6)
        ax.scatter(x,y,s=sizes,color=color,alpha=.8,edgecolors="white",linewidths=.6,
                   label=f"{label}: ECE={100*m['top_label_reliability']['ece']:.2f}%")
    ax.set(xlim=(0,1.02),ylim=(0,1.02),xlabel="Mean predicted confidence",title=arm)
    ax.grid(alpha=.17);ax.legend(loc="lower right",fontsize=8,frameon=True)
axes[0].set_ylabel("Empirical accuracy")
fig.suptitle("Calibration on validation clips excluded from temperature fitting",fontsize=12,y=.99)
fig.text(.5,.025,"2,698 GT crops · 15 fixed-width bins · Marker size reflects bin count\n"
         "Development validation data, not TEST; does not measure detector/background calibration.",
         ha="center",fontsize=8,color="#444444")
fig.tight_layout(rect=(0,.11,1,.94))
for ext in ("png","pdf","svg"):
    fig.savefig(OUT/f"HELMET_calibration_reliability_20261028.{ext}",dpi=180,bbox_inches="tight")
plt.close(fig)
lines=["# Bằng chứng hiệu chỉnh độ tin cậy G/Q5 — 28/10/2026","",
       "Phép đo dùng 2.698 crop GT được lấy từ mẫu ngẫu nhiên, thuộc các clip VAL không dùng fit nhiệt độ. "
       "Hai hệ số đã cố định từ 1.398 crop của các clip calibration. Đây là dữ liệu phát triển đã được xem trước đó, chưa phải TEST độc lập.","",
       "| Mô hình | Trạng thái | Accuracy (%) | Confidence (%) | ECE15 (%) | Class NLL | Brier |",
       "|---|---|---:|---:|---:|---:|---:|"]
for arm in ("G","Q5"):
    for stage,name in (("before","Gốc"),("after","Hiệu chỉnh")):
        m=d["arms"][arm][stage]
        lines.append(f"| {arm} | {name} | {100*m['accuracy']:.4f} | {100*m['confidence']:.4f} | "
                     f"{100*m['top_label_reliability']['ece']:.4f} | {m['constrained_class_NLL']:.6f} | "
                     f"{m['multiclass_Brier_sum_over_classes']:.6f} |")
lines+=["","ECE, NLL và Brier càng thấp càng tốt. Brier được cộng trên36 lớp rồi lấy trung bình mẫu. "
         "ECE dùng15 khoảng confidence đều, có báo số lượng mẫu mỗi khoảng trong JSON.",
         "","Ở Q5, accuracy thay đổi1 mẫu trong2.698 crop. Temperature áp dụng lên10 native logits trước constrained decoding, "
         "nên không mặc định bảo toàn argmax của36 lớp như temperature scaling trực tiếp trên categorical logits.",
         "","Các quyết định presence ởngưỡng0,5 không thay đổi sau khi chia logits cho một nhiệt độ dương. "
         "Mẫu này chỉ có8 crop cóP3; cả hai mô hình đều bỏ sót8 trường hợp ởngưỡng đó trước và sau calibration. "
         "Đây không phải bằng chứng đã khắc phục vai hiếm, không phải AP detection và không phải calibration của background/localization.",
         "","Kết luận được hỗ trợ: hiệu chỉnh nhiệt độ cải thiện độ tin cậy của dự đoán crop trên phần VAL không dùng fit hệ số. "
         "Chưa có kết luận Q5 vượt G hoặc phương pháp mới thắng baseline.",
         "","Nguồn: q5_calibration_reliability_20261028.json; script calibration_metrics.py đọc logits đã lưu, không cập nhật trọng số.",
         "",f"![Reliability diagram]({OUT}/HELMET_calibration_reliability_20261028.png)",""]
(OUT/"Bang_chung_calibration_20261028.md").write_text("\n".join(lines))
print("Saved measured table and PNG/PDF/SVG reliability figures")
