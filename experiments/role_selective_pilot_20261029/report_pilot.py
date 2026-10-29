"""Measured reports/figures for the completed fixed selective-prediction pilot."""
import os,json,hashlib
from pathlib import Path
ROOT=Path("/local/helmet-research")
OUT=ROOT/"output/role_selective_pilot_20261029"
os.environ["MPLCONFIGDIR"]=str(ROOT/".codex-build/q5_reports_20261028/mpl-cache")
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

NAMES={"crop_confidence":"Từ chối toàn crop",
       "native_role_confidence":"Confidence native từng vai",
       "role_marginal":"Xác suất biên từng vai",
       "topclass_role_marginal":"Top1 + xác suất biên",
       "set_consensus":"Đồng thuận cấu hình (ứng viên)"}
SHORT={"crop_confidence":"Whole-crop confidence","native_role_confidence":"Native role confidence",
       "role_marginal":"Role marginal","topclass_role_marginal":"Top1 + marginal",
       "set_consensus":"Set consensus (candidate)"}

def pct(x):
    return "—" if x is None else f"{100*x:.3f}"

def main():
    d=json.loads((OUT/"result.json").read_text())
    v=json.loads((OUT/"verification.json").read_text())
    assert v["result_sha256"]==hashlib.sha256((OUT/"result.json").read_bytes()).hexdigest()
    assert v["decisions_sha256"]==hashlib.sha256((OUT/"decisions.npz").read_bytes()).hexdigest()
    assert d["status"]=="completed" and not d["expand_candidate"]
    styles={"crop_confidence":("#777777","--"),"native_role_confidence":("#b58b39",":"),
            "role_marginal":("#176b87","-"),"topclass_role_marginal":("#70a2a0",":"),
            "set_consensus":("#c45b43","-")}
    plt.rcParams.update({"font.family":"DejaVu Sans","font.size":10,"axes.spines.top":False,
                         "axes.spines.right":False,"pdf.fonttype":42,"svg.fonttype":"none"})
    fig,axes=plt.subplots(1,2,figsize=(11,4.5),sharey=True)
    for ax,arm in zip(axes,("G","Q5")):
        for method,(color,ls) in styles.items():
            points=d["models"][arm]["methods"][method]["risk_coverage_curve"]
            ax.plot([p["coverage"]*100 for p in points],[p["risk"]*100 for p in points],
                    color=color,ls=ls,marker="o" if method=="set_consensus" else None,markersize=3,
                    label=SHORT[method],lw=1.6)
        ax.axvline(70,color="#dddddd",zorder=0)
        ax.set(title=arm,xlabel="Answered decisions per role (%)",xlim=(8,102))
        ax.grid(alpha=.15)
    axes[0].set_ylabel("Error among answered role states (%)")
    handles,labels=axes[0].get_legend_handles_labels()
    fig.legend(handles,labels,loc="lower center",ncol=3,fontsize=8,frameon=False,bbox_to_anchor=(.5,.035))
    fig.suptitle("Selective prediction: candidate vs four fixed controls",fontsize=12)
    fig.text(.5,.005,"2,698 GT crops from 93 VAL clips · Seed0, epoch24 · Same temperatures · Lower risk is better\n"
             "Retrospective equal coverage; includes easy absent-role decisions. Not TEST or detection AP.",ha="center",fontsize=8)
    fig.tight_layout(rect=(0,.20,1,.93))
    for ext in ("png","pdf","svg"):fig.savefig(OUT/f"risk_coverage.{ext}",dpi=180,bbox_inches="tight")
    plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(11,4.5),sharey=True)
    roles=list(d["evaluation_role_state_support"])
    xx=np.arange(5)
    for ax,arm in zip(axes,("G","Q5")):
        for k,method in enumerate(("crop_confidence","role_marginal","set_consensus")):
            rows=d["models"][arm]["methods"][method]["matched"]["0.7"]["roles"]
            coverage=[]
            for row in rows:
                gt=sum(x["GT_support"] for x in row["states"] if x["state"]!=0)
                accepted=sum(x["answered"] for x in row["states"] if x["state"]!=0)
                coverage.append(100*accepted/gt if gt else 0.)
            ax.bar(xx+(k-1)*.23,coverage,.23,label=SHORT[method],color=styles[method][0])
        ax.set_xticks(xx,roles);ax.set(title=arm,ylim=(0,80),xlabel="Role")
        ax.grid(axis="y",alpha=.15);ax.set_axisbelow(True)
    axes[0].set_ylabel("Answer rate when a rider is actually present (%)")
    handles,labels=axes[0].get_legend_handles_labels()
    fig.legend(handles,labels,loc="lower center",ncol=3,fontsize=9,frameon=False,bbox_to_anchor=(.5,.04))
    fig.suptitle("70% overall role coverage can hide rejection of rare riders",fontsize=12)
    counts=[sum(d["evaluation_role_state_support"][r][1:]) for r in roles]
    fig.text(.5,.005,"GT-present supports: "+", ".join(f"{r}={n}" for r,n in zip(roles,counts))+
             "\nAll eight P3-present cases are rejected at this operating point. Small-support results are uncertain.",
             ha="center",fontsize=8)
    fig.tight_layout(rect=(0,.18,1,.92))
    for ext in ("png","pdf","svg"):fig.savefig(OUT/f"present_role_coverage.{ext}",dpi=180,bbox_inches="tight")
    plt.close(fig)
    lines=["# Kết quả pilot dự đoán có quyền từ chối theo vai — 29/10/2026","",
      "**Ứng viên đồng thuận cấu hình chưa vượt đối chứng xác suất biên từng vai; không đạt gate mở rộng đã đặt trước.** "
      "Kết quả được báo đầy đủ, không đổi ngưỡng hoặc chọn subset sau khi thấy điểm.","",
      "## Phạm vi thực nghiệm","",
      "- G và Q5, seed0, checkpoint epoch24; không cập nhật trọng số hoặc sử dụng GPU cho pilot này.",
      "- Dùng4.096 crop ngẫu nhiên từ logits đã lưu:1.398 crop/52clip cho đặt ngưỡng;2.698 crop/93clip cho đánh giá. "
      "Phần VAL ngoài calibration toàn bộ có94clip, nhưng mẫu crop ngẫu nhiên này chỉ phủ93clip. "
      "Số94clip trong ghi chú kế hoạch ban đầu là dự kiến từ population đầy đủ; không thay đổi cách lấy mẫu để bù clip thiếu.",
      "- Cùng nhiệt độ đã fit trước cho mọi policy trong cùng model: G2.9041921868; Q53.2036922912.",
      "- Fit/evaluation tách clip, nhưng VAL đã được xem ở các thí nghiệm trước. Đây là development, không phải TEST độc lập.",
      "- Chỉ crop GT. Chưa đo lỗi detector, tracking, background, proposal crops hoặc dữ liệu Việt Nam.",
      "- Unknown=-1 khác absent=0. Không tính unknown là dự đoán đúng.","",
      "## So sánh chính: cùng mức trả lời gần70% ở từng vai","",
      "Mỗi policy trả lời đúng1.888/2.698 trường hợp của từng vai (69,9778%), tổng9.440 quyết định. "
      "Xếp hạng bằng confidence và phá hòa bằng hash định trước, không dùng nhãn để chọn. "
      "Đây là đánh giá risk-coverage hồi cứu; phần ngưỡng triển khai được báo riêng.","",
      "| Mô hình | Policy | Lỗi /9.440 câu trả lời | Risk (%) ↓ | Risk khi GT có người (%) ↓ | Coverage khi GT có người (%) |",
      "|---|---|---:|---:|---:|---:|"]
    for arm,mod in d["models"].items():
        for method,entry in mod["methods"].items():
            m=entry["matched"]["0.7"]
            lines.append(f"| {arm} | {NAMES[method]} | {m['errors']} | {pct(m['risk'])} | "
                         f"{pct(m['GT_present_risk'])} | {pct(m['GT_present_coverage'])} |")
    lines+=["","Ghi chú: risk là lỗi trạng thái trong các câu được trả lời; thấp hơn tốt hơn. "
            "Risk chung chịu ảnh hưởng lớn của các vai absent. Không quy đổi các số này thành accuracy toàn hệ thống hay AP50.",
            "","## Bất định theo clip","",
            "Delta = risk ứng viên − risk đối chứng; âm là ứng viên tốt hơn. Bootstrap cặp1.000 lần theo93clip, giữ acceptance mask đã xác định. "
            "Khoảng tin cậy chỉ có điều kiện trên mẫu/checkpoint/ngưỡng, không bao gồm biến thiên train seed hay fit nhiệt độ.","",
            "| Mô hình | Đối chứng | Delta risk (đpt) | Khoảng percentile95% (đpt) |",
            "|---|---|---:|---|"]
    for arm,mod in d["models"].items():
        for method in ("crop_confidence","role_marginal"):
            r=mod["primary_comparisons"][method];lo,hi=r["percentile95_pp"]
            lines.append(f"| {arm} | {NAMES[method]} | {r['delta_risk_pp']:+.4f} | [{lo:+.4f}; {hi:+.4f}] |")
    lines+=["","Ứng viên giảm một ít lỗi so với từ chối toàn crop, nhưng không thắng đối chứng mạnh hơn. "
            "VớiQ5, khoảng delta so với xác suất biên đi qua0; không có bằng chứng ưu thế. "
            "Gate1điểm phần trăm đã được đặt trước, không phải tiêu chuẩn nhận bài; cả hai model đều không đạt.",
            "","## Khi dùng ngưỡng chỉ đặt trên calibration","",
            "Mỗi ngưỡng được đặt bằng confidence của calibration để nhắm70% câu trả lời; "
            "coverage thực tế trên evaluation có thể khác. Không so risk giữa hàng khác coverage như bằng chứng ưu thế.","",
            "| Mô hình | Policy | Coverage thực (%) | Risk (%) |","|---|---|---:|---:|"]
    for arm,mod in d["models"].items():
        for method in ("crop_confidence","role_marginal","set_consensus"):
            r=mod["methods"][method]["calibration_thresholds"]["0.7"]["evaluation"]
            lines.append(f"| {arm} | {NAMES[method]} | {pct(r['coverage'])} | {pct(r['risk'])} |")
    lines+=["","## Vai hiếm: phải báo việc từ chối và bỏ sót","",
            "| Vai | GT absent | GT có mũ | GT không mũ | G ứng viên: coverage khi có người (%) | Q5 ứng viên: coverage khi có người (%) |",
            "|---|---:|---:|---:|---:|---:|"]
    for i,role in enumerate(roles):
        support=d["evaluation_role_state_support"][role];cov=[]
        for arm in ("G","Q5"):
            r=d["models"][arm]["methods"]["set_consensus"]["matched"]["0.7"]["roles"][i]
            present=sum(s["GT_support"] for s in r["states"] if s["state"]!=0)
            accepted=sum(s["answered"] for s in r["states"] if s["state"]!=0)
            cov.append(100*accepted/present if present else 0)
        lines.append(f"| {role} | {support[0]} | {support[1]} | {support[2]} | {cov[0]:.3f} | {cov[1]:.3f} |")
    lines+=["","Cả8 trường hợpP3 có người đều bị từ chối tại operating point70%; "
            "risk0% ởvaiP3 chỉ có nghĩa các câu trả lời còn lại là trường hợp absent dễ, không có nghĩa đã nhận dạng đượcP3. "
            "Chỉ có8 mẫu nên không thể kết luận tổng quát cho vai này.",
            "","## Cơ chế và những gì chưa chứng minh","",
            "Ứng viên sắp36 cấu hình theo xác suất. Confidence của mộtvai là tổng mass của prefix trước cấu hình đầu tiên bất đồng "
            "với trạng thái top1 của vai đó; chỉ trả lời khi confidence đủ. Vì các nhãn trả lời lấy từtop1, "
            "partial prediction luôn mở rộng được thành một cấu hình hợp lệ. Thuộc tính hợp lệ về logic không bảo đảm dự đoán đúng.",
            "","Đối chứng xác suất biên đơn giản cộng xác suất của tất cả cấu hình có cùng trạng thái mỗi vai. "
            "Trong thử nghiệm chính, nó tốt hơn ứng viên và không tạo partial prediction bất hợp lệ. "
            "Do đó hiện không có bằng chứng ứng viên thêm giá trị về chất lượng hoặc tính nhất quán tại operating point này.",
            "","Selective prediction và confidence sets có tiền lệ: "
            "[SelectiveNet](https://proceedings.mlr.press/v97/geifman19a.html), "
            "[Cauchois et al., JMLR2021](https://www.jmlr.org/papers/v22/20-753.html). "
            "Pilot này không triển khai SelectiveNet hay một thuật toán conformal có bảo đảm. "
            "Không được tuyên bố coverage lý thuyết trên các crop tương quan hoặc dưới domain shift.",
            "","## Quyết định","",
            "Chưa mở rộng ứng viên đồng thuận cấu hình thành đóng góp chính. Giữ nó như kết quả sàng lọc có đối chứng. "
            "Không train thêm mô hình dựa trên pilot này. Các lượttrain G/Q5 đã được lên lịch vẫn chạy riêng.",
            "","Phát hiện có ích cho nghiên cứu: selective risk trung bình thấp có thể che khuất việc bỏ trả lời ở vai hiếm. "
            "Đây là nhận xét từ mẫu development nhỏ; muốn thành đóng góp về đánh giá cần xác nhận trên nhiều seed, "
            "dữ liệu đầy đủ và TEST khóa, với các metric support/coverage/false alarm/miss đã định trước.",
            "","## Hình và bằng chứng","",
            f"![Risk coverage]({OUT}/risk_coverage.png)","",
            f"![Present-role coverage]({OUT}/present_role_coverage.png)","",
            "result.json chứa đủ50/70/90% cho5policy×2model, tất cảvai/trạng thái, threshold và CI; "
            "decisions.npz chứa predictions/masks/scores để kiểm tra lại. "
            "plan.json, sample_manifest.json và verification.json ghi protocol, split và SHA256. "
            "Số đo này không thay thế các bảngAP50 của đợttrain trước.",""]
    (OUT/"Bao_cao_pilot_tu_choi_theo_vai.md").write_text("\n".join(lines))
    print("Saved report and risk/rare-role coverage figures")

if __name__=="__main__":main()
