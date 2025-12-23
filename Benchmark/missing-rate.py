import numpy as np
import matplotlib.pyplot as plt

# --- Cấu hình Font ---
plt.rcParams.update({
    'font.size': 20,
    'axes.titlesize': 20,
    'axes.labelsize': 20,
    'xtick.labelsize': 20,
    'ytick.labelsize': 20,
    'legend.fontsize': 20,
    'figure.titlesize': 20
}) 

# --- Dữ liệu (Giữ nguyên) ---
missing_ratios = np.arange(0.0, 0.8, 0.1)
models = ['MoMKE', 'MPLMM', 'GCNet', 'MMIN']
datasets = ['(a) MOSI', '(b) MOSEI', '(c) IEMOCAP']

np.random.seed(42)
data = {}
for ds in datasets:
    data[ds] = {}
    for model in models:
        data[ds][model] = np.random.uniform(75, 85) - (missing_ratios * 20)

colors = ['#FFA500', '#00CC00', "#003B82", '#FF0000']
markers = ['s', '^', 'v', 'o']
linestyles = ['-', '-', '-', '-']

# --- VẼ BIỂU ĐỒ ---
# figsize=(20, 6): Chiều ngang rộng, chiều cao vừa phải
fig, axes = plt.subplots(1, 3, figsize=(20, 6), sharey=True)

# Loop vẽ
for col_idx, dataset_name in enumerate(datasets):
    ax = axes[col_idx]
    for i, model_name in enumerate(models):
        ax.plot(
            missing_ratios, 
            data[dataset_name][model_name], 
            label=model_name, # Gắn label để lấy handle sau
            color=colors[i],
            marker=markers[i],
            linewidth=2,
            clip_on=False
        )

    ax.set_title(dataset_name, y=1.02)
    ax.set_xlabel("Missing Ratio")
    if col_idx == 0: ax.set_ylabel("F1 Score")
    ax.grid(True, linestyle='--', alpha=0.5)
    ax.set_xlim(0.0, 0.7)

# --- XỬ LÝ LEGEND CHUẨN ---

# 1. Lấy handles/labels từ ax[0]
handles, labels = axes[0].get_legend_handles_labels()

# 2. Tạo Legend gắn vào Figure (để căn giữa 3 hình)
# loc='upper center': Điểm neo là giữa trên của hộp legend
# bbox_to_anchor=(0.5, 1.0): Đặt điểm neo đó vào giữa (0.5) và mép trên cùng (1.0) của khung hình
legend = fig.legend(
    handles, labels,
    loc='upper center',      
    bbox_to_anchor=(0.5, 1.0), 
    ncol=4, 
    frameon=True
)

# 3. Tự động căn chỉnh khoảng cách, CHỪA LẠI chỗ cho legend
# tight_layout sẽ tính toán tất cả các thành phần trong hình. 
# Tuy nhiên, fig.legend thường "trôi nổi", tight_layout đôi khi bỏ qua nó.
# Nhưng bằng cách set rect, ta ép tight_layout chỉ dùng 90% phía dưới.
plt.tight_layout(rect=[0, 0, 1, 0.9]) 

plt.savefig("final_chart.svg", bbox_inches='tight')
plt.show()