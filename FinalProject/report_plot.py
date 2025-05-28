import matplotlib.pyplot as plt
import numpy as np

# 類別與統計值
labels = ["CBB", "CBSD", "CGM", "CMD", "Healthy"]
precision = [0.6590, 0.8395, 0.8315, 0.9587, 0.7863]
recall = [0.6881, 0.8337, 0.8189, 0.9732, 0.7268]
f1_score = [0.6733, 0.8366, 0.8252, 0.9659, 0.7554]
support = [1087, 2189, 2386, 13158, 2577]

x = np.arange(len(labels))
width = 0.25

fig, ax = plt.subplots(figsize=(10, 6))

# 柱狀圖
bar1 = ax.bar(x - width, precision, width, label='Precision', color='skyblue')
bar2 = ax.bar(x, recall, width, label='Recall', color='lightgreen')
bar3 = ax.bar(x + width, f1_score, width, label='F1-score', color='salmon')

# 左側 y 軸設定
ax.set_ylabel('Score')
ax.set_ylim(0.6, 1.05)
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.set_title('Classification Report with Support')

# 黑線底部起始
ax.spines['bottom'].set_visible(False)
ax.axhline(0.6, color='black', linewidth=1.5)
ax.set_yticks(np.linspace(0.6, 1.0, 5))
ax.legend(loc='upper left')

# 加數值標籤
def add_labels(bars):
    for bar in bars:
        height = bar.get_height()
        ax.annotate(f'{height:.2f}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom')

add_labels(bar1)
add_labels(bar2)
add_labels(bar3)

# ➕ 右側 y 軸加上 support 折線圖
ax2 = ax.twinx()
ax2.plot(x, support, color='darkorange', marker='o', linewidth=2, label='Support')
ax2.set_ylabel('Support Count')
ax2.tick_params(axis='y', labelcolor='darkorange')
ax2.set_ylim(0, max(support) * 1.1)  # support 範圍
ax2.legend(loc='upper right')

plt.tight_layout()
plt.savefig("classification_report_with_support.png", dpi=300)
plt.show()
