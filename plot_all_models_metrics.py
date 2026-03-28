import matplotlib.pyplot as plt
import numpy as np

# ===============================
# 类别名称
# ===============================
classes = ['Badminton', 'Cricket', 'Karate', 'Soccer', 'Swimming', 'Tennis', 'Wrestling']
x = np.arange(len(classes))
width = 0.25  # 每组柱子的宽度

# ===============================
# 模型指标数据
# Precision 和 Recall
# 数据示例，可替换为你实际每个模型的测试结果
# ===============================

# 1. Baseline
precision_baseline = [0.97, 0.96, 0.97, 0.95, 0.99, 0.96, 0.98]
recall_baseline    = [0.97, 0.96, 0.91, 0.96, 0.96, 0.98, 0.98]

# 2. Augmentation
precision_aug = [0.96, 0.98, 0.93, 0.96, 0.97, 0.95, 0.97]
recall_aug    = [0.97, 0.95, 0.97, 0.95, 0.99, 0.95, 0.97]

# 3. Dropout
precision_dropout = [0.96, 0.98, 0.94, 0.97, 0.98, 0.96, 0.98]
recall_dropout    = [0.98, 0.96, 0.95, 0.96, 0.98, 0.96, 0.98]

# ===============================
# 绘图
# ===============================
fig, ax = plt.subplots(2, 1, figsize=(12,10))

# -------- Precision --------
ax[0].bar(x - width, precision_baseline, width, label='Baseline', color='#1f77b4')
ax[0].bar(x, precision_aug, width, label='Augmentation', color='#ff7f0e')
ax[0].bar(x + width, precision_dropout, width, label='Dropout', color='#2ca02c')

ax[0].set_ylabel('Precision')
ax[0].set_xticks(x)
ax[0].set_xticklabels(classes)
ax[0].set_ylim(0,1.05)
ax[0].set_title('Per-Class Precision Comparison')
ax[0].legend()

# -------- Recall --------
ax[1].bar(x - width, recall_baseline, width, label='Baseline', color='#1f77b4')
ax[1].bar(x, recall_aug, width, label='Augmentation', color='#ff7f0e')
ax[1].bar(x + width, recall_dropout, width, label='Dropout', color='#2ca02c')

ax[1].set_ylabel('Recall')
ax[1].set_xticks(x)
ax[1].set_xticklabels(classes)
ax[1].set_ylim(0,1.05)
ax[1].set_title('Per-Class Recall Comparison')
ax[1].legend()

plt.tight_layout()
plt.savefig("all_models_class_metrics.png", dpi=300)
plt.show()

print("✅ 三模型类别性能对比图已生成: all_models_class_metrics.png")