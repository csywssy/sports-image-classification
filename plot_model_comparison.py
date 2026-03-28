import matplotlib.pyplot as plt

# =============================
# 三个模型的Accuracy
# =============================

models = [
    "ResNet18",
    "ResNet18 + Augmentation",
    "ResNet18 + Dropout"
]

accuracy = [
    0.96,   # Baseline
    0.96,   # Augmented
    0.9708  # Dropout
]

# =============================
# 画柱状图
# =============================

plt.figure(figsize=(8,5))

bars = plt.bar(models, accuracy)

plt.ylabel("Accuracy")
plt.title("Model Performance Comparison")

# 在柱子上显示数值
for bar in bars:
    height = bar.get_height()
    plt.text(
        bar.get_x() + bar.get_width()/2,
        height,
        f"{height:.3f}",
        ha='center',
        va='bottom'
    )

plt.ylim(0.90,1.00)

plt.tight_layout()

# 保存图片
plt.savefig("model_comparison.png", dpi=300)

plt.show()

print("✅ 对比图已生成: model_comparison.png")