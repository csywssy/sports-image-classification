import torch
import torch.nn as nn
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import os
import numpy as np

# ======================
# 1. 配置
# ======================
data_dir = "./data"
model_path = "checkpoints/resnet18_final_hard.pth"
batch_size = 32
num_classes = 7

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# ======================
# 2. 数据加载
# ======================
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

val_dataset = datasets.ImageFolder(
    os.path.join(data_dir, "val"),
    transform=transform
)

val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

class_names = val_dataset.classes
print("Classes:", class_names)

# ======================
# 3. 加载模型
# ======================
model = models.resnet18(weights=None)

# ⚠️ 必须和训练一致（有Dropout）
model.fc = nn.Sequential(
    nn.Dropout(0.5),
    nn.Linear(model.fc.in_features, num_classes)
)

model.load_state_dict(torch.load(model_path, map_location=device))
model = model.to(device)
model.eval()

print("✅ 模型加载成功")

# ======================
# 4. 预测
# ======================
all_preds = []
all_labels = []

with torch.no_grad():
    for images, labels in val_loader:
        images = images.to(device)

        outputs = model(images)
        _, preds = torch.max(outputs, 1)

        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.numpy())

# ======================
# 5. 分类报告
# ======================
report = classification_report(all_labels, all_preds, target_names=class_names)

print("\n📊 Classification Report:\n")
print(report)

# 👉 保存txt（论文直接用）
os.makedirs("results", exist_ok=True)

with open("results/report_final_hard.txt", "w", encoding="utf-8") as f:
    f.write(report)

print("✅ 分类报告已保存：results/report_final_hard.txt")

# ======================
# 6. 混淆矩阵
# ======================
cm = confusion_matrix(all_labels, all_preds)

plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=class_names,
            yticklabels=class_names)

plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Confusion Matrix (Final Hard Model)")

plt.savefig("results/confusion_matrix_final_hard.png")
plt.show()

print("📈 混淆矩阵已保存：results/confusion_matrix_final_hard.png")