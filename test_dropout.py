# ======================
# test_dropout.py
# ======================

import os
import torch
import torch.nn as nn
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# ======================
# 配置
# ======================
data_dir = "./data/val"  # 验证集路径
batch_size = 32
num_classes = 7
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class_names = ['Badminton', 'Cricket', 'Karate', 'Soccer', 'Swimming', 'Tennis', 'Wrestling']

# ======================
# 数据加载
# ======================
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

val_dataset = datasets.ImageFolder(data_dir, transform)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

# ======================
# 模型加载
# ======================
model = models.resnet18(weights=None)
in_features = model.fc.in_features
model.fc = nn.Sequential(
    nn.Dropout(p=0.5),
    nn.Linear(in_features, num_classes)
)

model.load_state_dict(torch.load("checkpoints/resnet18_dropout.pth", map_location=device))
model = model.to(device)
model.eval()

print("✅ 模型加载成功")
print("Classes:", val_dataset.classes)

# ======================
# 测试循环
# ======================
all_labels = []
all_preds = []

with torch.no_grad():
    for images, labels in val_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)

        all_labels.extend(labels.cpu().numpy())
        all_preds.extend(predicted.cpu().numpy())

# ======================
# 分类报告
# ======================
report = classification_report(all_labels, all_preds, target_names=class_names)
print("\n📊 Classification Report:\n")
print(report)
# 保存分类报告到文件
with open("classification_report_dropout.txt", "w", encoding="utf-8") as f:
    f.write(report)

print("📄 分类报告已保存为 classification_report_dropout.txt")

# ======================
# 混淆矩阵
# ======================
cm = confusion_matrix(all_labels, all_preds)
plt.figure(figsize=(8,6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.tight_layout()
plt.savefig("confusion_matrix_dropout.png")
plt.show()
print("\n📈 混淆矩阵已保存为 confusion_matrix_dropout.png")