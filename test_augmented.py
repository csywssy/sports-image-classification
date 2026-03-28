import torch
import torch.nn as nn
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
import os
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# =============================
# 1 基本配置
# =============================
data_dir = "./data"
batch_size = 32
num_classes = 7
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Using device:", device)

# =============================
# 2 数据加载
# =============================
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

test_dataset = datasets.ImageFolder(
    os.path.join(data_dir, "val"),
    transform=transform
)

test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

class_names = test_dataset.classes
print("Classes:", class_names)

# =============================
# 3 加载模型
# =============================
model = models.resnet18(weights=None)
model.fc = nn.Linear(model.fc.in_features, num_classes)

model.load_state_dict(torch.load("checkpoints/resnet18_augmented.pth", map_location=device))
model = model.to(device)
model.eval()

print("✅ 模型加载成功")

# =============================
# 4 模型测试
# =============================
all_preds = []
all_labels = []

with torch.no_grad():
    for images, labels in test_loader:

        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)

        _, preds = torch.max(outputs, 1)

        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

# =============================
# 5 分类报告
# =============================
print("\n📊 Classification Report:\n")

report = classification_report(
    all_labels,
    all_preds,
    target_names=class_names
)

print(report)

# 保存报告
with open("classification_report.txt", "w") as f:
    f.write(report)

# =============================
# 6 混淆矩阵
# =============================
cm = confusion_matrix(all_labels, all_preds)

plt.figure(figsize=(8,6))
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=class_names,
    yticklabels=class_names
)

plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Confusion Matrix")

plt.savefig("confusion_matrix.png")
plt.show()

print("📈 混淆矩阵已保存为 confusion_matrix.png")