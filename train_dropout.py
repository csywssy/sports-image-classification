# ======================
# train_dropout.py
# ======================

import os
import torch
import torch.nn as nn
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

# ======================
# 配置
# ======================
data_dir = "./data"           # 数据集路径
batch_size = 32               # 批量大小
num_classes = 7               # 分类数量
epochs = 20                   # 训练轮次
lr = 1e-4                     # 学习率
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Using device:", device)

# ======================
# 数据增强和加载
# ======================
train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),  # 数据增强
    transforms.ToTensor(),
])

val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

train_dataset = datasets.ImageFolder(os.path.join(data_dir, "train"), transform=train_transform)
val_dataset = datasets.ImageFolder(os.path.join(data_dir, "val"), transform=val_transform)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

print("Classes:", train_dataset.classes)

# ======================
# 模型：ResNet18 + Dropout
# ======================
model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)

# 修改全连接层，加入 Dropout
in_features = model.fc.in_features
model.fc = nn.Sequential(
    nn.Dropout(p=0.5),  # Dropout 层，丢弃 50% 节点
    nn.Linear(in_features, num_classes)
)

model = model.to(device)

# ======================
# 损失函数 & 优化器
# ======================
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=lr)

# ======================
# 训练循环
# ======================
train_losses = []
val_accuracies = []

for epoch in range(epochs):
    model.train()
    running_loss = 0.0

    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        outputs = model(images)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    avg_loss = running_loss / len(train_loader)
    train_losses.append(avg_loss)

    # ===== 验证 =====
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    acc = 100 * correct / total
    val_accuracies.append(acc)

    print(f"Epoch [{epoch+1}/{epochs}] Loss: {avg_loss:.4f} | Val Acc: {acc:.2f}%")

# ======================
# 保存模型
# ======================
os.makedirs("checkpoints", exist_ok=True)
torch.save(model.state_dict(), "checkpoints/resnet18_dropout.pth")
print("✅ 模型已保存: checkpoints/resnet18_dropout.pth")

# ======================
# 绘制训练曲线
# ======================
plt.figure(figsize=(10,4))

plt.subplot(1,2,1)
plt.plot(train_losses, label="Train Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training Loss Curve")
plt.legend()

plt.subplot(1,2,2)
plt.plot(val_accuracies, label="Validation Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy (%)")
plt.title("Validation Accuracy Curve")
plt.legend()

plt.tight_layout()
plt.savefig("training_curve_dropout.png")
plt.show()
print("📈 曲线已保存为 training_curve_dropout.png")