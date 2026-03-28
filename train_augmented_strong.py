import os
import torch
import torch.nn as nn
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

# ======================
# 1. 基本配置
# ======================
data_dir = "./data"
batch_size = 32
num_classes = 7
epochs = 50
lr = 1e-4

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# ======================
# 2. 强数据增强（核心改进）
# ======================
train_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(20),
    transforms.ColorJitter(brightness=0.3, contrast=0.3),
    transforms.ToTensor(),
])

val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# ======================
# 3. 数据加载
# ======================
train_dataset = datasets.ImageFolder(
    os.path.join(data_dir, "train"),
    transform=train_transform
)

val_dataset = datasets.ImageFolder(
    os.path.join(data_dir, "val"),
    transform=val_transform
)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size)

print("Classes:", train_dataset.classes)

# ======================
# 4. 模型（ResNet18 + Dropout）
# ======================
model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)

# 加 Dropout
model.fc = nn.Sequential(
    nn.Dropout(0.5),
    nn.Linear(model.fc.in_features, num_classes)
)

model = model.to(device)

# ======================
# 5. 损失函数 & 优化器
# ======================
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=lr)

# ======================
# 6. 记录指标
# ======================
train_losses = []
val_accuracies = []

best_acc = 0  # 用于保存最优模型

# ======================
# 7. 训练
# ======================
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

    print(f"Epoch [{epoch+1}/{epochs}] "
          f"Loss: {avg_loss:.4f} | Val Acc: {acc:.2f}%")

    # ===== 保存最优模型 =====
    if acc > best_acc:
        best_acc = acc
        os.makedirs("checkpoints", exist_ok=True)
        torch.save(model.state_dict(), "checkpoints/resnet18_strong_aug.pth")
        print(f"🔥 保存最佳模型 (Acc: {best_acc:.2f}%)")

# ======================
# 8. 画曲线（论文用）
# ======================
plt.figure(figsize=(10, 4))

# Loss 曲线
plt.subplot(1, 2, 1)
plt.plot(train_losses, label="Train Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training Loss Curve")
plt.legend()

# Accuracy 曲线
plt.subplot(1, 2, 2)
plt.plot(val_accuracies, label="Validation Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy (%)")
plt.title("Validation Accuracy Curve")
plt.legend()

plt.tight_layout()
plt.savefig("training_curve_strong_aug.png")
plt.show()

print("📈 曲线已保存为 training_curve_strong_aug.png")
print(f"✅ 最佳准确率: {best_acc:.2f}%")