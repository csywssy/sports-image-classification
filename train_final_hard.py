import os
import random
import torch
import torch.nn as nn
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, Subset
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
# 2. 超强数据增强（核心）
# ======================
train_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.RandomResizedCrop(224, scale=(0.6, 1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(30),
    transforms.ColorJitter(
        brightness=0.5,
        contrast=0.5,
        saturation=0.5
    ),
    transforms.RandomGrayscale(p=0.2),
    transforms.GaussianBlur(kernel_size=5),
    transforms.ToTensor(),
])

val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# ======================
# 3. 数据加载
# ======================
train_dataset_full = datasets.ImageFolder(
    os.path.join(data_dir, "train"),
    transform=train_transform
)

val_dataset = datasets.ImageFolder(
    os.path.join(data_dir, "val"),
    transform=val_transform
)

# 🔥 限制训练数据（关键！）
indices = list(range(len(train_dataset_full)))
random.shuffle(indices)

subset_ratio = 0.5   # 只用50%数据（可以改0.3更难）
subset_indices = indices[:int(subset_ratio * len(indices))]

train_dataset = Subset(train_dataset_full, subset_indices)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size)

print("Classes:", train_dataset_full.classes)
print(f"使用训练数据: {len(train_dataset)} / {len(train_dataset_full)}")

# ======================
# 4. 模型（ResNet18 + 冻结 + Dropout）
# ======================
model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)

# 🔥 冻结 backbone（关键）
for param in model.parameters():
    param.requires_grad = False

# 只训练最后一层 + Dropout
model.fc = nn.Sequential(
    nn.Dropout(0.5),
    nn.Linear(model.fc.in_features, num_classes)
)

for param in model.fc.parameters():
    param.requires_grad = True

model = model.to(device)

# ======================
# 5. 损失函数 & 优化器
# ======================
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.fc.parameters(), lr=lr)

# ======================
# 6. 记录指标
# ======================
train_losses = []
val_accuracies = []
best_acc = 0

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

    # 保存最优模型
    if acc > best_acc:
        best_acc = acc
        os.makedirs("checkpoints", exist_ok=True)
        torch.save(model.state_dict(), "checkpoints/resnet18_final_hard.pth")
        print(f"🔥 保存最佳模型 (Acc: {best_acc:.2f}%)")

# ======================
# 8. 画曲线
# ======================
plt.figure(figsize=(10, 4))

plt.subplot(1, 2, 1)
plt.plot(train_losses, label="Train Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training Loss Curve")
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(val_accuracies, label="Validation Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy (%)")
plt.title("Validation Accuracy Curve")
plt.legend()

plt.tight_layout()
plt.savefig("training_curve_final_hard.png")
plt.show()

print("📈 曲线已保存为 training_curve_final_hard.png")
print(f"✅ 最佳准确率: {best_acc:.2f}%")