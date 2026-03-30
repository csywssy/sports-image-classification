import torch
import torch.nn as nn
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
import os
import matplotlib.pyplot as plt

# ======================
# 1. 配置
# ======================
data_dir = "./data"
batch_size = 32
num_classes = 7
epochs = 150
lr = 1e-4

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# ======================
# 2. 数据（无增强）
# ======================
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

train_dataset = datasets.ImageFolder(os.path.join(data_dir, "train"), transform)
val_dataset = datasets.ImageFolder(os.path.join(data_dir, "val"), transform)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size)

print("Classes:", train_dataset.classes)

# ======================
# 3. 模型
# ======================
model = models.resnet18(weights=None)

# ❗关键：部分解冻（只训练后面层）
for name, param in model.named_parameters():
    if "layer4" in name or "fc" in name:
        param.requires_grad = True
    else:
        param.requires_grad = False

# 替换分类层
model.fc = nn.Linear(model.fc.in_features, num_classes)

model = model.to(device)

# ======================
# 4. 损失 & 优化器
# ======================
criterion = nn.CrossEntropyLoss()

# ❗只优化可训练参数
optimizer = torch.optim.Adam(
    filter(lambda p: p.requires_grad, model.parameters()),
    lr=lr
)

# ======================
# 5. 记录
# ======================
train_losses = []
val_accs = []

# ======================
# 6. 训练
# ======================
for epoch in range(epochs):
    model.train()
    running_loss = 0

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
            _, preds = torch.max(outputs, 1)

            total += labels.size(0)
            correct += (preds == labels).sum().item()

    acc = 100 * correct / total
    val_accs.append(acc)

    print(f"Epoch [{epoch+1}/{epochs}] Loss: {avg_loss:.4f} | Val Acc: {acc:.2f}%")

# ======================
# 7. 保存模型
# ======================
os.makedirs("checkpoints", exist_ok=True)
torch.save(model.state_dict(), "checkpoints/resnet18_baseline_v2.pth")

# ======================
# 8. 画图
# ======================
plt.figure(figsize=(10,4))

plt.subplot(1,2,1)
plt.plot(train_losses)
plt.title("Train Loss")

plt.subplot(1,2,2)
plt.plot(val_accs)
plt.title("Val Accuracy")

plt.tight_layout()
plt.savefig("training_curve_baseline_v2.png")
plt.show()

print("✅ Baseline v2 训练完成")