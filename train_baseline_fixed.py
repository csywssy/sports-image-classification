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
epochs = 50   # baseline 不用太多
lr = 1e-3     # 稍微大一点让它学快一点

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
model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)

# ❗冻结所有层
for param in model.parameters():
    param.requires_grad = False

# ❗只训练最后一层
model.fc = nn.Linear(model.fc.in_features, num_classes)

model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.fc.parameters(), lr=lr)

# ======================
# 4. 记录
# ======================
train_losses = []
val_accs = []

# ======================
# 5. 训练
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
# 6. 保存
# ======================
os.makedirs("checkpoints", exist_ok=True)
torch.save(model.state_dict(), "checkpoints/resnet18_baseline.pth")

# ======================
# 7. 曲线
# ======================
plt.figure()

plt.plot(train_losses, label="Loss")
plt.plot(val_accs, label="Val Acc")

plt.xlabel("Epoch")
plt.legend()
plt.title("Baseline Training Curve")

plt.savefig("training_curve_baseline.png")
plt.show()

print("✅ Baseline训练完成")