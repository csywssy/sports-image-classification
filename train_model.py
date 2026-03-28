import torch
import torch.nn as nn
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
import os

# =========================
# 1 基础配置
# =========================

data_dir = "D:/Thesis_new/data"

batch_size = 32
num_classes = 7
epochs = 10
lr = 0.0001

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)


# =========================
# 2 数据增强（论文关键）
# =========================

train_transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ToTensor(),
])

val_transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
])


# =========================
# 3 加载数据
# =========================

train_dataset = datasets.ImageFolder(
    os.path.join(data_dir,"train"),
    transform=train_transform
)

val_dataset = datasets.ImageFolder(
    os.path.join(data_dir,"val"),
    transform=val_transform
)

train_loader = DataLoader(
    train_dataset,
    batch_size=batch_size,
    shuffle=True
)

val_loader = DataLoader(
    val_dataset,
    batch_size=batch_size,
    shuffle=False
)

print("类别映射:", train_dataset.class_to_idx)


# =========================
# 4 加载 ResNet18
# =========================

model = models.resnet18(pretrained=True)

model.fc = nn.Linear(model.fc.in_features, num_classes)

model = model.to(device)


# =========================
# 5 损失函数 & 优化器
# =========================

criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=lr
)


# =========================
# 6 训练函数
# =========================

def train_epoch():

    model.train()

    total_loss = 0
    correct = 0
    total = 0

    for images,labels in train_loader:

        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)

        loss = criterion(outputs,labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        _,pred = torch.max(outputs,1)

        total += labels.size(0)
        correct += (pred == labels).sum().item()

    acc = correct/total

    return total_loss/len(train_loader),acc


# =========================
# 7 验证函数
# =========================

def val_epoch():

    model.eval()

    correct = 0
    total = 0

    with torch.no_grad():

        for images,labels in val_loader:

            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)

            _,pred = torch.max(outputs,1)

            total += labels.size(0)
            correct += (pred == labels).sum().item()

    acc = correct/total

    return acc


# =========================
# 8 训练循环
# =========================

best_acc = 0

for epoch in range(epochs):

    train_loss,train_acc = train_epoch()

    val_acc = val_epoch()

    print(f"Epoch {epoch+1}/{epochs}")
    print(f"Train Loss: {train_loss:.4f}")
    print(f"Train Acc: {train_acc:.4f}")
    print(f"Val Acc: {val_acc:.4f}")

    if val_acc > best_acc:

        best_acc = val_acc

        torch.save(
            model.state_dict(),
            "best_model.pth"
        )

        print("模型已保存")


print("训练完成")