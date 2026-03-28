import os
import torch
import torch.nn as nn
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader

# ================= 配置 =================
data_dir = "./data"
batch_size = 32
num_classes = 7
epochs = 20
lr = 1e-4

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# ================= 数据增强 =================
train_transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
])

val_transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
])

train_dataset = datasets.ImageFolder(
    os.path.join(data_dir,"train"),
    transform=train_transform
)

val_dataset = datasets.ImageFolder(
    os.path.join(data_dir,"val"),
    transform=val_transform
)

train_loader = DataLoader(train_dataset,batch_size=batch_size,shuffle=True)
val_loader = DataLoader(val_dataset,batch_size=batch_size)

print("Classes:",train_dataset.classes)

# ================= 模型 =================
model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
model.fc = nn.Linear(model.fc.in_features,num_classes)
model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(),lr=lr)

# ================= 训练 =================
for epoch in range(epochs):

    model.train()
    running_loss = 0

    for images,labels in train_loader:

        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)
        loss = criterion(outputs,labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    # ===== 验证 =====
    model.eval()

    correct = 0
    total = 0

    with torch.no_grad():
        for images,labels in val_loader:

            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)

            _,predicted = torch.max(outputs,1)

            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    acc = 100 * correct / total

    print(f"Epoch [{epoch+1}/{epochs}] Loss:{running_loss:.4f} Acc:{acc:.2f}%")

# ================= 保存模型 =================
os.makedirs("checkpoints",exist_ok=True)

torch.save(model.state_dict(),"checkpoints/resnet18_augmented.pth")

print("✅ 增强模型训练完成")