import torch
import torchvision.transforms as transforms
from torchvision.models import swin_t, Swin_T_Weights
import numpy as np
from PIL import Image

import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import os
from os.path import basename
from glob import iglob

# Configure GPU
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {DEVICE}")

# Update path for dataset
TRAIN = './teaLeafBD/teaLeafBD'

# 创建自定义数据集类
class TeaLeafDataset(Dataset):
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')
        label = self.labels[idx]

        if self.transform:
            image = self.transform(image)

        return image, label

# 准备数据集
def prepare_dataset():
    print("准备数据集...")
    image_paths = []
    labels = []
    label_to_idx = {}

    for idx, class_folder in enumerate(iglob(TRAIN + '/*')):
        class_name = basename(class_folder)
        label_to_idx[class_name] = idx

        for img_path in iglob(class_folder + '/*.jpg'):
            image_paths.append(img_path)
            labels.append(idx)

    idx_to_label = {v: k for k, v in label_to_idx.items()}

    return image_paths, labels, label_to_idx, idx_to_label

# 数据增强和转换
train_transform = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

test_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# 创建微调模型
def create_finetune_model(num_classes):
    # 使用预训练的Swin Transformer模型
    model = swin_t(weights=Swin_T_Weights.IMAGENET1K_V1)

    # 替换最后的分类头
    num_features = model.head.in_features
    model.head = nn.Linear(num_features, num_classes)

    return model

# 训练函数
def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, num_epochs=10):
    best_acc = 0.0
    history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}

    for epoch in range(num_epochs):
        print(f'Epoch {epoch+1}/{num_epochs}')
        print('-' * 10)

        # 训练阶段
        model.train()
        running_loss = 0.0
        running_corrects = 0

        for inputs, labels in train_loader:
            inputs = inputs.to(DEVICE)
            labels = labels.to(DEVICE)

            optimizer.zero_grad()

            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)

        if scheduler:
            scheduler.step()

        epoch_loss = running_loss / len(train_loader.dataset)
        epoch_acc = running_corrects.double() / len(train_loader.dataset)

        history['train_loss'].append(epoch_loss)
        history['train_acc'].append(epoch_acc.item())

        print(f'Train Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

        # 验证阶段
        model.eval()
        running_loss = 0.0
        running_corrects = 0

        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs = inputs.to(DEVICE)
                labels = labels.to(DEVICE)

                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                loss = criterion(outputs, labels)

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

        epoch_loss = running_loss / len(val_loader.dataset)
        epoch_acc = running_corrects.double() / len(val_loader.dataset)

        history['val_loss'].append(epoch_loss)
        history['val_acc'].append(epoch_acc.item())

        print(f'Val Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

        # 保存最佳模型
        if epoch_acc > best_acc:
            best_acc = epoch_acc
            torch.save(model.state_dict(), os.path.join(os.getcwd(),'best_model.pth'))

    print(f'Best val Acc: {best_acc:.4f}')
    return model, history

# 评估函数
def evaluate_model(model, test_loader, idx_to_label):
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(DEVICE)
            labels = labels.to(DEVICE)

            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # 计算分类报告
    class_names = [idx_to_label[i] for i in range(len(idx_to_label))]
    report = classification_report(all_labels, all_preds, target_names=class_names, digits=4)
    print(report)

    # 计算混淆矩阵
    cm = confusion_matrix(all_labels, all_preds)

    # 绘制混淆矩阵
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.savefig(os.path.join(os.getcwd(),'confusion_matrix.png'))
    plt.show()

    return report, cm

# 单张图片预测
def predict_single_image(model, image_path, idx_to_label, transform=test_transform):
    model.eval()

    image = Image.open(image_path).convert('RGB')
    image_tensor = transform(image).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        outputs = model(image_tensor)
        probabilities = nn.functional.softmax(outputs, dim=1)
        _, predicted_idx = torch.max(outputs, 1)

    predicted_label = idx_to_label[predicted_idx.item()]
    probability = probabilities[0][predicted_idx].item()

    # 获取所有类别的概率
    probs_dict = {idx_to_label[i]: prob.item() for i, prob in enumerate(probabilities[0])}

    # 按概率从高到低排序
    sorted_probs = sorted(probs_dict.items(), key=lambda x: x[1], reverse=True)

    return predicted_label, probability, sorted_probs

# 绘制训练历史
def plot_training_history(history):
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Validation Loss')
    plt.title('Loss over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history['train_acc'], label='Train Accuracy')
    plt.plot(history['val_acc'], label='Validation Accuracy')
    plt.title('Accuracy over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(os.getcwd(),'training_history.png'))
    plt.show()

# 主函数
def main():
    # 准备数据
    image_paths, labels, label_to_idx, idx_to_label = prepare_dataset()
    print(f"发现 {len(set(labels))} 个茶叶疾病类别：")
    for idx, name in idx_to_label.items():
        count = labels.count(idx)
        print(f"  - {name}: {count} 张图片")

    # 数据集分割
    train_paths, test_paths, train_labels, test_labels = train_test_split(
        image_paths, labels, test_size=0.2, random_state=42, stratify=labels
    )

    train_paths, val_paths, train_labels, val_labels = train_test_split(
        train_paths, train_labels, test_size=0.15, random_state=42, stratify=train_labels
    )

    print(f"训练集: {len(train_paths)} 张图片")
    print(f"验证集: {len(val_paths)} 张图片")
    print(f"测试集: {len(test_paths)} 张图片")

    # 创建数据集和加载器
    train_dataset = TeaLeafDataset(train_paths, train_labels, transform=train_transform)
    val_dataset = TeaLeafDataset(val_paths, val_labels, transform=test_transform)
    test_dataset = TeaLeafDataset(test_paths, test_labels, transform=test_transform)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=4)

    # 创建模型
    num_classes = len(label_to_idx)
    model = create_finetune_model(num_classes)
    model = model.to(DEVICE)

    # 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss()

    # 使用不同的学习率策略 - 主干网络较小的学习率，分类头较大的学习率
    params_to_update = [
        {'params': [param for name, param in model.named_parameters() if 'head' not in name], 'lr': 1e-4},
        {'params': model.head.parameters(), 'lr': 1e-3}
    ]

    optimizer = optim.AdamW(params_to_update, weight_decay=0.01)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

    # 训练模型
    trained_model, history = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        num_epochs=15
    )

    # 绘制训练历史
    plot_training_history(history)

    # 加载最佳模型 - 修正文件扩展名
    model.load_state_dict(torch.load(os.path.join(os.getcwd(),'best_model.pth')))

    # 评估模型
    print("\n模型评估结果:")
    report, cm = evaluate_model(model, test_loader, idx_to_label)

    # 单张图片测试
    print("\n测试单张图片:")
    # 从测试集随机选择几张图片进行测试
    import random

    test_samples = random.sample(list(zip(test_paths, test_labels)), 5)

    for img_path, true_label in test_samples:
        pred_label, prob, sorted_probs = predict_single_image(model, img_path, idx_to_label)
        true_class = idx_to_label[true_label]

        print(f"\n图片: {os.path.basename(img_path)}")
        print(f"真实类别: {true_class}")
        print(f"预测类别: {pred_label} (置信度: {prob:.4f})")
        print("所有类别预测概率:")
        for class_name, p in sorted_probs[:3]:  # 只显示前3个概率最高的
            print(f"  - {class_name}: {p:.4f}")

        # 显示图片和预测结果
        img = Image.open(img_path).convert('RGB')
        plt.figure(figsize=(6, 6))
        plt.imshow(img)

        color = "green" if pred_label == true_class else "red"
        plt.title(f"真实: {true_class}\n预测: {pred_label} ({prob:.2f})", color=color)
        plt.axis('off')
        plt.show()

# 如果当前脚本是主程序，执行main函数
if __name__ == "__main__":
    main()