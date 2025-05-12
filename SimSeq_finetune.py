import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from sklearn.preprocessing import LabelEncoder
import pandas as pd
import os
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
from SimSeq_pretrain import SimSeqEncoder
from visualization import visualize_features
from augmentions import get_transform_group


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUMCLASS = 8  # 默认的分类类别数量
# 记录单次微调的p设置与mode
mode = 'FAST'  # 'RTO'或'FAST'
info = "p=0.02"  # 注意这里更改p设置后也需要在augmentions.py中对应修改
OUTPUT_DIR = f"C:\\ETC_proj\\TLS_ETC\\finetune\\finetune_output-{info}"
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)


# 加载微调数据集
class FineTuneDataset(Dataset):
    def __init__(self, csv_file, transform=get_transform_group(mode), label_encoder=None):  # 微调阶段需要使用同环境下的序列数据按照1：1的比例施加扰动
        self.data = pd.read_csv(csv_file)
        self.transform = transform

        # 标签编码器
        if label_encoder is None:
            self.label_encoder = LabelEncoder()
            self.data['CATEGORY'] = self.label_encoder.fit_transform(self.data['CATEGORY'])
        else:
            self.label_encoder = label_encoder
            self.data['CATEGORY'] = self.label_encoder.transform(self.data['CATEGORY'])

    def __len__(self):
        return len(self.data)

    def pad_and_clip(self, seq, max_len=30):  # 固定长度处理
        seq = seq[:max_len]  # 截断超过的部分
        return seq + [0] * (max_len - len(seq))  # 不足的部分补0

    def __getitem__(self, idx):
        sequence = eval(self.data.iloc[idx]['PPI'])  # 提取PPI字段
        label = self.data.iloc[idx]['CATEGORY']  # 提取编码后的标签

        # 对序列进行扰动处理
        t1, t2 = self.transform
        if idx % 2 == 0:  # 偶数索引使用第一个视图
            sequence = t1(sequence)
        else:  # 奇数索引使用第二个视图
            sequence = t2(sequence)
        
        # 最后对序列进行填充或截断处理
        sequence = self.pad_and_clip(sequence)

        return torch.tensor(sequence, dtype=torch.float).unsqueeze(-1), torch.tensor(label, dtype=torch.long)


# 加入自注意力机制的微调模型，将自注意力机制放入encoder中
class FineTuneModel(nn.Module):
    def __init__(self, pretrained_encoder, lstm_hidden_dim=512, num_classes=NUMCLASS):
        super(FineTuneModel, self).__init__()
        self.encoder = pretrained_encoder  # 使用预训练的编码器

        # 分类头
        self.classifier_FAST = nn.Sequential(
            nn.Linear(lstm_hidden_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes)
        )

        self.classifier_RTO = nn.Sequential(
            nn.Linear(lstm_hidden_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        if mode == 'RTO':
            out = self.classifier_RTO(self.encoder(x))
        else:
            out = self.classifier_FAST(self.encoder(x))
        return out


# 微调训练过程
def train_finetune(csv_train, csv_val, pretrained_encoder, batch_size, epochs, learning_rate):
    # 加载训练集，生成label_encoder
    train_dataset = FineTuneDataset(csv_train)
    label_encoder = train_dataset.label_encoder  # 保存训练时fit好的label_encoder

    val_dataset = FineTuneDataset(csv_val, label_encoder=label_encoder)  # 验证集使用同一个label_encoder
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)  # 验证集不用于参数更新，不打乱

    # 微调模型（只训练分类头）
    num_classes = len(label_encoder.classes_)  # 类别数量根据label_encoder自动确定
    model = FineTuneModel(pretrained_encoder, num_classes=num_classes).to(device)  # 将模型迁移至设备上

    # 冻结编码器，不更新encoder的权重，从而只训练头部，保持其从预训练模型中学到的知识
    for param in model.encoder.parameters():
        param.requires_grad = False

    criterion = nn.CrossEntropyLoss()  # 损失函数
    if mode == 'RTO':
        optimizer = optim.AdamW(model.classifier_RTO.parameters(), lr=learning_rate)  # AdamW优化器，只优化分类头部分
    else:
        optimizer = optim.Adam(model.classifier_FAST.parameters(), lr=learning_rate * 10)  # 扰动较小场景使用Adam优化器
        # optimizer = torch.optim.SGD(model.parameters(), lr=0.05, momentum=0.9, nesterov=True)  # 作为对照

    best_val_loss = float('inf')  # 当前最佳验证集上损失
    patience_counter = 0
    # 作图用损失列表
    train_loss_history = []
    val_loss_history = []
    val_acc_history = []

    for epoch in range(epochs):
        # 训练模式
        model.train()
        total_train_loss = 0
        for x, y in tqdm(train_loader, desc=f"Training Epoch {epoch+1}/{epochs}", disable=True):
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            output = model(x)
            loss = criterion(output, y)
            loss.backward()
            optimizer.step()
            total_train_loss += loss.item()

        avg_train_loss = total_train_loss / len(train_loader)
        train_loss_history.append(avg_train_loss)

        # 验证过程
        model.eval()
        total_val_loss = 0
        correct = 0
        total = 0
        with torch.no_grad():
            for x, y in tqdm(val_loader, desc=f"Validation Epoch {epoch+1}/{epochs}", disable=True):
                x, y = x.to(device), y.to(device)
                output = model(x)
                loss = criterion(output, y)
                total_val_loss += loss.item()

                _, predicted = torch.max(output, 1)
                total += y.size(0)
                correct += (predicted == y).sum().item()

        avg_val_loss = total_val_loss / len(val_loader)
        val_acc = 100 * correct / total
        val_loss_history.append(avg_val_loss)
        val_acc_history.append(val_acc)

        print(f"Epoch {epoch+1}/{epochs} - Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}, Val Accuracy: {val_acc:.2f}%")

        # 保存最好的模型
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            torch.save({
                'model_state_dict': model.state_dict(),
                'label_encoder': label_encoder,
            }, f'C:\\ETC_proj\\TLS_ETC\\finetune_models\\finetuned_model-{info}.pt')  # 保存的文件中包含了两个部分：模型的参数和标签编码器，确保加载模型后，使用相同的编码方式对标签进行解码
        else:  # 验证集没有提升
            patience_counter += 1
        if patience_counter >= 10:
            print("Early stop triggered")
            break

    # 可视化训练过程，在训练完成之后（完成所有epoch或者早停后）
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(range(1, len(train_loss_history) + 1), train_loss_history, label="Train Loss")
    plt.plot(range(1, len(val_loss_history) + 1), val_loss_history, label="Val Loss")
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Train&Val Loss')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(range(1, len(val_acc_history) + 1), val_acc_history, label="Val Accuracy")
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.title('Val Acc')
    plt.legend()

    plot_path = os.path.join(OUTPUT_DIR, f'finetune_train-{info}.png')
    plt.tight_layout()
    plt.savefig(plot_path)
    plt.close()


# 测试以及UMAP可视化
def evaluate(csv_test, pretrained_encoder, model_path=f'C:\\ETC_proj\\TLS_ETC\\finetune_models\\finetuned_model-{info}.pt', output_dir=OUTPUT_DIR):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 加载模型和label_encoder
    checkpoint = torch.load(model_path, map_location=device)
    label_encoder = checkpoint['label_encoder']

    test_dataset = FineTuneDataset(csv_test, label_encoder=label_encoder)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)  # 测试集也不用更新参数，无需打乱

    num_classes = len(label_encoder.classes_)
    model = FineTuneModel(pretrained_encoder, num_classes=num_classes)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()  # 评估模式

    correct = 0
    total = 0

    all_raw_inputs = []  # 原始输入序列
    all_features = []  # encoder输出
    all_true_labels = []  # 保存真实标签
    all_pred_labels = []  # 保存预测标签

    with torch.no_grad():
        for x, y in tqdm(test_loader, desc="Testing", disable=True):
            x, y = x.to(device), y.to(device)

            all_raw_inputs.append(x.squeeze(-1).cpu())  # [batch_size, sequence_length]
            context = model.encoder(x)
            all_features.append(context.cpu())
            all_true_labels.extend(y.cpu().numpy())

            output = model(x)
            _, predicted = torch.max(output, 1)
            total += y.size(0)
            correct += (predicted == y).sum().item()
            all_pred_labels.extend(predicted.cpu().numpy())  # 保存预测标签

    # 模型在测试集上的准确度
    accuracy = 100 * correct / total
    print(f"Test Accuracy: {accuracy:.2f}%")

    # 保存整体准确率到文件
    with open(f"{output_dir}/test_accuracy.txt", 'w') as f:
        f.write(f"Test Accuracy: {accuracy:.2f}%\n")

    # 计算准确率、召回率和F1-score
    report = classification_report(all_true_labels, all_pred_labels, target_names=label_encoder.classes_)

    # 保存分类报告到文件
    with open(f"{output_dir}/classification_report.txt", 'w') as f:
        f.write(report)

    # 混淆矩阵
    cm = confusion_matrix(all_true_labels, all_pred_labels)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=label_encoder.classes_, yticklabels=label_encoder.classes_)
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.tight_layout()
    plt.savefig(f"{output_dir}/confusion_matrix.png")
    plt.close()

    # 特征分布可视化
    visualize_features(all_raw_inputs, all_features, all_true_labels, label_encoder, output_dir)


if __name__ == '__main__':
    encoder_path = f"C:\\ETC_proj\\TLS_ETC\\SimSeq_Encoders\\SimSeqEncoder-{info}.pt"
    
    # 加载编码器
    encoder = SimSeqEncoder()
    encoder.load_state_dict(torch.load(encoder_path, map_location=device, weights_only=True))
    encoder.to(device)

    # 微调训练
    train_finetune(
        csv_train = "C:\\ETC_proj\\dataset_afterDivision\\_finetune_split\\train.csv",
        csv_val = "C:\\ETC_proj\\dataset_afterDivision\\_finetune_split\\val.csv",
        pretrained_encoder = encoder,  # 加载预训练阶段保存的encoder
        batch_size = 64,
        epochs = 100,  # 配合早停机制，5次没有性能提升就中止
        learning_rate = 0.0001  # Adam优化器会根据每个参数的历史梯度调整学习率
    )

    # 测试
    evaluate(
        csv_test = "C:\\ETC_proj\\dataset_afterDivision\\_finetune_split\\test.csv",
        pretrained_encoder = encoder
    )