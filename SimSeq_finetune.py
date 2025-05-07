import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from sklearn.preprocessing import LabelEncoder
import pandas as pd
import os
from tqdm import tqdm
import matplotlib.pyplot as plt
import umap  # encoder效果可视化相关
import seaborn as sns
from sklearn.decomposition import PCA  # 特征降维工具
from sklearn.metrics import classification_report, confusion_matrix
from collections import defaultdict
import random
import numpy as np
from SimSeq_pretrain import LSTMwithAttentionEncoder
from sklearn.manifold import TSNE


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUMCLASS = 8  # 默认的分类类别数量


# 加载微调数据集
class FineTuneDataset(Dataset):
    def __init__(self, csv_file, transform=None, label_encoder=None):
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

        # 对序列进行填充或截断处理
        sequence = self.pad_and_clip(sequence)

        # 如果提供了transform（针对长度序列的转换操作），应用于序列
        if self.transform:
            sequence = self.transform(sequence)

        return torch.tensor(sequence, dtype=torch.float).unsqueeze(-1), torch.tensor(label, dtype=torch.long)


# 加入自注意力机制的微调模型，将自注意力机制放入encoder中
class FineTuneModel(nn.Module):
    def __init__(self, pre_trained_encoder, lstm_hidden_dim=256, num_classes=NUMCLASS):
        super(FineTuneModel, self).__init__()
        self.encoder = pre_trained_encoder  # 使用预训练的编码器
        
        # 注意力层：对 LSTM 每个时间步的输出进行加权
        # self.attn = nn.Linear(lstm_hidden_dim, 1)  # 输出注意力得分（可理解为每个时间步的重要性）

        # 分类头
        self.classifier = nn.Sequential(
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
        context = self.encoder(x)
        # 分类
        out = self.classifier(context)
        return out


# 微调训练过程
def train_finetune(csv_train, csv_val, pre_trained_encoder, batch_size=64, epochs=30, learning_rate=0.001):
    # 加载训练集，生成label_encoder
    train_dataset = FineTuneDataset(csv_train)
    label_encoder = train_dataset.label_encoder  # 保存训练时fit好的label_encoder

    val_dataset = FineTuneDataset(csv_val, label_encoder=label_encoder)  # 验证集使用同一个label_encoder
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)  # 验证集不用于参数更新，不打乱

    # 微调模型（只训练分类头）
    num_classes = len(label_encoder.classes_)  # 类别数量根据label_encoder自动确定
    model = FineTuneModel(pre_trained_encoder, num_classes=num_classes).to(device)  # 将模型迁移至设备上

    # 冻结编码器，不更新encoder的权重，从而只训练头部，保持其从预训练模型中学到的知识
    for param in model.encoder.parameters():
        param.requires_grad = False

    criterion = nn.CrossEntropyLoss()  # 损失函数
    optimizer = optim.AdamW(model.classifier.parameters(), lr=learning_rate)  # Adam优化器，只优化分类头部分

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
        for x, y in tqdm(train_loader, desc=f"Training Epoch {epoch+1}/{epochs}"):
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
            for x, y in tqdm(val_loader, desc=f"Validation Epoch {epoch+1}/{epochs}"):
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
            }, 'finetuned_model.pt')  # 保存的文件中包含了两个部分：模型的参数和标签编码器，确保加载模型后，使用相同的编码方式对标签进行解码
        else:  # 验证集没有提升
            patience_counter += 1
        if patience_counter >= 10:
            print("Early stopping triggered")
            break

    # 可视化训练过程，在训练完成之后（完成所有epoch或者早停后）
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(range(1, len(train_loss_history) + 1), train_loss_history, label="Train Loss")
    plt.plot(range(1, len(val_loss_history) + 1), val_loss_history, label="Val Loss")
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Loss Curve')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(range(1, len(val_acc_history) + 1), val_acc_history, label="Val Accuracy")
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.title('Validation Accuracy')
    plt.legend()

    plt.show()


# 测试以及UMAP可视化过程
def evaluate(csv_test, pre_trained_encoder, model_path='finetuned_model.pt', output_dir='./finetune_output'):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 加载模型和label_encoder
    checkpoint = torch.load(model_path, map_location=device)
    label_encoder = checkpoint['label_encoder']

    test_dataset = FineTuneDataset(csv_test, label_encoder=label_encoder)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)  # 测试集也不用更新参数，无需打乱

    num_classes = len(label_encoder.classes_)
    model = FineTuneModel(pre_trained_encoder, num_classes=num_classes)
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
        for x, y in tqdm(test_loader, desc="Testing"):
            x, y = x.to(device), y.to(device)

            all_raw_inputs.append(x.squeeze(-1).cpu())  # [batch_size, sequence_length]
            context = model.encoder(x)
            # h = h.squeeze(0)  # [batch_size, hidden_dim]
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
    # 保存混淆矩阵为图片文件
    plt.savefig(f"{output_dir}/confusion_matrix.png")
    plt.close()


    # 特征分布可视化
    # 拼接所有样本
    all_raw_inputs = torch.cat(all_raw_inputs, dim=0).numpy()  # [num_samples, seq_len]
    all_features = torch.cat(all_features, dim=0).numpy()  # [num_samples, hidden_dim]
    all_true_labels_np = np.array(all_true_labels)

    # 每类最多保留 max_per_class 个样本用于可视化（随机选择）
    index_by_class = defaultdict(list)
    for idx, label in enumerate(all_true_labels_np):
        index_by_class[label].append(idx)

    selected_indices = []
    max_per_class = 2
    for label, indices in index_by_class.items():
        random.shuffle(indices)
        selected_indices.extend(indices[:max_per_class])

    # 按筛选出的索引取样本
    raw_inputs_vis = all_raw_inputs[selected_indices]
    features_vis = all_features[selected_indices]
    labels_vis = all_true_labels_np[selected_indices]

    # 原始输入降维可视化
    tsne_raw = TSNE(n_components=2, random_state=42, perplexity=5, n_iter=1000)
    raw_embedding = tsne_raw.fit_transform(raw_inputs_vis)

    plt.figure(figsize=(8, 6))
    sns.scatterplot(
        x=raw_embedding[:, 0], y=raw_embedding[:, 1],
        hue=[label_encoder.inverse_transform([label])[0] for label in labels_vis],
        palette='tab20', s=80, alpha=0.9
    )
    plt.title('UMAP of Raw Input')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(f"{output_dir}/umap_raw_input_sampled.png")
    plt.close()

    # 编码器输出降维可视化
    tsne_feat = TSNE(n_components=2, random_state=42, perplexity=5, n_iter=1000)
    feature_embedding = tsne_feat.fit_transform(features_vis)

    plt.figure(figsize=(8, 6))
    sns.scatterplot(
        x=feature_embedding[:, 0], y=feature_embedding[:, 1],
        hue=[label_encoder.inverse_transform([label])[0] for label in labels_vis],
        palette='tab20', s=80, alpha=0.9
    )
    plt.title('UMAP of Encoded Features')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(f"{output_dir}/umap_encoded_features_sampled.png")
    plt.close()


if __name__ == '__main__':
    encoder_path = 'C:\\ETC_proj\\TLS_ETC\\lstm_with_attention_encoder.pt'
    
    # 用于加载编码器
    encoder = LSTMwithAttentionEncoder()
    
    encoder.load_state_dict(torch.load(encoder_path, map_location=device, weights_only=True))
    encoder.to(device)

    # 微调训练
    train_finetune(
        csv_train = "C:\\ETC_proj\\dataset_afterDivision\\_finetune_split\\train.csv",
        csv_val = "C:\\ETC_proj\\dataset_afterDivision\\_finetune_split\\val.csv",
        pre_trained_encoder = encoder,  # 加载预训练阶段保存的encoder
        batch_size = 64,
        epochs = 20,  # 配合早停机制
        learning_rate = 0.0001  # Adam优化器会根据每个参数的历史梯度调整学习率
    )

    # 测试
    evaluate(
        csv_test = "C:\\ETC_proj\\dataset_afterDivision\\_finetune_split\\test.csv",
        pre_trained_encoder = encoder
    )