# pytorch框架下的1D-CNN和LSTM模型，尝试轻量化
# 可以切换使用1D-CNN或者LSTM

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import ast
from sklearn.preprocessing import LabelEncoder
import torch.nn.functional as F
from sklearn.metrics import classification_report, confusion_matrix


# 数据集类
# 加载并预处理csv文件中存储的PPI序列和对应标签'CATEGORY'，提供给DataLoader使用
class PPIDataset(Dataset):
    def __init__(self, csv_path):
        df = pd.read_csv(csv_path)
        self.sequences = df['PPI'].apply(lambda x: list(map(float, ast.literal_eval(x)))).tolist()
        self.labels = df['CATEGORY'].tolist()  # 提取'CATEGORY'作为label
        self.encoder = LabelEncoder()  # 将字符串标签转为整数索引
        self.labels = self.encoder.fit_transform(self.labels)  # 自动识别所有不同类别并转换

    def __len__(self):
        return len(self.sequences)  # 返回数据集大小（样本数量）

    def __getitem__(self, idx):  # 取样返回样本和label
        x = torch.tensor(self.sequences[idx], dtype=torch.float32)
        y = torch.tensor(self.labels[idx], dtype=torch.long)
        return x, y


# 1D-CNN模型类
class CNNClassifier(nn.Module):
    def __init__(self, num_classes):
        super(CNNClassifier, self).__init__()
        # 卷积层和池化层
        # 输入通道数为1，输出通道数为16，卷积核大小为3，步长为1，填充为1
        # 激活函数使用ReLU，池化层使用最大池化，池化核大小为2
        # 最后使用自适应最大池化将输出大小调整为1
        # 使用线性层将输出映射到num_classes个类别
        self.conv = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=3, padding=1),  # padding=1保持输入输出大小一致，输出尺寸变为[Batch_size, 16, 30]
            nn.ReLU(),
            nn.MaxPool1d(2),  # 最大池化，将序列长度减半
            nn.Conv1d(16, 32, kernel_size=3, padding=1),  # 保持长度不变，输出尺寸变为[Batch_size, 32, 15]
            nn.ReLU(),
            nn.AdaptiveMaxPool1d(1)  # 自适应池化到长度为1，输出尺寸变为[Batch_size, 32, 1]
        )
        self.fc = nn.Linear(32, num_classes)  # 将输出映射到num_classes个类别

    def forward(self, x):
        x = x.unsqueeze(1)  # 加上一个通道维度将尺寸变为[Batch_size, 1, 30]
        x = self.conv(x)    # 卷积，完成后输出尺寸为[Batch_size, 32, 1]
        x = x.squeeze(-1)   # 去掉最后的长度维度，使得尺寸变为[Batch_size, 32]
        return self.fc(x)   # 最终输出尺寸为[Batch_size, num_classes]，即每个样本对应的类别概率分布（logits）


# LSTM模型类
class LSTMClassifier(nn.Module):
    def __init__(self, hidden_size, num_classes):  # 参数：隐藏层的维度、分类的类别数量
        super(LSTMClassifier, self).__init__()
        self.lstm = nn.LSTM(input_size=1, hidden_size=hidden_size, batch_first=True)  # 每个时间步输入一个标量、每个时间步输出的维度
        self.fc = nn.Linear(hidden_size, num_classes)  # 最终输出尺寸为[Batch_size, num_classes]，即每个样本对应的类别概率分布（logits）

    def forward(self, x):
        x = x.unsqueeze(-1)  # 增加一个维度，每个时间步一个标量，尺寸变为[Batch_size, 30, 1]
        _, (hn, _) = self.lstm(x)  # hn：最后一个时间步的隐藏状态，尺寸为[1, Batch_size, hidden_size]
        return self.fc(hn[-1])  # 将最后的隐藏状态作为整条序列的表示（logits）


# 训练函数
def train_epoch(model, loader, optimizer, criterion, device): 
    model.train()  # 将模型设置为训练模式
    total_loss, correct = 0, 0  # 初始化损失和正确分类数
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()  # 清除上一轮的梯度
        out = model(x)  # 前向传播
        loss = criterion(out, y)  # 计算损失
        loss.backward()  # 计算每个参数的梯度
        optimizer.step()  # 根据梯度更新参数
        total_loss += loss.item() * x.size(0)
        correct += (out.argmax(dim=1) == y).sum().item()  # 每个样本预测的最大概率类与label相等的数量
    return total_loss / len(loader.dataset), correct / len(loader.dataset)  # 返回平均损失和准确率


# 评估函数
# 传入model是已经训练好的模型，loader加载验证集或测试集
def eval_model(model, loader, criterion, device, show_report=False):
    model.eval()  # 进入评估模式
    total_loss, correct = 0, 0
    all_preds, all_labels = [], []  # 用于存储所有预测和标签
    with torch.no_grad():  # 验证无需更新参数
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            out = model(x)
            loss = criterion(out, y)
            total_loss += loss.item() * x.size(0)
            preds = out.argmax(dim=1)
            correct += (out.argmax(dim=1) == y).sum().item()

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(y.cpu().numpy())
    
    acc = correct / len(loader.dataset)
    avg_loss = total_loss / len(loader.dataset)

    if show_report:
        print("混淆矩阵:")
        print(confusion_matrix(all_labels, all_preds))
        print("\n分类报告:")
        print(classification_report(all_labels, all_preds, digits=4))

    return avg_loss, acc


def main(model_type='CNN'):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 数据加载 ： 四种类型的数据
    # 读取PPI字段的序列，解析为数值列表，提取'CATEGORY'作为标签且对标签进行编码
    train_set = PPIDataset('D:/ETC_proj/dataset_afterProcess/FixLBasewithMoreS/_train.csv')
    val_set = PPIDataset('D:/ETC_proj/dataset_afterProcess/FixLBasewithMoreS/_val.csv')
    test_set = PPIDataset('D:/ETC_proj/dataset_afterProcess/FixLBasewithMoreS/_test.csv')

    # train_set = PPIDataset('D:/ETC_proj/dataset_afterProcess/FixLBasewithMoreSBase/_train.csv')
    # val_set = PPIDataset('D:/ETC_proj/dataset_afterProcess/FixLBasewithMoreSBase/_val.csv')
    # test_set = PPIDataset('D:/ETC_proj/dataset_afterProcess/FixLBasewithMoreSBase/_test.csv')

    # train_set = PPIDataset('D:/ETC_proj/dataset_afterProcess/FixLwithMoreS/_train.csv')
    # val_set = PPIDataset('D:/ETC_proj/dataset_afterProcess/FixLwithMoreS/_val.csv')
    # test_set = PPIDataset('D:/ETC_proj/dataset_afterProcess/FixLwithMoreS/_test.csv')

    # train_set = PPIDataset('D:/ETC_proj/dataset_afterProcess/FixLwithMoreSBase/_train.csv')
    # val_set = PPIDataset('D:/ETC_proj/dataset_afterProcess/FixLwithMoreSBase/_val.csv')
    # test_set = PPIDataset('D:/ETC_proj/dataset_afterProcess/FixLwithMoreSBase/_test.csv')

    num_classes = len(set(train_set.labels))  # 标签总类数（最后一层的输出维度）

    train_loader = DataLoader(train_set, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=32)
    test_loader = DataLoader(test_set, batch_size=32)

    # 训练参数选择
    # 1D-CNN
    if model_type == 'CNN':  # 捕捉局部序列特征
        model = CNNClassifier(num_classes)
    # LSTM
    else:  # 捕捉时序特征
        model = LSTMClassifier(hidden_size=64, num_classes=num_classes)

    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)  # Adam优化器，学习率为0.001

    # 早停参数配置
    best_val_acc = 0.0
    patience = 5  # 最多接受多少次验证集无提升
    trigger_times = 0  # 触发次数
    best_model_path = 'best_model.pth'  # 最佳模型保存路径

    # 训练过程
    for epoch in range(1, 51):  # 最多训练50个epoch，结合早停策略
        # 每个epoch使用所有的batch进行训练和验证
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_acc = eval_model(model, val_loader, criterion, device)
        print(f"[Epoch {epoch}] Train Acc: {train_acc:.4f}, Val Acc: {val_acc:.4f}")

        # 判断模型在验证集上的准确率是否提升
        # 训练集上的准确率几乎一定是持续上涨的，早停策略应当依赖于验证集表现
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            trigger_times = 0
            torch.save(model.state_dict(), best_model_path) # 保存最佳模型
            print(f"模型保存至 {best_model_path}")
        else:
            trigger_times += 1
            if trigger_times >= patience:
                print(f"验证集准确率连续{patience}次未提升，提前停止")
                break

    # 最终测试评估模型性能
    model.load_state_dict(torch.load(best_model_path))  # 加载最佳模型测试
    test_loss, test_acc = eval_model(model, test_loader, criterion, device, )
    print(f"Test Acc: {test_acc:.4f}")

if __name__ == '__main__':
    main(model_type='CNN')  # 'CNN'或'LSTM'