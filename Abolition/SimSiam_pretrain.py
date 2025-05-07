import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
from augmentions import get_transform_group
from torchvision import models

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MAX_LEN = 30

# 场景选择
mode = 'RTO'  # 'RTO'或'FAST'

# 数据集加载
class SimSiamDataset(Dataset):
    def __init__(self, csv_file, transform_group):
        self.data = pd.read_csv(csv_file)
        self.transform_group = transform_group  # 生成两侧视图的方法组合

    def __len__(self):
        return len(self.data)
    
    def pad_and_clip(self, seq):  # 固定PPI字段的序列长度为MAX_LEN
        seq = seq[:MAX_LEN]
        return seq + [0] * (MAX_LEN - len(seq))

    def __getitem__(self, idx):  # 让此类的实例支持像列表那样使用方括号索引访问
        sequence = eval(self.data.iloc[idx]['PPI'])  # 无标签自监督
        t1, t2 = self.transform_group
        view1 = self.pad_and_clip(t1(sequence))
        view2 = self.pad_and_clip(t2(sequence))  # 定长两个视图的序列
        return torch.tensor(view1, dtype=torch.float).unsqueeze(-1), torch.tensor(view2, dtype=torch.float).unsqueeze(-1)  # （MAX_LEN, 1）形状


# # SimSiam投影头/预测头
# class MLP(nn.Module):
#     def __init__(self, input_dim, hidden_dim=512, output_dim=128):  # 参数：输入特征维度、中间隐藏层维度、输出特征维度
#         super().__init__()
#         self.net = nn.Sequential(
#             nn.Linear(input_dim, hidden_dim),  # 第一个全连接层，将输入映射到隐藏层维度
#             nn.BatchNorm1d(hidden_dim),
#             nn.ReLU(),
#             nn.Dropout(0.2),
#             nn.Linear(hidden_dim, output_dim)  # 第二个全连接层，把隐藏特征映射到输出维度
#         )

#     def forward(self, x):
#         return self.net(x)


# # SimSiam模型，encoder为将视图定长后作为输入的LSTM
# class SimSiam(nn.Module):
#     def __init__(self, lstm_input_dim=1, lstm_hidden_dim=256):  # 参数：输入序列的特征维度（1表示每个时间步只有一个长度值）、LSTM的隐藏层维度，控制编码后的表示维度
#         super().__init__()
#         self.encoder = nn.LSTM(input_size=lstm_input_dim, hidden_size=lstm_hidden_dim, batch_first=True)  # LSTM编码器，batch_first指示输入形状为（batch_size,seq_len,input_size）
#         # LSTM对一个序列进行编码，返回所有时间步的输出以及最后一个时间步的隐藏状态
#         self.projector = MLP(lstm_hidden_dim, hidden_dim=512, output_dim=128)  # 投影头：把编码器输出的向量投影到另一个特征空间 
#         self.predictor = MLP(128, hidden_dim=512, output_dim=128)  # 预测头：预测另一个视图的投影向量，注意输入是128维

#     def forward(self, x1, x2):
#         _, (h1, _) = self.encoder(x1)  # h1和h2是两个视图经过LSTM后的最后一个时间步的隐藏状态，形状为（1,batch_size,hidden_dim）
#         _, (h2, _) = self.encoder(x2)
#         # 投影头
#         z1 = self.projector(h1.squeeze(0))  # 降维至形状（Batch_size, hidden_dim）后投影至新的特征空间
#         z2 = self.projector(h2.squeeze(0))
#         # 预测头
#         p1 = self.predictor(z1)  # 让某一个预测接近另一个或反之
#         p2 = self.predictor(z2)
#         return p1, z2.detach(), p2, z1.detach()  # 视图1的预测向量、视图2的投影向量（不参与梯度更新）、视图2的预测向量、视图1的投影向量（不参与梯度更新）


class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        return self.net(x)

class SimSiam(nn.Module):
    def __init__(self, lstm_input_dim=1, lstm_hidden_dim=256, num_layers=2, dropout=0.3):
        super().__init__()
        self.lstm_hidden_dim = lstm_hidden_dim
        self.bidirectional = True

        self.encoder = nn.LSTM(
            input_size=lstm_input_dim,
            hidden_size=lstm_hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout,
            bidirectional=self.bidirectional
        )

        # 双向LSTM，隐藏状态维度翻倍
        projection_input_dim = lstm_hidden_dim * 2 if self.bidirectional else lstm_hidden_dim

        self.projector = MLP(projection_input_dim, hidden_dim=512, output_dim=128)
        self.predictor = MLP(128, hidden_dim=512, output_dim=128)

    def forward(self, x1, x2):
        # x1, x2: (batch, seq_len, input_dim)
        _, (h1, _) = self.encoder(x1)  # h1: (num_layers * num_directions, batch, hidden_dim)
        _, (h2, _) = self.encoder(x2)

        # 获取最后一层的正向和反向隐藏状态并拼接
        if self.bidirectional:
            # 从最后两层（正向、反向）取出，拼接为 (batch, hidden*2)
            h1 = torch.cat((h1[-2], h1[-1]), dim=1)
            h2 = torch.cat((h2[-2], h2[-1]), dim=1)
        else:
            h1 = h1[-1]
            h2 = h2[-1]

        z1 = self.projector(h1)
        z2 = self.projector(h2)

        p1 = self.predictor(z1)
        p2 = self.predictor(z2)

        return p1, z2.detach(), p2, z1.detach()


# 负余弦相似度损失函数
def simsiam_loss(p1, z2, p2, z1):
    def D(p, z):
        return -F.cosine_similarity(p, z.detach(), dim=-1).mean()
    return D(p1, z2) / 2 + D(p2, z1) / 2


# 预训练函数
def train_simsiam(csv_file, transform_group, batch_size, epochs, patience):
    dataset = SimSiamDataset(csv_file, transform_group)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model = SimSiam().to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.05, momentum=0.9, weight_decay=1e-4)  # 使用SGD优化器
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)

    best_loss = float('inf')
    patience_counter = 0

    loss_history = []  # 记录每个epoch的loss
    index = 0

    model.train()  # 开启训练模式

    for epoch in range(epochs):
        index += 1
        total_loss = 0
        for x1, x2 in tqdm(dataloader):
            x1, x2 = x1.to(device), x2.to(device)
            optimizer.zero_grad()
            p1, z2, p2, z1 = model(x1, x2)
            loss = simsiam_loss(p1, z2, p2, z1)  # 计算单个损失
            loss.backward()  # 反向传播
            optimizer.step()  # 执行梯度下降
            scheduler.step()  # 每个epoch后更新学习率
            total_loss += loss.item()

        avg_loss = total_loss/len(dataloader)
        loss_history.append(avg_loss)
        print(f"[Epoch {epoch+1}], Loss: {total_loss/len(dataloader):.4f}")

        if avg_loss < best_loss - 1e-4:
            best_loss = avg_loss
            patience_counter = 0
            torch.save(model.encoder.state_dict(), 'simsiam_lstm_encoder.pt')
            print("Encoder saved as simsiam_lstm_encoder.pt")
        else:
            patience_counter += 1
        if patience_counter >= patience:
            print(f"Early stop for non-improvment {patience} times")
            break
        
        # 没触发早停训练完所有epoch后才绘制损失变化趋势图
        if index == 30:
            plt.plot(range(1, len(loss_history) + 1), loss_history, label="Training Loss")
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.title(f"Loss Curve--epoch={index}")
            plt.legend()
            plt.grid(True)
            plt.show()


if __name__ == '__main__':
    # 训练参数设置
    CSV_FILE = 'C:\\ETC_proj\\dataset_afterDivision\\pretrain.csv'  # 预训练数据集
    Batch_size = 512  # 批大小
    Epochs = 30  # 训练轮次
    Patience = 30  # 早停容忍次数
    # 场景选择/获取两个视图方案选择
    Transform_group = get_transform_group(mode)

    train_simsiam(
        csv_file=CSV_FILE,
        batch_size=Batch_size,
        transform_group=Transform_group,
        epochs=Epochs,
        patience=Patience
    )