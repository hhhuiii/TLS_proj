import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
from augmentions import get_transform_group


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 场景选择
mode = 'RTO'  # 'RTO'或'FAST'
# LSTM输入定长设置
MAX_LEN = 30


# SimSiam投影头/预测头
class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim=512, output_dim=128):  # 参数：输入特征维度、中间隐藏层维度、输出特征维度
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),  # 第一个全连接层，将输入映射到隐藏层维度
            nn.LayerNorm(hidden_dim),  # LSTM更常用层归一化，对每个样本的所有特征维度归一化，加快收敛，稳定训练
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, output_dim)  # 第二个全连接层，把隐藏特征映射到输出维度
        )

    def forward(self, x):
        return self.net(x)


class SimSiamDataset(Dataset):
    def __init__(self, csv_file, transform_group):
        self.data = pd.read_csv(csv_file)
        self.transform_group = transform_group

    def __len__(self):
        return len(self.data)
    
    def pad_and_clip(self, seq):
        seq = seq[:MAX_LEN]
        return seq + [0] * (MAX_LEN - len(seq))

    def __getitem__(self, idx):
        sequence = eval(self.data.iloc[idx]['PPI'])
        t1, t2 = self.transform_group
        view1 = self.pad_and_clip(t1(sequence))
        view2 = self.pad_and_clip(t2(sequence))
        return torch.tensor(view1, dtype=torch.float).unsqueeze(-1), \
               torch.tensor(view2, dtype=torch.float).unsqueeze(-1)


# attention池化
class AttentionPooling(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.attn = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.Tanh(),
            nn.Linear(128, 1)
        )

    def forward(self, x):  # x: [B, T, H]
        attn_scores = self.attn(x)  # [B, T, 1]
        attn_weights = torch.softmax(attn_scores, dim=1)  # [B, T, 1]
        context = torch.sum(attn_weights * x, dim=1)  # [B, H]
        return context


# 附加attention机制的LSTM编码器
class LSTMwithAttentionEncoder(nn.Module):
    def __init__(self, input_dim=1, hidden_dim=256):
        super().__init__()
        self.lstm = nn.LSTM(input_size=input_dim, hidden_size=hidden_dim, batch_first=True, bidirectional=True)
        self.attn_pool = AttentionPooling(hidden_dim * 2)

    def forward(self, x):  # x: [B, T, D]
        output, _ = self.lstm(x)  # [B, T, H]
        context = self.attn_pool(output)  # [B, H]
        return context


# SimCLR类似模型
class SimCLR(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = LSTMwithAttentionEncoder()  # 将注意力机制引入encoder中的LSTM
        self.projector = MLP(512, 512, 128)

    def forward(self, x):
        z = self.projector(self.encoder(x))
        return F.normalize(z, dim=1)


# NT-Xent损失函数，最小化正样本对距离，最大化负样本（同batch中其他样本）对距离
def NT_Xent_loss(z1, z2, temperature=0.07):
    batch_size = z1.size(0)
    z = torch.cat([z1, z2], dim=0)  # [2B, D]
    z = F.normalize(z, dim=1)

    # 相似度矩阵 [2B, 2B]
    similarity_matrix = torch.matmul(z, z.T) / temperature

    # 屏蔽对角线（自身相似度）
    mask = torch.eye(2 * batch_size, dtype=torch.bool).to(z.device)
    similarity_matrix.masked_fill_(mask, -1e9)

    # 构造正样本标签
    labels = torch.arange(batch_size, device=z.device)
    labels = torch.cat([labels + batch_size, labels])  # z1 的正样本是 z2，z2 的正样本是 z1

    # 交叉熵损失：每一行的正样本是 labels[i]
    loss = F.cross_entropy(similarity_matrix, labels)
    return loss


def train_simclr(csv_file, transform_group, batch_size, epochs, patience):
    dataset = SimSiamDataset(csv_file, transform_group)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model = SimCLR().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)

    best_loss = float('inf')
    patience_counter = 0
    loss_history = []

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for x1, x2 in tqdm(dataloader):
            x1, x2 = x1.to(device), x2.to(device)
            z1 = model(x1)
            z2 = model(x2)
            loss = NT_Xent_loss(z1, z2)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(dataloader)
        loss_history.append(avg_loss)
        print(f"Epoch {epoch+1}: Loss = {avg_loss:.4f}")

        if avg_loss < best_loss - 1e-4:
            best_loss = avg_loss
            patience_counter = 0
            torch.save(model.encoder.state_dict(), 'LSTM_with_AttentionInEncoder.pt')
            print("Encoder saved as LSTM_with_AttentionInEncoder.pt")
        else:
            patience_counter += 1
        if patience_counter >= patience:
            print(f"Early stopping triggered after {patience} no-improvement epochs.")
            break

    plt.plot(range(1, len(loss_history) + 1), loss_history, label="Training Loss")
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('NT-Xent Loss Curve')
    plt.legend()
    plt.grid(True)
    plt.show()


if __name__ == '__main__':
    # 训练参数设置
    CSV_FILE = 'C:\\ETC_proj\\dataset_afterDivision\\pretrain.csv'  # 预训练数据集
    Batch_size = 2048  # 批大小
    Epochs = 100  # 训练轮次，趋近0为收敛
    Patience = 30  # 早停容忍次数
    # 场景选择/获取两个视图方案选择
    Transform_group = get_transform_group(mode)

    train_simclr(
        csv_file=CSV_FILE,
        batch_size=Batch_size,
        transform_group=Transform_group,
        epochs=Epochs,
        patience=Patience
    )