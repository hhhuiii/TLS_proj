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
mode = 'FAST'  # 'RTO'或'FAST'
# LSTM输入定长设置
MAX_LEN = 30
info = "p=0.02"  # 记录某次训练的丢包率设置


class SimSeqDataset(Dataset):
    def __init__(self, csv_file, transform_group):
        self.data = pd.read_csv(csv_file)
        self.transform_group = transform_group  # 传入的增强组，增强组属于一个场景下

    def __len__(self):
        return len(self.data)
    
    def pad_and_clip(self, seq):  # 所有输入长度为30定长，长则截断，短则补0
        seq = seq[:MAX_LEN]
        return seq + [0] * (MAX_LEN - len(seq))

    def __getitem__(self, idx):
        sequence = eval(self.data.iloc[idx]['PPI'])
        t1, t2 = self.transform_group
        view1 = self.pad_and_clip(t1(sequence))
        view2 = self.pad_and_clip(t2(sequence))
        # 返回序列扰动后的两个视图
        return torch.tensor(view1, dtype=torch.float).unsqueeze(-1), \
               torch.tensor(view2, dtype=torch.float).unsqueeze(-1)


# attention池化，在时间维度上给每个时刻的输出分配权重，然后加权求和得到序列的整体表示
class AttentionPooling(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.attn = nn.Sequential(
            nn.Linear(hidden_dim, 128),  # 将每个时间步的隐状态压缩为128维
            nn.Tanh(),  # 引入非线性，提升表达能力
            nn.Linear(128, 1)  # 映射为一个标量打分
        )

    def forward(self, x):  # x形状[B, T, H]batch_size、序列长度/时间步长度、每个时间步的隐状态维度
        attn_scores = self.attn(x)  # 形状[B, T, 1]，为每个时间步计算一个标量打分
        attn_weights = torch.softmax(attn_scores, dim=1)  # 形状[B, T, 1]，对时间维度T上的注意力分数做softmax，确保一个序列中所有时间步的权重加起来为1
        context = torch.sum(attn_weights * x, dim=1)  # 形状[B, H]（广播），沿着时间维度T做加和，得到每个序列的加权表示
        return context


# 附加attention机制的LSTM编码器
class SimSeqEncoder(nn.Module):
    def __init__(self, input_dim=1, hidden_dim=256):
        super().__init__()
        self.lstm = nn.LSTM(input_size=input_dim, hidden_size=hidden_dim, batch_first=True, bidirectional=True)
        self.attn_pool = AttentionPooling(hidden_dim * 2)

    def forward(self, x):  # x: [Batch_size, T, D]
        output, _ = self.lstm(x)  # [B, T, H]
        context = self.attn_pool(output)  # [B, H]
        return context


# 投影头，将encoder输出映射到对比空间
class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim=512, output_dim=128):  # 参数：输入特征维度、中间隐藏层维度、输出特征维度
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),  # 第一个全连接层，将输入映射到隐藏层维度
            nn.LayerNorm(hidden_dim),  # LSTM更常用层归一化，对每个样本的所有特征维度归一化，加快收敛，稳定训练
            nn.ReLU(),
            nn.Dropout(0.2),  # 提升训练稳定性
            nn.Linear(hidden_dim, output_dim)  # 第二个全连接层，把隐藏特征映射到输出维度
        )

    def forward(self, x):
        return self.net(x)


# 对比学习模型
class SimSeq(nn.Module):
    def __init__(self):
        super().__init__()
        # 双向LSTM+attention机制
        self.encoder = SimSeqEncoder()  # 将注意力机制引入encoder中的LSTM
        self.projector = MLP(512, 512, 128)  # 输入、隐藏、输出维度

    def forward(self, x):
        z = self.projector(self.encoder(x))
        return F.normalize(z, dim=1)  # 将每个表示向量映射到单位球上，避免不同尺度影响相似度计算


# NT-Xent损失函数，最小化正样本对距离，最大化负样本（同batch中其他样本）对距离
def NT_Xent_loss(z1, z2, temperature=0.07):  # temperature用于调整softmax的平滑程度
    batch_size = z1.size(0)
    z = torch.cat([z1, z2], dim=0)  # [2*Batch_size, Dimension]
    z = F.normalize(z, dim=1)  # 对每个样本进行L2归一化，使得它们的向量模长为1

    # 相似度矩阵 [2B, 2B]，每个元素Sij表示第i个样本与第j个样本的相似度，且经过温度缩放
    similarity_matrix = torch.matmul(z, z.T) / temperature

    # 屏蔽对角线（自身相似度），避免模型把一个样本自身当作正样本，通过将对角线设置为极小值的方式，在softmax中效果等同于忽略
    mask = torch.eye(2 * batch_size, dtype=torch.bool).to(z.device)
    similarity_matrix.masked_fill_(mask, -1e9)

    # 构造正样本标签
    # 在相似度矩阵中：对于第i个样本，z1[i]的正样本是z2[i]，即labels[i]=i+B，z2[i]的正样本是z1[i]，即labels[i+B]=i
    labels = torch.arange(batch_size, device=z.device)
    # labels某一位的值指的是该位置的样本对应的正样本的索引位置
    labels = torch.cat([labels + batch_size, labels])  # z1 的正样本是 z2，z2 的正样本是 z1

    # 交叉熵损失：逐行计算损失，每一行的正样本是labels[i]，每一行的所有候选样本是similarity_matrix[i]
    loss = F.cross_entropy(similarity_matrix, labels)  # 每一行表示一个anchor与所有可能样本之间的相似度
    return loss


# 训练函数
def train_simseq(csv_file, transform_group, batch_size, epochs, patience):
    dataset = SimSeqDataset(csv_file, transform_group)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model = SimSeq().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)

    best_loss = float('inf')  # 最大化初始损失
    patience_counter = 0  # 损失无提升次数
    loss_history = []

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for x1, x2 in tqdm(dataloader, disable=True):
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
            torch.save(model.encoder.state_dict(), f'C:\\ETC_proj\\TLS_ETC\\SimSeq_Encoders\\SimSeqEncoder-{info}.pt')
            print(f"Encoder saved as SimSeqEncoder-{info}.pt")
        else:
            patience_counter += 1
        if patience_counter >= patience:
            print(f"Early stopping triggered after {patience} no-improvement epochs.")
            break

    plt.plot(range(1, len(loss_history) + 1), loss_history, label="Training Loss")
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('NT-Xent Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig(f"C:\\ETC_proj\\TLS_ETC\\NT_Xent_loss\\nt_xent_loss-{info}.png", bbox_inches='tight')
    plt.close()
 

if __name__ == '__main__':
    # 训练参数设置
    CSV_FILE = 'C:\\ETC_proj\\dataset_afterDivision\\pretrain.csv'  # 预训练数据集
    Batch_size = 2048  # 批大小，尽可能大会更好，受显存限制
    Epochs = 1000  # 训练轮次，理想趋近0为收敛
    Patience = 10  # 早停容忍次数
    # 场景选择/获取两个视图方案选择
    Transform_group = get_transform_group(mode)

    train_simseq(
        csv_file=CSV_FILE,
        batch_size=Batch_size,
        transform_group=Transform_group,
        epochs=Epochs,
        patience=Patience
    )