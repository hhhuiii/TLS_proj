import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt


# 场景选择
mode = 'RTO'  # 'RTO'或'FAST'

device = torch.device("cuda" if torch.cude.is_available() else "cpu")

# 丢包率
P = 0.2
# RTO参数设置
L_MIN = 2  # 最小连续丢包数量
L_MAX = 5  # 最大连续丢包数量

# LSTM输入定长设置
MAX_LEN = 35


# 场景 1：网络状况较拥塞

# 扰动算法 1：基于 RTO 的子序列重复
def rto_rep(packet_seq, p, lmin, lmax):
    res = []  # 扰动后的返回序列
    retrans = []  # 待重传序列
    i = 0
    n = len(packet_seq)

    while i < n:  # 只要还有未重传/发送数据包
        if random.random() < p:  # 发生丢包
            L = random.randint(lmin, lmax)  # 随机确定丢包序列长度
            end = min(i + L, n)  # 丢包序列尾部小于初始序列尾部
            subseq = packet_seq[i:end]  # 截取丢包序列
            retrans.extend(subseq)  # 丢包序列加入到待重传序列
            res.extend(subseq)  # 假定是在丢包点上游捕获的流量，如此会发生子序列重复捕获
            i = end
        else:
            res.append(packet_seq[i])  # 未发生丢包
            res.extend(retrans)  # 网络情况好转，重传成功
            retrans = []  # 清空重传序列
            i += 1  # 继续判定下一个包是否会丢包

    res.extend(retrans)
    return res


# 扰动算法 2：基于 RTO 的子序列移位
def rto_shift(packet_seq, p, lmin, lmax):
    retrans = []  # 暂存重传序列
    res = []  # 返回的结果序列
    i = 0
    while i < len(packet_seq):
        if random.random() < p:  # 未发生丢包
            L = random.randint(lmin, lmax)  # 随机确定连续丢包个数
            L = min(L, len(packet_seq) - i)
            retrans.extend(packet_seq[i:i+L])  # 将丢失子序列加入重传序列中，但是捕获点在丢包点下游，不将之加入结果序列中
            i += L
        else:
            res.append(packet_seq[i])  # 未发生丢包
            res.extend(retrans)  # 重传也成功
            retrans = []
            i += 1

    res.extend(retrans)
    return res


# 场景 2：网络状况不那么拥塞

# 扰动算法 3：基于 Fast Retransmit 的子序列重复
def fast_retransmit_rep(packet_seq, p):
    class Packet:
        def __init__(self, value):
            self.value = value
            self.flag = 'unsent'

    ori = [Packet(v) for v in packet_seq]  # 初始序列全部初始化为未发送
    res = []  # 返回序列

    while len(ori) > 0:  # 只要还有未发送数据包
        for i in range(len(ori)):
            pkt = ori[i]
            res.append(pkt.value)  # 捕获点在丢包点上游，会重复收到，首先加入结果序列
            if random.random() > p or pkt.flag == 'lost':  # 若未发生丢包或该包已经丢过一次，重传成功（假定第二次重传就能够成功）
                ori.pop(i)  # 该包发送成功
                break  # 通过break控制只要一个重复ACK就能出发重传（对应的过程是成功发送了一个数据包）
            else:
                pkt.flag = 'lost'  # 发生丢包，将当前包标记为丢包
                break  # 模拟单次ACK就发生重传

    return res


# 扰动算法 4：基于 Fast Retransmit 的子序列移位
def fast_retransmit_shift(packet_seq, p):
    ori = [{"val": val, "flag": "unsent"} for val in packet_seq]  # 初始化全部序列为未发送
    res = []  # 返回序列

    while len(ori) > 0:
        for i in range(len(ori)):
            if random.random() > p or ori[i]['flag'] == 'lost':  # 若未发生丢包或当前包已经丢过一次，发送成功
                res.append(ori[i]['val'])  # 捕获点在丢包点下游，因此只有在发送成功时才将数据包加入到结果序列中
                ori.pop(i)
                break
            else:
                ori[i]['flag'] = 'lost'  # 发生丢包，将当前包标记为已丢失过一次
                break  # 立即回到while循环尝试重传

    return res


# 根据使用的场景选择根据原始样本生成两个视图的方法
def get_transform_group(scene=mode):
    if scene == 'RTO':
        return (lambda x : rto_rep(x, p=P, lmin=L_MIN, lmax=L_MAX),
                lambda x : rto_shift(x, p=P, lmin=L_MIN, lmax=L_MAX))
    elif scene == 'FAST':
        return (lambda x : fast_retransmit_rep(x, p=P),
                lambda x : fast_retransmit_shift(x, p=P))
    else:
        raise ValueError("scene must be either 'RTO' or 'FAST'")


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


# SimSiam投影头/预测头
class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim=512, output_dim=128):  # 参数：输入特征维度、中间隐藏层维度、输出特征维度
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),  # 第一个全连接层，将输入映射到隐藏层维度
            nn.BatchNorm1d(hidden_dim),  # 批归一化，加快收敛，稳定训练
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)  # 第二个全连接层，把隐藏特征映射到输出维度
        )

    def forward(self, x):
        return self.net(x)


# SimSiam模型，encoder为将视图定长后作为输入的LSTM
class SimSiam(nn.Module):
    def __init__(self, lstm_input_dim=1, lstm_hidden_dim=256):  # 参数：输入序列的特征维度（1表示每个时间步只有一个长度值）、LSTM的隐藏层维度，控制编码后的表示维度
        super().__init__()
        self.encoder = nn.LSTM(input_size=lstm_input_dim, hidden_size=lstm_hidden_dim, batch_first=True)  # LSTM编码器，batch_first指示输入形状为（batch_size,seq_len,input_size）
        # LSTM对一个序列进行编码，返回所有时间步的输出以及最后一个时间步的隐藏状态
        self.projector = MLP(lstm_hidden_dim)  # 投影头：把编码器输出的向量投影到另一个特征空间 
        self.predictor = MLP(lstm_hidden_dim)  # 预测头：预测另一个视图的投影向量

    def forward(self, x1, x2):
        _, (h1, _) = self.encoder(x1)  # h1和h2是两个视图经过LSTM后的最后一个时间步的隐藏状态，形状为（1,batch_size,hidden_dim）
        _, (h2, _) = self.encoder(x2)
        # 投影头
        z1 = self.projector(h1.squeeze(0))  # 降维至形状（Batch_size, hidden_dim）后投影至新的特征空间
        z2 = self.projector(h2.squeeze(0))
        # 预测头
        p1 = self.predictor(z1)  # 让某一个预测接近另一个或反之
        p2 = self.predictor(z2)
        return p1, z2.detach(), p2, z1.detach()  # 视图1的预测向量、视图2的投影向量（不参与梯度更新）、视图2的预测向量、视图1的投影向量（不参与梯度更新）


# 负余弦相似度损失函数
def simsiam_loss(p1, z2, p2, z1):
    def D(p, z):
        return -F.cosine_similarity(p, z.detach(), dim=-1).mean()
    return D(p1, z2) / 2 + D(p2, z1) / 2


# 预训练函数
def train_simsiam(csv_file, transform_group, batch_size, epochs, lr):
    dataset = SimSiamDataset(csv_file, transform_group)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model = SimSiam().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    best_loss = float('inf')
    patience_counter = 0
    patience = 10

    loss_history = []  # 记录每个epoch的loss

    model.train()  # 开启训练模式

    for epoch in range(epochs):
        total_loss = 0
        for x1, x2 in tqdm(dataloader):
            x1, x2 = x1.to(device), x2.to(device)
            optimizer.zero_grad()
            p1, z2, p2, z1 = model(x1, x2)
            loss = simsiam_loss(p1, z2, p2, z1)  # 计算单个损失
            loss.backward()  # 反向传播
            optimizer.step()  # 执行梯度下降
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
            print("Early stop for non-improvment {patience} times")
            break

    # 绘制损失曲线
    plt.plot(range(1, len(loss_history) + 1), loss_history, label="Training Loss")
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Loss Curve')
    plt.legend()
    plt.grid(True)
    plt.show()

if __name__ == '__main__':
    # 训练参数设置
    CSV_FILE = 'C:\\ETC_proj\\dataset_afterDivision\\pretrain.csv'  # 预训练数据集
    Batch_size = 512  # 批大小
    Epochs = 100  # 训练轮次
    Learning_rate = 1e-3  # 学习率
    # 场景选择/获取两个视图方案选择
    Transform_group = get_transform_group(mode)

    train_simsiam(
        csv_file=CSV_FILE,
        batch_size=Batch_size,
        transform_group=Transform_group,
        epochs=Epochs,
        lr=Learning_rate
    )