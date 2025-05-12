import random

# 丢包率 四种设置0.001（作为理想网络对照），0.01（FAST），0.02（FAST），0.05（RTO），0.1（RTO）
rate = 5  # 扰动效果放大倍率
P = 0.02 * rate  # 前两种设置对应场景mode：FAST，后两种设置对应场景mode：RTO
# RTO参数设置
L_MIN = 2  # 最小连续丢包数量
L_MAX = 5  # 最大连续丢包数量


# 场景 1：网络状况较拥塞：RTO

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


# 场景 2：网络状况不那么拥塞：FAST

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
def get_transform_group(scene):
    if scene == 'RTO':
        return (lambda x : rto_rep(x, p=P, lmin=L_MIN, lmax=L_MAX),
                lambda x : rto_shift(x, p=P, lmin=L_MIN, lmax=L_MAX))
    elif scene == 'FAST':
        return (lambda x : fast_retransmit_rep(x, p=P),
                lambda x : fast_retransmit_shift(x, p=P))
    else:
        raise ValueError("scene must be either 'RTO' or 'FAST'")