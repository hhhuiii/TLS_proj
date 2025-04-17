import csv
import random
import ast


LOSS_RATE = 0.2  # 丢包概率
FIX_LEN = 30  # 固定长度
MODE = 'RTO'  # 调用的增强算法：选择'RTO'或'FAST'


# ------------------ 增强算法 1：基于 RTO 的子序列复制 ------------------
def rto_augmentation(packet_sequence, p, L_min, L_max):
    """
    RTO
    模拟超时重传使得序列长度增加到固定长度，在网络较拥塞时起作用
    """
    M = []  # 增强后的序列
    T = []  # 暂存模拟连续丢包的子序列
    i = 0
    n = len(packet_sequence)

    while i < n:
        if random.random() < p:  # 发生丢包（网络可能产生拥塞）
            L = random.randint(L_min, L_max)  # 在丢包长度范围内随机取本次丢包长度
            end = min(i + L, n)  # 丢包序列即为位置
            subseq = packet_sequence[i:end]
            T.extend(subseq)      # 加入暂存重传序列
            M.extend(subseq)      # 模拟第一次传输的数据包到达（子序列重复与子序列移位的不同之处）
            i = end
        else:  # 成功发送
            M.append(packet_sequence[i])
            M.extend(T)           # 如果之前有重传序列，这时插入
            T = []                # 清空暂存区
            i += 1

    M.extend(T)                   # 最后可能还有重传未写入
    return M


# ------------------ 增强算法 2：基于 Fast Retransmit 的子序列复制 ------------------
def fast_retransmit_augmentation(packet_sequence, p):
    """
    Fast Retransmit
    模拟快速重传使得序列长度增加到固定长度，在网络不那么拥塞时起作用
    """
    class Packet:
        def __init__(self, value):
            self.value = value
            self.flag = 'unsent'  # 设置状态位

    O = [Packet(v) for v in packet_sequence]  # 为每个长度封装初始状态为'unsent'
    M = []

    while len(O) > 0:  # 模拟持续重传过程
        for i in range(len(O)):
            pkt = O[i]
            M.append(pkt.value)  # 模拟第一次传输成功
            if random.random() > p or pkt.flag == 'lost':  # 未丢包或为待重传包
                O.pop(i)  # 成功发送
                break
            else:
                pkt.flag = 'lost'  # 模拟丢包，下一次会立即重传
                break  # 丢包后立即进入while循环检查重传

    return M


# ------------------ 主逻辑：对CSV文件批量增强 PPI 字段 ------------------
def augment_ppi_from_csv(input_file, output_file, mode=MODE, p=LOSS_RATE, L_min=2, L_max=5):
    """
    对CSV中所有PPI字段进行数据增强并截断或填充至长度FIX_LEN
    options:
        input_file: str, 输入CSV路径，需包含PPI字段（为字符串形式的三行数组）
        output_file: str, 输出CSV路径
        mode: 'RTO' or 'FAST', 增强方式
        p: 丢包概率
        L_min: 模拟丢失的最小包数
        L_max: 模拟丢失的最大包数
    """
    with open(input_file, 'r', encoding='utf-8') as infile, \
         open(output_file, 'w', newline='', encoding='utf-8') as outfile:

        reader = csv.DictReader(infile)
        fieldnames = reader.fieldnames
        writer = csv.DictWriter(outfile, fieldnames=fieldnames)
        writer.writeheader()
        index = 0  # 行索引
        for row in reader:
            try:
                # 解析字符串形式的嵌套列表
                ppi = ast.literal_eval(row['PPI'])
                index += 1  # 增加行索引
                print(f"processing:{index}, PPI:{ppi}")  # 打印当前行索引和PPI字段
                
                length_seq = ppi  # 取最后一行作为长度序列

                # 循环执行增强算法，确保结果长度 ≥ FIX_LEN
                
                if mode == 'RTO':
                    augmented = rto_augmentation(length_seq, p, L_min, L_max)
                        
                elif mode == 'FAST':
                    augmented = fast_retransmit_augmentation(length_seq, p)

                # 处理为固定长度
                if len(augmented) > FIX_LEN:
                    augmented = augmented[:FIX_LEN]
                else:
                    augmented += [0] * (FIX_LEN - len(augmented))

                # 更新PPI字段
                row['PPI'] = str(augmented)
                writer.writerow(row)

            except Exception as e:
                print(f"跳过错误项: {e}")
                continue


if __name__ == "__main__":
    # 文件路径
    # 输入文件是经过样本数量增强后的文件
    # input_csv = "D:/ETC_proj/dataset/augmentedMoreSBase.csv"
    input_csv = "D:/ETC_proj/dataset/augmentedMoreS.csv"
    output_csv = "D:/ETC_proj/dataset/augmentedFixL.csv"

    # 使用 RTO 方式增强所有流的 PPI 字段
    augment_ppi_from_csv(input_csv, output_csv, mode=MODE, p=LOSS_RATE, L_min=2, L_max=4)

    print("固定长度增强处理完成，结果保存在:", output_csv)
