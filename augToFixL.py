# data_augmentation_pipeline.py
import csv
import random
import ast

# ------------------ 增强算法 1：基于 RTO 的子序列复制 ------------------
def rto_augmentation(packet_sequence, p, L_min, L_max):
    """
    RTO-based packet subsequence duplication augmentation
    模拟超时重传导致的分组重复
    """
    M = []  # 最终增强后的序列
    T = []  # 暂存连续丢包的子序列
    i = 0
    n = len(packet_sequence)

    while i < n:
        if random.random() < p:
            L = random.randint(L_min, L_max)
            end = min(i + L, n)
            subseq = packet_sequence[i:end]
            T.extend(subseq)      # 加入暂存重传序列
            M.extend(subseq)      # 第一次传输
            i = end
        else:
            M.append(packet_sequence[i])
            M.extend(T)           # 如果之前有重传序列，这时插入
            T = []                # 清空暂存区
            i += 1

    M.extend(T)                   # 最后可能还有重传未写入
    return M


# ------------------ 增强算法 2：基于 Fast Retransmit 的子序列复制 ------------------
def fast_retransmit_augmentation(packet_sequence, p):
    """
    Fast Retransmit-based augmentation
    模拟快速重传导致的重复数据包
    """
    class Packet:
        def __init__(self, value):
            self.value = value
            self.flag = 'unsent'

    O = [Packet(v) for v in packet_sequence]
    M = []

    while len(O) > 0:
        for i in range(len(O)):
            pkt = O[i]
            M.append(pkt.value)
            if random.random() > p or pkt.flag == 'lost':
                O.pop(i)  # 成功发送
                break
            else:
                pkt.flag = 'lost'  # 模拟丢包，下一次会立即重传
                break

    return M


# ------------------ 主逻辑：对CSV文件批量增强 PPI 字段 ------------------
def augment_ppi_from_csv(input_file, output_file, mode='rto', p=0.2, L_min=2, L_max=5):
    """
    对CSV中所有PPI字段进行数据增强并截断或填充为长度30
    
    :param input_file: str, 输入CSV路径，需包含PPI字段（为字符串形式的三行数组）
    :param output_file: str, 输出CSV路径
    :param mode: 'rto' or 'fast', 增强方式
    :param p: 丢包概率
    :param L_min: 模拟丢失的最小包数（用于RTO）
    :param L_max: 模拟丢失的最大包数（用于RTO）
    """
    with open(input_file, 'r', encoding='utf-8') as infile, \
         open(output_file, 'w', newline='', encoding='utf-8') as outfile:

        reader = csv.DictReader(infile)
        fieldnames = reader.fieldnames
        writer = csv.DictWriter(outfile, fieldnames=fieldnames)
        writer.writeheader()

        for row in reader:
            try:
                # 解析字符串形式的嵌套列表
                ppi = ast.literal_eval(row['PPI'])
                if len(ppi) != 3:
                    continue  # 格式异常跳过
                length_seq = ppi[-1]  # 取最后一行作为长度序列

                # 执行增强算法
                if mode == 'rto':
                    augmented = rto_augmentation(length_seq, p, L_min, L_max)
                elif mode == 'fast':
                    augmented = fast_retransmit_augmentation(length_seq, p)
                else:
                    raise ValueError("Unknown mode: choose 'rto' or 'fast'")

                # 截断或填充为固定长度30
                if len(augmented) > 30:
                    augmented = augmented[:30]
                else:
                    augmented += [0] * (30 - len(augmented))

                # 更新PPI字段
                row['PPI'] = str(augmented)
                writer.writerow(row)

            except Exception as e:
                print(f"跳过错误项: {e}")
                continue


if __name__ == "__main__":
    # 修改为你的实际文件路径
    input_csv = "D:/ETC_proj/dataset/filtered.csv"
    output_csv = "D:/ETC_proj/dataset/augmented_ppi.csv"

    # 使用 RTO 方式增强所有流的 PPI 字段
    augment_ppi_from_csv(input_csv, output_csv, mode='rto', p=0.3, L_min=2, L_max=4)

    print("增强处理完成，结果保存在:", output_csv)
