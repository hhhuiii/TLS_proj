import csv
import random
import ast
from typing import List

# 配置增强逻辑的参数
TARGET_FIELD = 'CATEGORY'                               # 筛选字段名，用于选择需要增强的行
TARGET_VALUES = {'Remote Desktop', 'Virtual assistant'} # CATEGORY字段的值集合，只有这些值的行会被增强
AUG_PER_SAMPLE = {
    'Remote Desktop': 4,  # 对于 'Remote Desktop' 增强 4 次
    'Virtual assistant': 15  # 对于 'Virtual assistant' 增强 15 次
}# 每个样本生成的增强数据数量
DEFAULT_AUG_SAMPLE = 1                                  # 默认增强次数
PPI_FIELD = 'PPI'                                       # 要增强的字段名，存储的是需要处理的序列
AUG_METHOD = 'RTO'  # 增强方法选择: 'RTO' 或 'FAST'（超时重传或快速重传）
TARGET_LEN = None   # 可设为 30 或 None（None 表示不限制长度）# 在移位增强中只增加少数类样本数量

P = 0.2            # 丢包率

# RTO 算法参数
LMIN = 2           # 最小丢包长度
LMAX = 5           # 最大丢包长度

# RTO（超时重传）增强算法（通常在网络较为拥塞时起作用，因此丢包时大概率会连续丢若干个包）
def shift_rto_augmentation(packet_seq: List[int], p: float, lmin: int, lmax: int, target_len: int = None) -> List[int]:
    """
    使用 RTO 算法对数据包序列进行增强
    options:
        packet_seq: 原始数据包序列
        p: 丢包概率
        lmin: 最小丢包长度
        lmax: 最大丢包长度
        target_len: 目标序列长度（optional）
    return value:
        增强后的数据包序列
    """
    T, M = [], []  # T 用于暂存丢包段，M 是最终的增强序列
    i = 0
    while i < len(packet_seq):
        if random.random() < p:  # 根据丢包率决定是否丢包
            L = random.randint(lmin, lmax)  # 随机生成丢包长度
            L = min(L, len(packet_seq) - i)  # 确保丢包长度不超过剩余序列长度
            T.extend(packet_seq[i:i+L])  # 将丢包段加入暂存区
            i += L  # 处理整个序列长度
        else:  # 没有丢包（引起超时的因素可能不存在了），即假设重传成功
            M.append(packet_seq[i])  # 将当前包加入结果序列
            M.extend(T)  # 将暂存区的丢包段加入结果序列
            T = []  # 清空暂存区
            i += 1
    M.extend(T)  # 将剩余的丢包段加入结果序列

    # 如果设置了目标长度，调整结果序列长度
    if target_len is None:
        return M
    return M[:target_len] if len(M) > target_len else M + [0] * (target_len - len(M))


# 快速重传增强算法(通常在网络不那么拥塞时起作用)
def shift_fast_retransmit_augmentation(packet_seq: List[int], p: float, target_len: int = None) -> List[int]:
    """
    使用快速重传算法对数据包序列进行增强
    options:
        packet_seq: 原始数据包序列
        p: 丢包概率
        target_len: 目标序列长度（optional）
    return value:
        增强后的数据包序列
    """
    O = [{"val": val, "flag": "unsent"} for val in packet_seq]  # 初始化数据包状态为未发送
    M = []  # 最终的增强序列

    while len(O) > 0:  # 只要还有数据包未处理（处于unsent状态）
        for i in range(len(O)):
            if random.random() > p or O[i]['flag'] == 'lost':  # 根据概率决定是否丢包
                M.append(O[i]['val'])  # 将数据包加入结果序列，若状态为lost则表示重传成功
                O.pop(i)  # 从原始序列中移除
                break  # 模拟每一轮收发过程中只处理一个数据包的交互节奏
                        # 成功发送一个数据包后立即break进入while循环考虑列表O中lost状态数据包的重传操作
                        # 若想模拟多个数据包的交互，引入计数，计数满足n个ACK后 再break模拟收到多个ACK后重传的行为   
            else:
                O[i]['flag'] = 'lost'  # 标记数据包为丢失，等待下一次while循环重传

    # 如果设置了目标长度，调整结果序列长度
    if target_len is None:
        return M
    return M[:target_len] if len(M) > target_len else M + [0] * (target_len - len(M))


# 增强函数
def augment_ppi_sequence(ppi_seq: List[int]) -> List[int]:
    """
    根据配置的增强方法对 PPI 序列进行增强
    options:
        ppi_seq: 原始 PPI 序列
    return value:
        增强后的 PPI 序列
    """
    if AUG_METHOD == 'RTO':
        return shift_rto_augmentation(ppi_seq, p=P, lmin=LMIN, lmax=LMAX, target_len=TARGET_LEN)
    elif AUG_METHOD == 'FAST':
        return shift_fast_retransmit_augmentation(ppi_seq, p=P, target_len=TARGET_LEN)
    else:
        raise ValueError("AUG_METHOD 应为 'rto' 或 'fast'")


# ------------ CSV处理 ------------ #
def augment_csv(input_csv: str, output_csv: str):
    """
    对 CSV 文件中的数据进行增强，并将结果保存到新的 CSV 文件
    options:
        input_csv: 输入 CSV 文件路径
        output_csv: 输出 CSV 文件路径
    """
    augmented_rows = []  # 存储增强后的行

    # 读取输入 CSV 文件
    with open(input_csv, 'r', encoding='utf-8') as infile:
        reader = csv.DictReader(infile)
        fieldnames = reader.fieldnames  # 获取字段名
        rows = list(reader)  # 将所有行读取到列表中

    # 选择指定字段的行进行增强
    for row in rows:
        if row[TARGET_FIELD] in TARGET_VALUES:  # 判断行是否符合筛选条件
            try:
                ppi_seq = ast.literal_eval(row[PPI_FIELD])  # 将 PPI 字段解析为列表
                # 获取当前字段值对应的增强次数
                aug_count = AUG_PER_SAMPLE.get(row[TARGET_FIELD], DEFAULT_AUG_SAMPLE)  # 默认增强一次
                for _ in range(aug_count):  # 根据配置生成多个增强样本
                    new_ppi = augment_ppi_sequence(ppi_seq)  # 调用增强函数
                    new_row = row.copy()  # 复制原始行
                    new_row[PPI_FIELD] = new_ppi  # 替换增强后的 PPI 字段
                    augmented_rows.append(new_row)  # 添加到结果列表
            except Exception as e:
                print(f"跳过无效PPI: {row[PPI_FIELD]}, 错误: {e}")  # 捕获解析错误并跳过
        else:
            augmented_rows.append(row)  # 不符合条件的行直接添加到结果列表

    # 将增强后的数据写入输出 CSV 文件
    with open(output_csv, 'w', encoding='utf-8', newline='') as outfile:
        writer = csv.DictWriter(outfile, fieldnames=fieldnames)
        writer.writeheader()  # 写入表头
        for row in augmented_rows:
            # 确保以字符串格式写入列表字段
            row[PPI_FIELD] = str(row[PPI_FIELD])
            writer.writerow(row)

    print(f"RTO/FAST增强完成，已保存至 {output_csv}，使用的增强方法: {AUG_METHOD}")


# ------------ 路径配置并运行 ------------ #
if __name__ == "__main__":
    # 输入和输出 CSV 文件路径
    input_csv_path = "D:\\ETC_proj\\dataset\\filtered.csv"
    output_csv_path = "D:\\ETC_proj\\dataset\\MoreS.csv"
    augment_csv(input_csv_path, output_csv_path)  # 调用增强函数