# 划分训练集、验证集和测试集，同时对长度序列中的长度信息进行标准化
import csv
import ast
import random
from sklearn.model_selection import train_test_split

# 配置参数
FIX_LEN = 30  # 序列固定长度
STANDARDIZE_DIVISOR = 1460  # 标准化除数
LABEL_FIELD = 'CATEGORY'  # 按照标签字段划分三种数据集

def normalize_sequence(seq, fix_len=FIX_LEN, divisor=STANDARDIZE_DIVISOR):
    """标准化并调整序列长度"""
    # 标准化
    seq = [x / divisor for x in seq]
    # 调整长度
    if len(seq) > fix_len:
        return seq[:fix_len]
    else:
        return seq + [0.0] * (fix_len - len(seq))

def load_and_preprocess(csv_path):
    """加载CSV，返回样本列表和标签列表"""
    samples, labels = [], []
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                ppi_raw = ast.literal_eval(row['PPI'])
                ppi_last_line = ppi_raw if isinstance(ppi_raw[0], (int, float)) else ppi_raw[-1]
                norm_ppi = normalize_sequence(ppi_last_line)
                samples.append((norm_ppi, row))
                labels.append(row[LABEL_FIELD])
            except Exception as e:
                print(f"跳过错误样本: {e}")
    return samples, labels

def save_split_to_csv(split_data, output_file, fieldnames):
    with open(output_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for norm_ppi, row in split_data:
            row['PPI'] = str(norm_ppi)
            writer.writerow(row)

def split_and_save(input_csv, output_prefix, test_size=0.1, val_size=0.1, random_seed=42):
    samples, labels = load_and_preprocess(input_csv)

    # 拆分出测试集 stratify参数保证按照每个标签在整个数据集中的分布比例进行划分
    train_val, test = train_test_split(samples, test_size=test_size, stratify=labels, random_state=random_seed)

    # 再从 train_val 中拆出验证集
    train_val_labels = [row[LABEL_FIELD] for _, row in train_val]
    train, val = train_test_split(train_val, test_size=val_size, stratify=train_val_labels, random_state=random_seed)

    fieldnames = list(train[0][1].keys())

    save_split_to_csv(train, f'{output_prefix}\\_train.csv', fieldnames)
    save_split_to_csv(val, f'{output_prefix}\\_val.csv', fieldnames)
    save_split_to_csv(test, f'{output_prefix}\\_test.csv', fieldnames)

    print("数据划分和标准化完成。输出文件：")
    print(f"- 训练集: {output_prefix}\_train.csv")
    print(f"- 验证集: {output_prefix}\_val.csv")
    print(f"- 测试集: {output_prefix}\_test.csv")


if __name__ == "__main__":
    input_csv_path = "D:/ETC_proj/dataset/augmentedFixL.csv"  # 输入路径
    output_prefix = "D:/ETC_proj/dataset_afterProcess"  # 输出文件名前缀

    split_and_save(input_csv_path, output_prefix)