import csv
import ast
import os
from sklearn.model_selection import train_test_split

# 配置参数
STANDARDIZE_DIVISOR = 1460  # 标准化除数
LABEL_FIELD = 'CATEGORY'  # 标签字段


def normalize_sequence(seq, divisor=STANDARDIZE_DIVISOR):
    """只做标准化，不调整序列长度"""
    return [x / divisor for x in seq]


def load_and_preprocess(csv_path):
    """加载CSV并返回样本和标签"""
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
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for norm_ppi, row in split_data:
            row['PPI'] = str(norm_ppi)
            writer.writerow(row)


def split_for_pretrain_and_finetune(input_csv, output_prefix, pretrain_ratio=0.8, random_seed=42):
    samples, labels = load_and_preprocess(input_csv)

    # 划分预训练集和微调集
    pretrain, finetune = train_test_split(
        samples, test_size=(1 - pretrain_ratio), stratify=labels, random_state=random_seed
    )

    print(f"预训练集大小: {len(pretrain)}")
    print(f"微调集大小: {len(finetune)}")

    # 微调集再划分为 train/val/test（按照8:1:1的比例）
    finetune_labels = [row[LABEL_FIELD] for _, row in finetune]
    ft_trainval, ft_test = train_test_split(
        finetune, 
        test_size=0.1, 
        stratify=finetune_labels, 
        random_state=random_seed
    )

    # 再从剩下的90%中拿出10%的val集
    ft_trainval_labels = [row[LABEL_FIELD] for _, row in ft_trainval]
    ft_train, ft_val = train_test_split(
        ft_trainval, 
        test_size=1/9,  # 因为剩下的是90%，所以拿出其中的1/9就是10%
        stratify=ft_trainval_labels, 
        random_state=random_seed
    )

    fieldnames = list(pretrain[0][1].keys())

    # 保存预训练集
    save_split_to_csv(pretrain, f'{output_prefix}/pretrain.csv', fieldnames)

    # 微调集输出子目录
    ft_output_dir = os.path.join(output_prefix, '_finetune_split')
    save_split_to_csv(ft_train, os.path.join(ft_output_dir, 'train.csv'), fieldnames)
    save_split_to_csv(ft_val, os.path.join(ft_output_dir, 'val.csv'), fieldnames)
    save_split_to_csv(ft_test, os.path.join(ft_output_dir, 'test.csv'), fieldnames)

    print("数据划分完成：")
    print(f"- 预训练集: {output_prefix}_pretrain.csv")
    print(f"- 微调训练集: {ft_output_dir}/train.csv")  # 8:1:1
    print(f"- 微调验证集: {ft_output_dir}/val.csv")
    print(f"- 微调测试集: {ft_output_dir}/test.csv")


if __name__ == "__main__":
    input_csv_path = "C:/ETC_proj/dataset/filtered.csv"
    output_prefix = "C:/ETC_proj/dataset_afterDivision"

    split_for_pretrain_and_finetune(input_csv_path, output_prefix)