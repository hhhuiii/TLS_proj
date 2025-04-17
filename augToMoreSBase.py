import csv
import ast
from typing import Dict

# 重采样增加少数类样本数量

# ------------ 配置 ------------ #
TARGET_FIELD = 'CATEGORY'                               # 筛选字段名
TARGET_VALUES = {'Remote Desktop', 'Virtual assistant'} # 需要增强的类别
AUG_PER_SAMPLE = {
    'Remote Desktop': 4,
    'Virtual assistant': 15
}
DEFAULT_AUG_SAMPLE = 1
PPI_FIELD = 'PPI'

# ------------ 重采样增强函数 ------------ #
def copy_based_augmentation(input_csv: str, output_csv: str):
    """
    根据设定比例复制目标类别样本，作为采用重采样方法增强样本数量的对照增强集
    """
    with open(input_csv, 'r', encoding='utf-8') as infile:
        reader = csv.DictReader(infile)
        fieldnames = reader.fieldnames
        rows = list(reader)

    augmented_rows = []

    for row in rows:
        category = row[TARGET_FIELD]
        if category in TARGET_VALUES:
            try:
                _ = ast.literal_eval(row[PPI_FIELD])  # 校验字段格式
                copies = AUG_PER_SAMPLE.get(category, DEFAULT_AUG_SAMPLE)
                # 添加原始样本 + 复制若干份
                augmented_rows.extend([row.copy() for _ in range(copies + 1)])
            except Exception as e:
                print(f"跳过无效PPI: {row[PPI_FIELD]}, 错误: {e}")
        else:
            augmented_rows.append(row)

    # 写入增强后的 CSV 文件
    with open(output_csv, 'w', encoding='utf-8', newline='') as outfile:
        writer = csv.DictWriter(outfile, fieldnames=fieldnames)
        writer.writeheader()
        for row in augmented_rows:
            row[PPI_FIELD] = str(row[PPI_FIELD])  # 转换为字符串
            writer.writerow(row)

    print(f"重采样增强完成，已保存至 {output_csv}，按设定比例复制样本")

# ------------ 路径配置并运行 ------------ #
if __name__ == "__main__":
    input_csv_path = "D:\\ETC_proj\\dataset\\filtered.csv"
    output_csv_path = "D:\\ETC_proj\\dataset\\augmentedMoreSBase.csv"
    copy_based_augmentation(input_csv_path, output_csv_path)