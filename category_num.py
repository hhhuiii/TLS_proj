import csv
from collections import Counter

def count_field_values(input_csv, field_name, output_txt):
    # 用于存储字段值出现次数的计数器
    field_counts = Counter()

    # 打开 CSV 文件并读取数据
    with open(input_csv, newline='', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile)  # 使用字典读取，每一行是一个字典
        for row in reader:
            # 获取指定字段的值并进行计数
            field_value = row.get(field_name)
            if field_value:
                field_counts[field_value] += 1

    # 将结果保存到 txt 文件中
    with open(output_txt, 'w', encoding='utf-8') as txtfile:
        for value, count in field_counts.items():
            txtfile.write(f"{value}: {count}\n")

    print(f"结果已保存到 {output_txt}")

# 示例：统计 CSV 文件中 'Category' 字段的不同值出现次数
# input_csv = "D:\\ETC_proj\\dataset_afterProcess\\base\\_test.csv"
input_csv = "D:\\ETC_proj\\dataset_afterProcess\\FixLBasewithMoreS\\_train.csv"  # CSV 文件路径
field_name = 'CATEGORY'  # 要统计的字段名称['ID', 'PPI', 'PPI_LEN', 'APP', 'CATEGORY']
output_txt = 'train.txt'  # 输出的文本文件路径

count_field_values(input_csv, field_name, output_txt)