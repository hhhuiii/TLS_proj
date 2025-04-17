import csv
import ast

FIX_LEN = 30  # 固定长度

# 使用0填充PPI字段直到序列长度达到指定的固定长度，由于PPI字段长度都是不足30的，因此以0填充每个PPI字段

def pad_ppi_sequence(ppi_seq, target_len=FIX_LEN):
    """
    将 PPI 序列填充至指定长度，不做增强，只补0
    """
    if len(ppi_seq) >= target_len:
        return ppi_seq[:target_len]
    else:
        return ppi_seq + [0] * (target_len - len(ppi_seq))


def pad_ppi_from_csv(input_file, output_file):
    """
    读取 CSV 并将每个 PPI 字段填充至固定长度（仅补0，不做增强）
    """
    with open(input_file, 'r', encoding='utf-8') as infile, \
         open(output_file, 'w', newline='', encoding='utf-8') as outfile:

        reader = csv.DictReader(infile)
        fieldnames = reader.fieldnames
        writer = csv.DictWriter(outfile, fieldnames=fieldnames)
        writer.writeheader()

        for row in reader:
            try:
                ppi = ast.literal_eval(row['PPI'])
                padded_ppi = pad_ppi_sequence(ppi)  # 填充0表示流的结束
                row['PPI'] = str(padded_ppi)
                writer.writerow(row)
            except Exception as e:
                print(f"跳过错误项: {e}")
                continue


if __name__ == "__main__":
    # 输入为进行数量增强后的样本，但未进行序列长度标准化
    # input_csv = "D:/ETC_proj/dataset/augmentedMoreSBase.csv"
    input_csv = "D:/ETC_proj/dataset/augmentedMoreS.csv"
    output_csv = "D:/ETC_proj/dataset/augmentedFixLBase.csv"

    pad_ppi_from_csv(input_csv, output_csv)

    print("固定长度对照处理完成，结果保存在:", output_csv)