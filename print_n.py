import csv

# 输出csv文件的前n行

ROWS = 10  # 设置要打印的行数

def print_first_n_rows(csv_file_path):
    try:
        with open(csv_file_path, 'r', newline='', encoding='utf-8') as csvfile:
            reader = csv.reader(csvfile)
            
            # 打印前100行
            for i, row in enumerate(reader):
                if i >= ROWS:
                    break
                print(row)
                
    except FileNotFoundError:
        print(f"错误：文件 '{csv_file_path}' 未找到")
    except Exception as e:
        print(f"发生错误: {e}")

# 使用示例 - 替换为你的CSV文件路径
csv_file_path = "D:\\ETC_proj\\dataset\\augmentedFixL.csv"  # 修改为你的CSV文件路径
print_first_n_rows(csv_file_path)