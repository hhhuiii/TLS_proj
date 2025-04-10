import csv
import ast


def extract_valid_ppi_last_row(ppi_str):
    """
    从CSV文件中提取指定列并保存为新文件
    PPI字段替换为仅保留最后一行

    """
    try:
        ppi_data = ast.literal_eval(ppi_str)
        if not isinstance(ppi_data, list) or len(ppi_data) != 3:
            return None
        for row in ppi_data:
            if not isinstance(row, list) or len(row) < 10:
                return None
        # 只保留最后一行数据
        return str(ppi_data[2])
    except Exception:
        return None


def filter_csv_columns(input_file, output_file, columns_to_keep):
    """
    从CSV文件中提取指定列并保存为新文件
    
    参数:
        input_file: 输入CSV文件路径
        output_file: 输出CSV文件路径
        columns_to_keep: 需要保留的列名列表
    """
    try:
        with open(input_file, 'r', newline='', encoding='utf-8') as infile, \
             open(output_file, 'w', newline='', encoding='utf-8') as outfile:
            
            reader = csv.DictReader(infile)
            
            # 检查请求的列是否存在于原始文件中
            available_columns = reader.fieldnames
            missing_columns = [col for col in columns_to_keep if col not in available_columns]
            
            if missing_columns:
                print(f"警告: 以下列不存在于原始文件中: {missing_columns}")
                # 只保留存在的列
                columns_to_keep = [col for col in columns_to_keep if col in available_columns]
            
            if not columns_to_keep:
                print("错误: 没有有效的列可保留")
                return
            
            writer = csv.DictWriter(outfile, fieldnames=columns_to_keep)
            writer.writeheader()
            
            filtered_count = 0
            written_count = 0

            for row in reader:
                # 只写入指定的列
                if 'PPI' in columns_to_keep:
                    ppi_last_row = extract_valid_ppi_last_row(row.get('PPI', ''))
                    if ppi_last_row is None:
                        filtered_count += 1
                        continue
                    row['PPI'] = ppi_last_row
                filtered_row = {col: row[col] for col in columns_to_keep}
                writer.writerow(filtered_row)
                written_count += 1
                
        print(f"成功处理文件，保留 {written_count} 条，过滤掉 {filtered_count} 条无效 PPI 数据。结果已保存到 {output_file}")
        
    except FileNotFoundError:
        print(f"错误: 文件 '{input_file}' 未找到")
    except Exception as e:
        print(f"发生错误: {e}")

# 参数设置
input_csv = "D:\\ETC_proj\\dataset\\flows.csv"          # 原始CSV文件路径
output_csv = "D:\\ETC_proj\\dataset\\filtered.csv"  # 输出文件路径
columns_to_keep = ['ID', 'PPI', 'PPI_LEN', 'APP', 'CATEGORY']  # 需要保留的列名

filter_csv_columns(input_csv, output_csv, columns_to_keep)