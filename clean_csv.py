import csv
import ast
import random
from collections import defaultdict


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


def filter_csv_columns_with_sampling(input_file, output_file, columns_to_keep, category_column='CATEGORY', sampling_ratios=None, random_seed=42):
    """
    从CSV文件中提取指定列并保存为新文件，支持按指定类别自定义采样比例设置保留的样本数量
    
    options:
        input_file: 输入CSV文件路径
        output_file: 输出CSV文件路径
        columns_to_keep: 需要保留的列名列表
        catagory_column: 类别字段名
        sampling_ratios: 字典，指定每个类别的采样比例,eg:{'Video': 0.2, 'Social': 0.5, 'Chat': 1.0}
            没有指定类别采样比例时自动生成采样比例
        random_seed: 随机种子，用于可重复的采样结果
    """
    try:
        random.seed(random_seed)
        
        # 第一阶段：先读取所有数据并统计类别分布
        category_counts = defaultdict(int)
        all_rows = []
        
        with open(input_file, 'r', newline='', encoding='utf-8') as infile:
            reader = csv.DictReader(infile)
            
            # 检查列是否存在
            available_columns = reader.fieldnames
            missing_columns = [col for col in columns_to_keep if col not in available_columns]
            
            if missing_columns:
                print(f"警告: 以下列不存在于原始文件中: {missing_columns}")
                columns_to_keep = [col for col in columns_to_keep if col in available_columns]
            
            if not columns_to_keep:
                print("错误: 没有有效的列可保留")
                return
                
            if category_column not in available_columns:
                print(f"错误: 类别字段 '{category_column}' 不存在")
                return
            
            # 读取所有行并统计类别
            for row in reader:
                all_rows.append(row)
                category_counts[row[category_column]] += 1
        
        # 如果没有提供采样比例，自动生成动态比例
        if sampling_ratios is None:
            sampling_ratios = {}
            total_samples = sum(category_counts.values())
            for category, count in category_counts.items():
                if count > total_samples * 0.1:  # 高频类别（>10%）
                    sampling_ratios[category] = 0.2  # 保留20%
                elif count > total_samples * 0.01:  # 中频类别（1%-10%）
                    sampling_ratios[category] = 0.5  # 保留50%
                else:  # 低频类别
                    sampling_ratios[category] = 1.0  # 保留100%   
        
        # 第二阶段：按类别采样并写入
        filtered_count = 0
        written_count = 0
        category_written = defaultdict(int)
        
        with open(output_file, 'w', newline='', encoding='utf-8') as outfile:
            writer = csv.DictWriter(outfile, fieldnames=columns_to_keep)
            writer.writeheader()
            
            for row in all_rows:
                category = row[category_column]
                ratio = sampling_ratios.get(category, 1.0)  # 默认保留
                
                # 决定是否保留该行
                if random.random() <= ratio:
                    # 处理PPI字段
                    if 'PPI' in columns_to_keep:
                        ppi_last_row = extract_valid_ppi_last_row(row.get('PPI', ''))
                        if ppi_last_row is None:
                            filtered_count += 1
                            continue
                        row['PPI'] = ppi_last_row
                    
                    # 写入过滤后的行
                    filtered_row = {col: row[col] for col in columns_to_keep}
                    writer.writerow(filtered_row)
                    written_count += 1
                    category_written[category] += 1
                else:
                    filtered_count += 1
        
        # 打印统计信息
        print(f"成功处理文件，保留 {written_count} 条，过滤掉 {filtered_count} 条")
        print("\n类别分布统计:")
        print(f"{'类别':<20} {'原始数量':>10} {'保留数量':>10} {'保留比例':>10}")
        for category, count in category_counts.items():
            kept = category_written.get(category, 0)
            ratio = kept / count if count > 0 else 0
            print(f"{category:<20} {count:>10} {kept:>10} {ratio:>10.1%}")
        
    except FileNotFoundError:
        print(f"错误: 文件 '{input_file}' 未找到")
    except Exception as e:
        print(f"发生错误: {e}")

# 参数设置
input_csv = "C:\\ETC_proj\\dataset\\flows.csv"     # 原始CSV文件路径
output_csv = "C:\\ETC_proj\\dataset\\filtered.csv" # 输出文件路径
columns_to_keep = ['PPI', 'PPI_LEN', 'CATEGORY']  # 需要保留的列名：数据包长度序列、序列原始长度、标签


custom_ratios = {  # 下采样比例
    'Videoconferencing' : 0.2,
    'Streaming media' : 0.1,
    'Software updates' : 0.2,
    'Social' : 0.12,
    'Analytics & Telemetry' : 0.12,
    'Other services and APIs' : 0, # 去掉此类
    'Instant messaging' : 0.6,
    'Search' : 0.4,
    'Music' : 0.24,
    'Weather services' : 0.4,
    'Advertising' : 0.08,
    'Information Systems' : 1,
    'Authentication services' : 0.2,
    'File sharing' : 0.2,
    'Antivirus' : 0.2,
    'Mail' : 0.2,
    'Games' : 0.4,
    'Notification services' : 0.4,
    'Remote Desktop' : 1.0,
    'Internet Banking' : 1.0,
    'Virtual assistant' : 1.0,
}  # 自定义采样比例

filter_csv_columns_with_sampling(input_csv, output_csv, columns_to_keep, sampling_ratios=custom_ratios, random_seed=42)