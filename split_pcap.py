# -*- coding:utf-8 -*-

"""
批量拆分pcap文件的脚本使用示例，在split_cap.py所在目录下运行
python split_pcap.py
      D:\\ETC_proj\\ISCX_dataset\\VPN-PCAPS
       --level flow
"""

# SplitCap.exe生成的拆分后的文件名格式通常包含以下信息：
# <原始文件名>.<协议>_<源IP>_<源端口>_<目的IP>_<目的端口>.pcap
# 即属于同一个流的所有数据包

import os
import argparse
import subprocess
from glob import glob
from scapy.all import rdpcap, UDP


def extract_label_from_filename(filename):
    # 从文件名中提取标签
    base_name = os.path.splitext(filename)[0] # 去掉.pcap后缀
    label_app = base_name.split("_")[0] # 提取标签部分（具体应用）
    # label_type = base_name.split("_")[1] # 提取标签部分（流量类型）
    return label_app

# def is_udp_flow(file_path):
#     # 判断一个PCAP文件是否属于UDP流量
#     try:
#         packets = rdpcap(file_path)
#         for packet in packets:
#             if UDP in packet:
#                 return True
#         return False

#     except Exception as e:
#         print(f"无法读取文件 {file_path}:{e}")
#         return False

def split_cap(pcap_path, pcap_file, pcap_name, pcap_label='', dataset_level='flow'):
    # 创建拆分路径
    splitcap_dir = "D:\\ETC_proj\\ISCX_dataset\\SplitByFlow"
    if not os.path.exists(splitcap_dir):
        os.mkdir(splitcap_dir)

    output_path = os.path.join(splitcap_dir, pcap_name)

    # 创建输出目录
    os.makedirs(output_path, exist_ok=True)

    # 按流（具相同五元组的数据包集合）拆分pcap文件
    if dataset_level == 'flow':
        cmd = ["D:\\ETC_proj\\TLS_ETC\\SplitCap.exe", "-r", pcap_file, "-s", "flow", "-o", output_path]
    # 按单个数据包拆分pcap文件
    elif dataset_level == 'packet':
        cmd = ["D:\\ETC_proj\\TLS_ETC\\SplitCap.exe", "-r", pcap_file, "-s", "packets", "1", "-o", output_path]
    else:
        raise ValueError("Invalid dataset_level. Must be 'flow' or 'packet'.")

    # 执行拆分命令
    try:
        subprocess.run(cmd, check=True)
        print(f"PCAP文件已成功拆分到目录:{output_path}")
    except subprocess.CalledProcessError as e:
        print(f"拆分失败:{e}")
    except FileNotFoundError:
        print("SplitCap.exe工具未找到，检查路径是否正确")
        return None

    # 筛选UDP流
    # udp_files = 0
    # for root, dirs, files in os.walk(output_path):
    #     for file in files:
    #         if file.endswith(".pcap"):
    #             file_path = os.path.join(root, file)
    #             if is_udp_flow(file_path):
    #                 # 判断为UDP流量
    #                 udp_files += 1
    #             else:
    #                 os.remove(file_path)
    #                 print(f"删除非UDP流量文件：{file_path}")
    # print(f"保留{udp_files}个UDP文件")

    return output_path


def batch_split_cap(pcap_path, pcap_label='', dataset_level='flow'):
    # 获取文件夹中的所有pcap文件
    pcap_files = glob(os.path.join(pcap_path, "*.pcap"))
    for pcap_file in pcap_files:
        pcap_name = os.path.splitext(os.path.basename(pcap_file))[0]
        pcap_label = extract_label_from_filename(pcap_name)

        print(f"正在处理文件:{pcap_file}，提取的标签：{pcap_label}")
        split_cap(pcap_path, pcap_file, pcap_name, pcap_label, dataset_level)


if __name__ == "__main__":
    # 设置命令行参数
    parser = argparse.ArgumentParser(description="按流或数据包拆分PCAP文件，自动从文件名中提取标签")
    parser.add_argument("pcap_path", help="PCAP文件所在目录")
    parser.add_argument("--level", help="拆分级别（flow or packet）", default="flow", choices=["flow", "packet"])
    args = parser.parse_args()

    # 调用批量拆分函数
    batch_split_cap(args.pcap_path, args.level)