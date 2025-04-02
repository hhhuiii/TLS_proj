# -*- coding:utf-8 -*-

"""
v1增加的功能：按照IP对合并其中的UDP流，拆分策略变为按照IP对合并UDP流
但是SplitCap不支持自定义静默超时时间

批量拆分pcap文件的脚本使用示例，在split_cap_v1.py所在目录下运行
python split_pcap_v1.py
      D:\\ETC_proj\\TLS_ETC\\ISCX_dataset\\VPN-PCAPS
       --level flow
"""

# SplitCap.exe生成的拆分后的文件名格式通常包含以下信息：
# <原始文件名>.<协议>_<源IP>_<源端口>_<目的IP>_<目的端口>.pcap
# 即属于同一个流的所有数据包


import os
import argparse
import subprocess
from glob import glob
from scapy.all import rdpcap, wrpcap
from scapy.layers.inet import IP, UDP


def extract_label_from_filename(filename):
    # 从文件名中提取标签
    base_name = os.path.splitext(filename)[0] # 去掉.pcap后缀
    label_app = base_name.split("_")[0] # 提取标签部分（具体应用）
    # label_type = base_name.split("_")[1] # 提取标签部分（流量类型）
    return label_app


def is_udp_flow(file_path):
    # 判断一个PCAP文件是否属于UDP流量
    try:
        packets = rdpcap(file_path)
        ip_pairs = set()  # 存储不同的IP对
        for packet in packets:
            if UDP in packet and IP in packet:
                src_ip = packet[IP].src
                dst_ip = packet[IP].dst
                ip_pairs.add((src_ip, dst_ip))
        return ip_pairs

    except Exception as e:
        print(f"无法读取文件 {file_path}:{e}")
        return False


def split_cap(pcap_path, pcap_file, pcap_name, pcap_label='', dataset_level='flow'):
    # 创建拆分路径
    splitcap_dir = "D:\\ETC_proj\\TLS_ETC\\ISCX_dataset\\SplitByFlow"
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

    # 获取UDP流并按源和目的IP合并
    ip_pairs = set()
    for root, dirs, files in os.walk(output_path):
        for file in files:
            if file.endswith(".pcap"):
                file_path = os.path.join(root, file)
                ip_pairs.update(is_udp_flow(file_path))  # 将每个流的IP对加入到集合中

    # 按源IP和目的IP合并UDP流
    merged_flow_dir = os.path.join(splitcap_dir, pcap_name + "_merged")
    os.makedirs(merged_flow_dir, exist_ok=True)

    for ip_pair in ip_pairs:
        # 按照源IP和目的IP合并流量
        merged_file_path = os.path.join(merged_flow_dir, f"{ip_pair[0]}_{ip_pair[1]}.pcap")
        with open(merged_file_path, 'wb') as merged_file:
            for root, dirs, files in os.walk(output_path):
                for file in files:
                    if file.endswith(".pcap"):
                        file_path = os.path.join(root, file)
                        packets = rdpcap(file_path)
                        filtered_packets = [pkt for pkt in packets if IP in pkt and UDP in pkt and
                                            pkt[IP].src == ip_pair[0] and pkt[IP].dst == ip_pair[1]]
                        # 写入文件
                        if filtered_packets:
                            with open(merged_file_path, 'wb') as merged_file:
                                wrpcap(merged_file, filtered_packets)
                        else:
                            print(f"没有找到符合条件的数据包，跳过写入文件：{merged_file_path}")
    print(f"合并后的UDP流已保存到目录：{merged_flow_dir}")
    return merged_flow_dir


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
