"""
针对 tshark 拆分的 pcap 文件，统计每个 UDP 流（PCAP 文件）中的数据包数量，
仅保留 100 个数据包以内的流，并绘制直方图。
"""

import os
import matplotlib.pyplot as plt
from scapy.all import rdpcap
from collections import Counter

# 限制最大数据包数量
MAX_PACKET_LIMIT = 50

def count_packets_in_flows(flow_dir, max_packets=MAX_PACKET_LIMIT):
    """统计每个 UDP 流（PCAP 文件）中的数据包数量，仅保留 max_packets 以内的流"""
    packet_counts = []
    
    for root, _, files in os.walk(flow_dir):
        for file in files:
            if file.endswith(".pcap"):
                file_path = os.path.join(root, file)
                try:
                    packets = rdpcap(file_path)
                    packet_count = len(packets)

                    if packet_count <= max_packets:  # 仅保留 100 个数据包以内的流
                        packet_counts.append(packet_count)

                except Exception as e:
                    print(f"无法读取 {file_path}: {e}")
    
    return packet_counts

def plot_histogram(packet_counts):
    """绘制数据包数量的直方图"""
    count_distribution = Counter(packet_counts)
    sorted_counts = sorted(count_distribution.items())
    
    x_values = [item[0] for item in sorted_counts]  # 数据包数量
    y_values = [item[1] for item in sorted_counts]  # 具有该数据包数量的流的数量
    
    plt.figure(figsize=(10, 6))
    plt.bar(x_values, y_values, width=1.0, edgecolor='black', alpha=0.7, color='blue')
    plt.xlabel("数据包数量", fontsize=12)
    plt.ylabel("流的数量", fontsize=12)
    plt.title(f"数据包数量 ≤ {MAX_PACKET_LIMIT} 的 UDP 流分布", fontsize=14)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.show()

if __name__ == "__main__":
    flow_dir = "D:\\ETC_proj\\ISCX_dataset\\SplitByFlow"  # 修改为你的 UDP 流文件夹路径
    packet_counts = count_packets_in_flows(flow_dir)
    plot_histogram(packet_counts)