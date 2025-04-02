import os
import matplotlib.pyplot as plt
from glob import glob
from scapy.all import rdpcap
from concurrent.futures import ThreadPoolExecutor


def count_packets_in_pcap(pcap_file):
    """计算一个PCAP文件中的数据包数量"""
    try:
        packets = rdpcap(pcap_file)
        return len(packets)
    except Exception as e:
        print(f"无法读取文件 {pcap_file}: {e}")
        return 0


def gather_packet_counts(input_dir, max_packet_count=500):
    """并行遍历目录中的每个PCAP文件，统计每个文件的数据包数量，并只保留数据包数量 <= max_packet_count 的流"""
    packet_counts = []

    # 获取所有PCAP文件路径
    pcap_files = [os.path.join(root, file) for root, dirs, files in os.walk(input_dir) for file in files if
                  file.endswith(".pcap")]

    # 使用线程池并行处理每个PCAP文件
    with ThreadPoolExecutor() as executor:
        # 获取每个PCAP文件的数据包数量
        all_counts = list(executor.map(count_packets_in_pcap, pcap_files))

    # 筛选出数据包数量 <= max_packet_count 的流
    packet_counts = [count for count in all_counts if count > 1 and count <= max_packet_count]

    return packet_counts


def plot_packet_count_distribution(packet_counts):
    """绘制数据包数量的分布图"""
    if not packet_counts:
        print("没有符合条件的流数据")
        return

    plt.figure(figsize=(10, 6))
    plt.hist(packet_counts, bins=15, edgecolor="black")  # 这里调整bins的数量
    plt.title("数据包数量 <= 500 的UDP流分布")
    plt.xlabel("数据包数量")
    plt.ylabel("流的数量")
    plt.grid(True)
    plt.show()


if __name__ == "__main__":
    # 设置拆分后的PCAP文件目录
    result_dir = "D:\\ETC_proj\\TLS_ETC\\ISCX_dataset\\VPN-PCAPS"  # 替换成你实际的路径

    # 获取数据包数量 <= 500 的流
    packet_counts = gather_packet_counts(result_dir, max_packet_count=500)

    if packet_counts:
        print(f"统计到 {len(packet_counts)} 个UDP流，其中数据包数量 <= 500")
        plot_packet_count_distribution(packet_counts)
    else:
        print("没有符合条件的UDP流")