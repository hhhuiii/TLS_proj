"""
处理 UDP PCAP 文件，去除重复数据包并筛选出符合条件的流

"""

import os
from scapy.layers.inet import IP, UDP
from scapy.all import rdpcap, wrpcap
from hashlib import sha256


def remove_duplicate_packets(packets):
    """去除重复的数据包"""
    seen = set()  # 存储已见过的数据包哈希值
    unique_packets = []  # 存储去重后的数据包

    for packet in packets:
        if UDP in packet and IP in packet:
            # 计算数据包的唯一标识（五元组+载荷哈希）
            payload = bytes(packet[UDP].payload)
            packet_hash = sha256(payload).hexdigest()
            packet_id = (
                packet[IP].src,  # 源IP
                packet[UDP].sport,  # 源端口
                packet[IP].dst,  # 目的IP
                packet[UDP].dport,  # 目的端口
                packet_hash  # 载荷哈希
            )
            if packet_id not in seen:
                seen.add(packet_id)
                unique_packets.append(packet)

    return unique_packets


def sort_packets_by_timestamp(packets):
    """按时间戳排序数据包"""
    return sorted(packets, key=lambda p: p.time)


def filter_udp_streams_by_threshold(input_file, output_file, min_packets_threshold=10):
    """
    处理 UDP PCAP 文件：
    1. 读取 pcap，提取 UDP 数据包
    2. 过滤出 UDP 数据包数量 >= min_packets_threshold 的流
    3. 按时间戳排序并写入新的 pcap
    """

    # 读取 PCAP 文件
    packets = rdpcap(input_file)

    # 去重
    unique_packets = remove_duplicate_packets(packets)

    # 统计 UDP 数据包数量
    udp_packets = [pkt for pkt in unique_packets if UDP in pkt]

    if len(udp_packets) < min_packets_threshold:
        print(f"跳过 {input_file}，因 UDP 数据包数 {len(udp_packets)} < {min_packets_threshold}")
        return False  # 过滤掉不符合条件的 pcap 文件

    # 按时间戳排序
    sorted_packets = sort_packets_by_timestamp(udp_packets)

    # 保存结果
    wrpcap(output_file, sorted_packets)
    print(f"保存筛选后的 UDP 数据包 ({len(sorted_packets)} 个) 到 {output_file}")

    return True  # 处理成功


def batch_process_udp_pcap(input_dir, output_dir, min_packets_threshold=20):
    """批量处理 UDP PCAP 文件，只保留数据包数 ≥ min_packets_threshold 的流"""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    processed_files = 0  # 统计符合条件的流数量

    for root, dirs, files in os.walk(input_dir):
        for file in files:
            if file.endswith(".pcap"):
                input_file = os.path.join(root, file)
                output_file = os.path.join(output_dir, file)

                print(f"正在处理文件：{input_file}")

                if filter_udp_streams_by_threshold(input_file, output_file, min_packets_threshold):
                    processed_files += 1  # 计数符合条件的流

    print(f"\n筛选后符合 UDP 数据包数 ≥ {min_packets_threshold} 的流文件总数：{processed_files}")


if __name__ == "__main__":
    re_input_dir = "D:\\ETC_proj\\TLS_ETC\\ISCX_dataset\\SplitByFlow"
    re_output_dir = "D:\\ETC_proj\\TLS_ETC\\ISCX_dataset\\ProcessedUDP"

    min_packets_threshold = 20  # 设置 UDP 流的最小数据包数阈值
    batch_process_udp_pcap(re_input_dir, re_output_dir, min_packets_threshold)