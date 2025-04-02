"""
使用 tshark 和 editcap 拆分 PCAP 文件，仅保留UDP流

"""

import os
import argparse
import subprocess
from glob import glob


def extract_label_from_filename(filename):
    base_name = os.path.splitext(filename)[0]
    return base_name.split("_")[0]  # 提取应用名作为标签


def get_udp_flows(pcap_file):
    """使用 tshark 提取 UDP 流的五元组"""
    cmd = [
        "tshark", "-r", pcap_file, "-T", "fields",
        "-e", "ip.src", "-e", "ip.dst", "-e", "udp.srcport", "-e", "udp.dstport",
        "-Y", "udp"
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        flows = set()
        for line in result.stdout.splitlines():
            fields = line.split("\t")
            if len(fields) == 4:
                flows.add(tuple(fields))  # (src_ip, dst_ip, src_port, dst_port)
        return flows
    except subprocess.CalledProcessError as e:
        print(f"tshark 解析 {pcap_file} 失败: {e}")
        return set()


def split_udp_flows(pcap_path, pcap_file, output_dir):
    """使用 tshark 拆分 UDP 流量"""
    flows = get_udp_flows(pcap_file)
    os.makedirs(output_dir, exist_ok=True)
    
    for i, (src_ip, dst_ip, src_port, dst_port) in enumerate(flows):
        flow_file = os.path.join(output_dir, f"{src_ip}_{src_port}_{dst_ip}_{dst_port}.pcap")
        cmd = [
            "tshark", "-r", pcap_file, "-w", flow_file,
            "-Y", f"udp && ip.src=={src_ip} && ip.dst=={dst_ip} && udp.srcport=={src_port} && udp.dstport=={dst_port}"
        ]
        try:
            subprocess.run(cmd, check=True)
            print(f"提取 UDP 流: {flow_file}")
        except subprocess.CalledProcessError as e:
            print(f"tshark 提取 UDP 失败: {e}")


def split_packets(flow_dir):
    """使用 editcap 拆分单个数据包"""
    for root, _, files in os.walk(flow_dir):
        for file in files:
            if file.endswith(".pcap"):
                file_path = os.path.join(root, file)
                output_path = os.path.join(root, "packet_%d.pcap")
                cmd = ["editcap", "-c", "1", file_path, output_path]
                try:
                    subprocess.run(cmd, check=True)
                    print(f"拆分单个数据包: {file_path}")
                    os.remove(file_path)  # 删除原流文件
                except subprocess.CalledProcessError as e:
                    print(f"editcap 失败: {e}")


def batch_split_pcap(pcap_path, dataset_level="flow"):
    """批量处理 PCAP"""
    pcap_files = glob(os.path.join(pcap_path, "*.pcap"))
    splitcap_dir = "D:\\ETC_proj\\ISCX_dataset\\SplitByFlow"
    os.makedirs(splitcap_dir, exist_ok=True)

    for pcap_file in pcap_files:
        pcap_name = os.path.splitext(os.path.basename(pcap_file))[0]
        label = extract_label_from_filename(pcap_name)
        output_dir = os.path.join(splitcap_dir, pcap_name)
        print(f"处理 {pcap_file}, 标签: {label}")

        split_udp_flows(pcap_path, pcap_file, output_dir)

        if dataset_level == "packet":
            split_packets(output_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="使用tshark和editcap按流或数据包拆分PCAP")
    parser.add_argument("pcap_path", help="PCAP文件目录")
    parser.add_argument("--level", help="拆分级别（flow or packet）", default="flow", choices=["flow", "packet"])
    args = parser.parse_args()

    batch_split_pcap(args.pcap_path, args.level)