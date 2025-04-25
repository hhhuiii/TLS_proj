import torch
import torch.nn as nn

print(torch.cuda.is_available())
if torch.cuda.is_available():
    print(f"当前 CUDA 设备: {torch.cuda.get_device_name(0)}")
    print(f"CUDA 版本: {torch.version.cuda}")