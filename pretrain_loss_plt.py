import torch
import torch.nn as nn
from matplotlib import pyplot as plt

# print(torch.cuda.is_available())
# if torch.cuda.is_available():
#     print(f"当前 CUDA 设备: {torch.cuda.get_device_name(0)}")
#     print(f"CUDA 版本: {torch.version.cuda}")

# 为什么据此判断其已经收敛，我的依据是损失绝对值的下降幅度几乎是损失的万分之一，且已经相当接近损失的最小值，因此判断其已经收敛
pretrainLoss_history = [-0.9876, -0.9882, -0.9926, -0.9930, -0.9932, -0.9933, -0.9942, -0.9949, -0.9969, -0.9971, -0.9971, -0.9972, -0.9972, -0.9973, -0.9973,
                        -0.9973, -0.9974, -0.9974, -0.9974, -0.9975, -0.9975, -0.9976, -0.9976, -0.9977, -0.9977, -0.9978, -0.9978, -0.9979, -0.9979, -0.9980,
                        -0.9981, -0.9981, -0.9982, -0.9982, -0.9983, -0.9983, -0.9985, -0.9985, -0.9986, -0.9986]

plt.plot(range(1, len(pretrainLoss_history) + 1), pretrainLoss_history, label="Training Loss")
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title(f"Loss Curve--epoch=40")
plt.legend()
plt.grid(True)
plt.show()