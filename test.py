import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from sklearn.preprocessing import LabelEncoder
import pandas as pd
import os
from tqdm import tqdm
import matplotlib.pyplot as plt
import umap  # encoder效果可视化相关
import seaborn as sns
from sklearn.decomposition import PCA  # 特征降维工具
from sklearn.metrics import classification_report, confusion_matrix

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

encoder_path = 'C:\\ETC_proj\\TLS_ETC\\simsiam_lstm_encoder.pt'
checkpoint = torch.load(encoder_path, map_location=device)
print(checkpoint.keys())