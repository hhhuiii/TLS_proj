import numpy as np
import torch
import random
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
from sklearn.decomposition import PCA
import umap


# 可视化部分
def visualize_features(raw_inputs, features, labels, label_encoder, output_dir, max_per_class=2):
    # 转换为 numpy 格式
    raw_inputs = torch.cat(raw_inputs, dim=0).numpy()
    features = torch.cat(features, dim=0).numpy()
    labels = np.array(labels)

    # 每类最多保留 max_per_class 个样本用于可视化
    index_by_class = defaultdict(list)
    for idx, label in enumerate(labels):
        index_by_class[label].append(idx)

    selected_indices = []
    for label, indices in index_by_class.items():
        random.shuffle(indices)
        selected_indices.extend(indices[:max_per_class])

    raw_inputs_vis = raw_inputs[selected_indices]
    features_vis = features[selected_indices]
    labels_vis = labels[selected_indices]

    n_components_raw = min(30, raw_inputs_vis.shape[0], raw_inputs_vis.shape[1])
    n_components_feat = min(30, features_vis.shape[0], features_vis.shape[1])

    # 原始输入PCA降维后UMAP
    pca_raw = PCA(n_components=n_components_raw, random_state=42)
    raw_pca = pca_raw.fit_transform(raw_inputs_vis)

    umap_raw = umap.UMAP(n_components=2, random_state=42)
    raw_embedding = umap_raw.fit_transform(raw_pca)

    plt.figure(figsize=(8, 6))
    sns.scatterplot(
        x=raw_embedding[:, 0], y=raw_embedding[:, 1],
        hue=[label_encoder.inverse_transform([label])[0] for label in labels_vis],
        palette='tab20', s=80, alpha=0.9
    )
    plt.title('UMAP of Raw Input')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(f"{output_dir}/UMAP_raw_input_sampled.png")
    plt.close()

    # 特征向量的PCA降维后UMAP
    pca_feat = PCA(n_components=n_components_feat, random_state=42)
    feat_pca = pca_feat.fit_transform(features_vis)

    umap_feat = umap.UMAP(n_components=2, random_state=42)
    feature_embedding = umap_feat.fit_transform(feat_pca)

    plt.figure(figsize=(8, 6))
    sns.scatterplot(
        x=feature_embedding[:, 0], y=feature_embedding[:, 1],
        hue=[label_encoder.inverse_transform([label])[0] for label in labels_vis],
        palette='tab20', s=80, alpha=0.9
    )
    plt.title('UMAP of Encoded Features')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(f"{output_dir}/UMAP_encoded_features_sampled.png")
    plt.close()