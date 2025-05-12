### **`<font style="color:rgb(64, 64, 64);">`1. 扰动机制设计`</font>`**

#### `<font style="color:rgb(64, 64, 64);">`（1）`</font>`**`<font style="color:rgb(64, 64, 64);">`扰动类型`</font>`**`<font style="color:rgb(64, 64, 64);">`：`</font>`

+ **`<font style="color:rgb(64, 64, 64);">`基于TCP重传逻辑的扰动（合理性）`</font>`**`<font style="color:rgb(64, 64, 64);">`：`</font>`
  - `<font style="color:rgb(64, 64, 64);">`模拟真实网络中的丢包和重传行为，生成两种扰动：`</font>`
    * **`<font style="color:rgb(64, 64, 64);">`子序列重复`</font>`**`<font style="color:rgb(64, 64, 64);">`（对应超时重传RTO，数据包重复）。`</font>`
    * **`<font style="color:rgb(64, 64, 64);">`子序列移位`</font>`**`<font style="color:rgb(64, 64, 64);">`（对应快速重传FAST，数据包乱序）。`</font>`
  - **`<font style="color:rgb(64, 64, 64);">`扰动参数控制`</font>`**`<font style="color:rgb(64, 64, 64);">`：`</font>`
    * `<font style="color:rgb(64, 64, 64);">`丢包率（Packet Loss Rate）：控制扰动强度（如0.1%（理想网络）,1%（轻度拥塞）, 5%（中度拥塞）, 10%（重度拥塞）），前两种情况使用FAST场景，后两种情况使用RTO场景。`</font>`
    * `<font style="color:rgb(64, 64, 64);">`捕获点位置（上游/下游）：决定扰动表现形式（重复或移位）。`</font>`

#### `<font style="color:rgb(64, 64, 64);">`（2）`</font>`**`<font style="color:rgb(64, 64, 64);">`语义保留性`</font>`**`<font style="color:rgb(64, 64, 64);">`：`</font>`

+ `<font style="color:rgb(64, 64, 64);">`强调扰动是`</font>`**`<font style="color:rgb(64, 64, 64);">`协议相关的`</font>`**`<font style="color:rgb(64, 64, 64);">`（TLS over TCP），因此重传行为不会破坏流量的语义特征。`</font>`
+ `<font style="color:rgb(64, 64, 64);">`通过理论分析或实验验证扰动前后样本的相似性（如计算序列编辑距离或余弦相似度），目前使用的是UMAP，只有当每类随机取两个样本时候才能得到能够接收的样本相似性变化。`</font>`

---

### **`<font style="color:rgb(64, 64, 64);">`2. 对比学习框架`</font>`**

#### `<font style="color:rgb(64, 64, 64);">`（1）`</font>`**`<font style="color:rgb(64, 64, 64);">`视图生成`</font>`**`<font style="color:rgb(64, 64, 64);">`：`</font>`

+ `<font style="color:rgb(64, 64, 64);">`对同一原始样本应用`</font>`**`<font style="color:rgb(64, 64, 64);">`同一场景下的两种扰动`</font>`**`<font style="color:rgb(64, 64, 64);">`（如RTO重复 + RTO移位），生成正样本对。`</font>`
+ `<font style="color:rgb(64, 64, 64);">`负样本来自其他流量的扰动结果（类似SimCLR的取同一批次内其他所有样本作为负样本）。`</font>`

#### `<font style="color:rgb(64, 64, 64);">`（2）`</font>`**`<font style="color:rgb(64, 64, 64);">`编码器与损失函数`</font>`**`<font style="color:rgb(64, 64, 64);">`：`</font>`

+ **`<font style="color:rgb(64, 64, 64);">`编码器`</font>`**`<font style="color:rgb(64, 64, 64);">`：LSTM配合attention机制的Pooling共同作为编码器（适合时序数据）。`</font>`
+ **`<font style="color:rgb(64, 64, 64);">`损失函数`</font>`**`<font style="color:rgb(64, 64, 64);">`：NT-Xent损失，最大化正样本对的相似性，最小化负样本对的相似性。`</font>`

#### `<font style="color:rgb(64, 64, 64);">`（3）`</font>`**`<font style="color:rgb(64, 64, 64);">`类似SimCLR的对比学习逻辑`</font>`**`<font style="color:rgb(64, 64, 64);">`：`</font>`

+ **预训练pretrain**：使用带注意力机制的双向单层LSTM作为编码器，对序列样本进行自监督训练。核心为通过不同的增强视图构造正负样本对，用NT-Xent损失最小化正样本对的距离，最大化负样本对的距离，学习富表征力表示，保存encoder供finetune阶段使用

**损失函数**：NT-Xent-loss是SimCLR风格对比学习中使用的NT-Xent（Normalized Temperature-scaled Cross Entropy Loss）标准化温度缩放交叉熵损失在自动拉近正样本的同时压低负样本的相似度，因为它是通过竞争式的softmax达到的，而不是显式最小化负样本相似度

+ **微调finetune**：
  使用预训练阶段保存的自注意力机制结合LSTM结构的编码器，再有标签数据上进行微调训练，然后在测试集上评估和可视化分析（不一定用）

---

### **`<font style="color:rgb(64, 64, 64);">`3. 实验设置`</font>`**

#### `<font style="color:rgb(64, 64, 64);">`（1）`</font>`**`<font style="color:rgb(64, 64, 64);">`数据集`</font>`**`<font style="color:rgb(64, 64, 64);">`：`</font>`

+ `<font style="color:rgb(64, 64, 64);">`使用`</font>`**`<font style="color:rgb(64, 64, 64);">`CESNET-TLS22`</font>`**`<font style="color:rgb(64, 64, 64);">`，预处理为固定长度序列（如30个数据包长度），标准化（除以1460）。`</font>`

#### `<font style="color:rgb(64, 64, 64);">`（2）`</font>`**`<font style="color:rgb(64, 64, 64);">`实验组设计`</font>`**`<font style="color:rgb(64, 64, 64);">`：`</font>`

| **`<font style="color:rgb(64, 64, 64);">`实验组`</font>`**              | **`<font style="color:rgb(64, 64, 64);">`目的`</font>`**           | **`<font style="color:rgb(64, 64, 64);">`对照设置`</font>`**                                      |
| --------------------------------------------------------------------------------- | ---------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------- |
| **`<font style="color:rgb(64, 64, 64);">`原始数据 + 监督学习`</font>`** | `<font style="color:rgb(64, 64, 64);">`Baseline（下限）`</font>`         | `<font style="color:rgb(64, 64, 64);">`LSTM/RF直接训练`</font>`                                         |
| **`<font style="color:rgb(64, 64, 64);">`扰动数据 + 监督学习`</font>`** | `<font style="color:rgb(64, 64, 64);">`验证扰动是否破坏特征分布`</font>` | `<font style="color:rgb(64, 64, 64);">`对比学习 vs. 直接监督学习`</font>`                               |
| **`<font style="color:rgb(64, 64, 64);">`不同丢包率实验`</font>`**      | `<font style="color:rgb(64, 64, 64);">`分析扰动强度的影响`</font>`       | `<font style="color:rgb(64, 64, 64);">`丢包率梯度（0.1%, 1%, 5%, 10%），不用丢包率使用不同场景`</font>` |

#### `<font style="color:rgb(64, 64, 64);">`（3）`</font>`**`<font style="color:rgb(64, 64, 64);">`评估指标`</font>`**`<font style="color:rgb(64, 64, 64);">`：`</font>`

+ **`<font style="color:rgb(64, 64, 64);">`主指标`</font>`**`<font style="color:rgb(64, 64, 64);">`：分类准确率、F1-score、混淆矩阵等、视情况或许加入AUC-ROC。`</font>`
+ **`<font style="color:rgb(64, 64, 64);">`辅助分析`</font>`**`<font style="color:rgb(64, 64, 64);">`：`</font>`
  - `<font style="color:rgb(64, 64, 64);">`UMAP可视化特征分布（验证对比学习的聚类效果），这里横纵坐标值没有意义，注重结构分布。`</font>`

---

### **`<font style="color:rgb(64, 64, 64);">`4. 结论`</font>`**

1. **`<font style="color:rgb(64, 64, 64);">`扰动有效性`</font>`**`<font style="color:rgb(64, 64, 64);">`：`</font>`
   - `<font style="color:rgb(64, 64, 64);">`在适度丢包率（如<5%）以下，扰动后的对比学习模型性能`</font>`**`<font style="color:rgb(64, 64, 64);">`优于传统增强方法`</font>`**`<font style="color:rgb(64, 64, 64);">`（如重采样）。`</font>`
   - `<font style="color:rgb(64, 64, 64);">`高丢包率（如>10%）时性能下降，但仍保持鲁棒性。`</font>`
2. **`<font style="color:rgb(64, 64, 64);">`对比学习优势`</font>`**`<font style="color:rgb(64, 64, 64);">`：`</font>`
   - `<font style="color:rgb(64, 64, 64);">`比直接使用扰动数据训练监督模型的性能提升`</font>`**`<font style="color:rgb(64, 64, 64);">`5-10% F1-score`</font>`**`<font style="color:rgb(64, 64, 64);">`。`</font>`
   - `<font style="color:rgb(64, 64, 64);">`跨扰动场景（RTO+FAST）比单一场景效果更好。`</font>`

---

### **`<font style="color:rgb(64, 64, 64);">`5. points`</font>`**

+ **`<font style="color:rgb(64, 64, 64);">`协议感知的扰动`</font>`**`<font style="color:rgb(64, 64, 64);">`：不同于随机噪声，扰动模拟真实网络行为（TCP重传）。`</font>`
+ **`<font style="color:rgb(64, 64, 64);">`无需标签的增强`</font>`**`<font style="color:rgb(64, 64, 64);">`：对比学习利用无标签数据生成鲁棒表示。`</font>`
+ **`<font style="color:rgb(64, 64, 64);">`实际意义`</font>`**`<font style="color:rgb(64, 64, 64);">`：预计模型在真实网络环境（如移动网络、Wi-Fi）中表现稳定。`</font>`
