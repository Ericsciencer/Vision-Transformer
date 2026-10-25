


# Vision Transformer

### 选择语言 | Language

[中文简介](#%E7%AE%80%E4%BB%8B) | [English](#introduction)

### 结果 | Result

训练结果：
<img width="3569" height="1466" alt="image" src="https://github.com/user-attachments/assets/2c775b95-661d-4731-9bcd-a7afe5c368ac" />

注意力可视化：
<img width="1264" height="409" alt="image" src="https://github.com/user-attachments/assets/6b6d7960-ec35-4ea1-b8ac-650750320d10" />

---

## 简介

Vision Transformer（简称ViT）由 **Alexey Dosovitskiy, Lucas Beyer, Alexander Kolesnikov 等人** 于2020年提出（ICLR 2021），是计算机视觉领域的里程碑式革新。其核心贡献在于**首次将纯Transformer架构直接应用于图像分类任务，完全摒弃卷积归纳偏置**，通过将图像切分为Patch序列并采用标准Transformer编码器进行全局建模，打破了卷积神经网络在视觉任务中的长期垄断，证明了纯注意力架构在大规模视觉数据下的性能潜力，为后续视觉大模型、多模态基础模型奠定了核心架构基础。

在 ViT 出现之前，计算机视觉的主流方案均以卷积神经网络为核心，通过堆叠局部感受野提取层级视觉特征，天然具备空间局部性归纳偏置，但全局长距离依赖建模能力受限，且深层网络的计算与优化成本较高。即使引入注意力机制作为辅助模块，卷积结构本身仍是视觉建模的核心框架。ViT 首次借鉴自然语言处理中的序列建模思路，将二维图像转化为一维的Patch令牌序列，配合可学习位置编码注入空间信息，直接使用标准Transformer编码器完成全图的全局特征交互。在大规模预训练数据的支撑下，ViT 在ImageNet等分类基准上全面超越同期最优CNN模型，同时具备极强的迁移学习能力与泛化性能。

在 ImageNet-1K 图像分类任务上，ViT-L/16 模型取得了 **88.55% Top-1准确率**，超越当时所有卷积架构；在更大规模预训练下，ViT-H/14 可达到90.45%的Top-1准确率。本复现基于 PyTorch 从零实现完整的 Vision Transformer 编码器架构，严格遵循原论文的 Pre-LN 编码器结构、可学习位置编码、多头注意力与 CLS 令牌分类设计，采用 **CIFAR-10 彩色图像数据集** 进行验证，使用分类准确率作为核心评估指标，完整包含数据预处理、模型训练、测试评估、单图推理、注意力热力图可视化与训练曲线可视化全流程。

## 架构

本次复现严格遵循《An Image is Worth 16x16 Words》原文的标准架构，整体分为「Patch嵌入与位置编码」「堆叠Transformer编码器」「分类输出头」三大模块，完整保留原论文的所有核心设计。

- **Patch嵌入层**：输入图像首先被均匀切分为固定大小的图像块（Patch），每个Patch通过线性投影映射为 embed_dim 维的特征向量，等价于使用与Patch尺寸相同的卷积核进行非重叠卷积运算。Patch是ViT的基本处理单元，对应NLP任务中的词令牌，将二维图像转化为一维序列输入。
- **CLS令牌与位置编码**：在Patch序列的起始位置拼接一个可学习的分类令牌（CLS Token），最终以该令牌的输出特征完成分类预测，设计思想源自BERT的 
设计。同时为所有令牌叠加**可学习的位置编码**，注入空间位置信息，弥补自注意力本身的位置无关性。
- **Transformer编码器堆叠**：由 N 个完全相同的编码器层堆叠而成。每层包含两个子层：**多头自注意力层**与**位置前馈网络层**，每个子层均采用「层归一化 + 子层运算 + 残差连接」的 Pre-LN 结构，保证深层网络的训练稳定性。编码器对所有令牌进行全局信息交互，并行完成全序列的特征提取。
- **多头注意力机制**：将 Query、Key、Value 通过线性投影后拆分为 h 个并行的注意力头，每个头独立计算缩放点积注意力，最后将多个头的输出拼接并做线性投影。多头机制让模型可以同时捕捉不同特征子空间、不同区域的视觉依赖关系，同步建模局部纹理细节与全局语义关联。
- **前馈神经网络（MLP）**：由两层全连接层与 GELU 激活函数组成，先将特征维度按 mlp_ratio 升维再降回原维度，对每个位置的特征进行独立的非线性变换，进一步增强模型的特征表达能力。
- **分类输出头**：编码器全部堆叠运算完成后，取出序列首位的CLS令牌对应的特征向量，经过最终层归一化后接入线性分类头，映射为类别数维度的输出，通过Softmax得到各类别的预测概率。训练时使用交叉熵损失端到端优化，推理时直接取概率最高的类别作为预测结果。

<img width="922" height="456" alt="image" src="https://github.com/user-attachments/assets/8a393f59-b61b-4d77-8398-5375abd751c0" />


**具体网络细节**可以查看我的这篇文章：[Vision Transformer深入浅出]()

**注意**：为降低训练开销、便于快速复现验证，本实现采用轻量化 Mini-ViT 配置（4层堆叠、128维特征、4头注意力），数据集使用 CIFAR-10 小型图像语料，但模型结构、注意力计算、位置编码、训练范式均严格对齐原论文标准。

## 数据集

本次复现使用 **CIFAR-10 彩色图像数据集**，是计算机视觉领域经典的小型基准数据集。

- **任务类型**：10类别通用图像分类任务。
- **数据规模**：训练集 50000 张图像，测试集 10000 张图像；所有图像均为 32×32 分辨率三通道彩色图片，涵盖飞机、汽车、鸟类、猫、鹿、狗、青蛙、马、船、卡车共10个日常物体类别，类别分布均衡，适合快速验证模型能力。
- **预处理方式**：训练集采用随机裁剪（带填充）、随机水平翻转的数据增强策略，缓解过拟合；测试集仅做归一化处理；使用数据集全局统计的均值与标准差进行像素值归一化，加速模型收敛。
- **评估指标**：采用图像分类领域通用标准指标 **Top-1 准确率**，即预测类别与真实标签一致的样本占总样本的比例，数值范围为 0~100%，数值越高代表分类精度越好。同时配合交叉熵损失监控训练收敛状态，输出损失与准确率双维度训练曲线。

| Top-1准确率 | 效果说明 |
| --- | --- |
| 60 ~ 70% | 具备基础分类能力，可识别大部分主体类别，存在少量误判 |
| 70 ~ 80% | 分类效果良好，常见类别识别准确，泛化性较好 |
| 80 ~ 90% | 高精度分类，接近轻量CNN模型水平，鲁棒性较强 |
| 90%+ | 优秀分类性能，接近该数据集上的模型性能上限 |

数据集官方地址：[https://www.cs.toronto.edu/~kriz/cifar.html](https://www.cs.toronto.edu/~kriz/cifar.html)

---

## Introduction

Vision Transformer (ViT for short), proposed by **Alexey Dosovitskiy, Lucas Beyer, Alexander Kolesnikov et al.** in 2020 (ICLR 2021), is a milestone innovation in the field of computer vision. Its core contribution is that it **applies the pure Transformer architecture directly to image classification tasks for the first time, completely abandoning convolutional inductive bias**. By splitting images into patch sequences and using a standard Transformer encoder for global modeling, it breaks the long-term monopoly of convolutional neural networks in visual tasks, proves the performance potential of pure attention architecture on large-scale visual data, and lays the core architectural foundation for subsequent large vision models and multimodal foundation models.

Before ViT, mainstream computer vision solutions were centered on convolutional neural networks, extracting hierarchical visual features by stacking local receptive fields. They naturally have spatial locality inductive bias, but have limited ability to model global long-distance dependencies, and the computational and optimization cost of deep networks is high. Even with attention mechanisms as auxiliary modules, the convolutional structure itself remained the core framework of visual modeling. For the first time, ViT draws on the idea of sequence modeling in natural language processing, converts 2D images into 1D patch token sequences, injects spatial information with learnable position encoding, and directly uses the standard Transformer encoder to complete global feature interaction of the whole image. Supported by large-scale pre-training data, ViT comprehensively outperforms the state-of-the-art CNN models of the same period on classification benchmarks such as ImageNet, and also has strong transfer learning ability and generalization performance.

On the ImageNet-1K image classification task, the ViT-L/16 model achieved **88.55% Top-1 accuracy**, surpassing all previous convolutional architectures; with larger-scale pre-training, ViT-H/14 can reach 90.45% Top-1 accuracy. This reproduction implements the complete Vision Transformer encoder architecture from scratch based on PyTorch, strictly following the original paper's Pre-LN encoder structure, learnable position encoding, multi-head attention and CLS token classification design. It is validated on the **CIFAR-10 color image dataset** and uses classification accuracy as the core evaluation metric, covering the full pipeline of data preprocessing, model training, test evaluation, single image inference, attention heatmap visualization and training curve visualization.

## Architecture

This implementation faithfully follows the standard architecture from *An Image is Worth 16x16 Words*, consisting of three main modules: patch embedding with position encoding, stacked Transformer encoder, and classification output head, with all core designs from the original paper preserved.

- **Patch Embedding Layer**: The input image is first evenly divided into fixed-size image patches. Each patch is mapped to an embed_dim-dimensional feature vector through linear projection, which is equivalent to a non-overlapping convolution operation with a kernel size equal to the patch size. Patch is the basic processing unit of ViT, corresponding to word tokens in NLP tasks, converting 2D images into 1D sequence inputs.
- **CLS Token & Positional Encoding**: A learnable classification token (CLS Token) is concatenated at the beginning of the patch sequence, and the output feature of this token is used for final classification prediction, inspired by the design in BERT. At the same time, **learnable position encoding** is added to all tokens to inject spatial position information and compensate for the position invariance of self-attention itself.
- **Stacked Transformer Encoder**: Stacked by N identical encoder layers. Each layer contains two sub-layers: a **multi-head self-attention layer** and a **position-wise feed-forward network layer**. Each sub-layer adopts the Pre-LN structure of "LayerNorm + sub-layer computation + residual connection" to ensure the training stability of deep networks. The encoder performs global information interaction for all tokens and completes feature extraction of the whole sequence in parallel.
- **Multi-Head Attention**: Query, Key and Value are linearly projected and split into h parallel attention heads. Each head independently computes scaled dot-product attention, and outputs of all heads are finally concatenated and linearly projected. The multi-head mechanism allows the model to simultaneously capture visual dependencies from different feature subspaces and different regions, modeling both local texture details and global semantic associations.
- **Feed-Forward Network (MLP)**: It consists of two fully connected layers and GELU activation function. The feature dimension is first upscaled by mlp_ratio and then reduced back to the original dimension, performing independent nonlinear transformation on features of each position to further enhance the feature expression ability of the model.
- **Classification Output Head**: After all encoder layers are computed, the feature vector corresponding to the CLS token at the first position of the sequence is extracted, passed through the final LayerNorm, and then connected to a linear classification head, which maps to the output of category number dimension. The prediction probability of each category is obtained through Softmax. During training, it is optimized end-to-end with cross-entropy loss; during inference, the category with the highest probability is directly taken as the prediction result.

**Note**: To reduce training cost and facilitate quick reproduction and verification, this implementation uses a lightweight Mini-ViT configuration (4 stacked layers, 128-dimensional features, 4 attention heads) and the small CIFAR-10 image corpus. However, model structure, attention computation, position encoding, and training paradigm are all strictly aligned with the original paper standard.

## Dataset

This reproduction uses the **CIFAR-10 color image dataset**, a classic small benchmark dataset in the field of computer vision.

- **Task**: 10-class general image classification task.
- **Data Scale**: 50,000 training images and 10,000 test images. All images are 32×32 resolution three-channel color pictures, covering 10 daily object categories: airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck. The category distribution is balanced, which is suitable for quickly verifying model capability.
- **Preprocessing**: The training set adopts data augmentation strategies including random crop (with padding) and random horizontal flip to alleviate overfitting; the test set only performs normalization. Pixel value normalization is performed using the globally calculated mean and standard deviation of the dataset to accelerate model convergence.
- **Evaluation Metric**: **Top-1 Accuracy**, the universal standard metric for image classification. It refers to the proportion of samples whose predicted category is consistent with the ground truth label, with a value range of 0~100%. Higher values indicate better classification accuracy. Cross-entropy loss is also used to monitor training convergence, and dual-dimensional training curves of loss and accuracy are output.

Official dataset page: [https://www.cs.toronto.edu/~kriz/cifar.html](https://www.cs.toronto.edu/~kriz/cifar.html)

---

## 原文章 | Original article

Dosovitskiy A, Beyer L, Kolesnikov A, et al. An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale[C]//International Conference on Learning Representations. 2021.
