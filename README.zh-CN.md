# mech-interp-SAE-demo

本仓库是一个轻量级的 AI 机械可解释性复现与学习项目，核心围绕 TransformerLens、Sparse Autoencoder（SAE，稀疏自编码器）以及 activation patching（激活补丁 / 激活干预）展开。

项目目标不是提出新的可解释性方法，而是复现并搭建一条成熟、清晰、可运行的机械可解释性工作流，用于展示对该方向的理解、工程实现能力和持续学习过程。

---

## 一、项目目标

本项目面向小型语言模型，逐步完成以下内容：

1. 使用 TransformerLens 读取并缓存模型内部激活；
2. 理解 transformer 中的 residual stream、attention output、MLP output 等关键中间对象；
3. 使用 activation patching 对模型内部激活进行因果干预；
4. 使用 Sparse Autoencoder 对模型激活进行稀疏特征分解；
5. 分析 SAE latent / feature 的激活模式；
6. 构建一个小型 mechanistic interpretability case study；
7. 最终形成一个结构清晰、可复现、适合作为 GitHub 展示项目的仓库。

---

## 二、项目定位

本项目是一个复现 / 展示性质的工程项目，不是原创研究项目。

当前版本重点关注：

- 成熟工具链复现；
- 清晰的工程结构；
- 可复现的代码流程；
- 对核心概念的逐步学习；
- 对外展示机械可解释性方向的兴趣与技术能力。

当前版本暂不涉及：

- 自定义新型可解释性损失函数；
- 自定义 transformer 架构；
- 完整复现 cross-layer transcoder；
- 大规模模型实验；
- 论文级创新结果。

这些更高阶想法会放在 `notes/future_directions.md` 中，作为后续研究方向保留。

---

## 三、核心技术路线

本项目的基本技术路线为：

```text
小型语言模型
    ↓
TransformerLens 读取内部激活
    ↓
缓存 residual stream / attention / MLP 等中间表示
    ↓
Activation patching 做因果干预
    ↓
SAE 对激活进行稀疏特征分解
    ↓
Feature / latent 激活模式分析
    ↓
小型机制解释报告