# Learning Transferable Dynamics Priors from Action to World Modeling

> **一句话总结：** <!-- Action-to-Video 世界模型预训练作为迁移动力学先验的范式；A2World：在大规模机器人操作数据上预训练的 diffusion world model，可适配为 simulator（A2World-sim）和 policy（A2World-policy）。 -->

## 1. Paper Information

- **标题**: Learning Transferable Dynamics Priors from Action to World Modeling
- **作者**: Ze Huang, Jiahui Zhang, Hairuo Liu, Chenxi Zhang, Ran Cheng, Li Zhang
- **发表**: ECCV 2026
- **链接**: [arXiv](https://arxiv.org/abs/2606.29501) | [Code](https://github.com/LogosRoboticsGroup/A2World)

---

## 2. Motivation

### 2.1 Background

<!-- 现有世界模型（Dreamer 系列、PlaNet 等）从零训练，缺乏可迁移的先验知识；机器人领域大模型预训练（LLM, VLM）成功，但世界模型的预训练尚未被充分探索 -->

### 2.2 Goal

<!-- 核心研究问题：如何在大规模机器人操作数据上预训练 action-conditioned 世界模型，学到一个可迁移的 dynamics prior，同时服务于 simulator 和 policy？ -->

### 2.3 核心洞察

<!-- Actions 提供了因果监督信号：场景、视角千变万化，但交互规则（抓、推、放）由 actions 驱动。学 action→video 的预测等价于学可迁移的交互先验 -->

---

## 3. Core Ideas

### ① Action-to-Video 世界模型预训练

<!-- 不学 latent action，不学 text→video，而是直接用真实 action 标注做 action→video 的条件生成，学跨 embodiment 的 dynamics prior -->

### ② A2World：多视角 Diffusion World Model

<!-- 基于 DiT 的 diffusion 架构，支持多视角条件生成，在 2.1M+ 轨迹数据上预训练 -->

### ③ A2World-sim：适配为长期模拟器

<!-- 将预训练权重适配为 history-aware 自回归 simulator，支持 long-horizon rollout 和 policy evaluation -->

### ④ A2World-policy：适配为机器人策略

<!-- 同一套预训练权重，适配为 MoE-like video-action 联合预测模型，做 instruction-conditioned 控制 -->

---

## 4. Overall Framework

### A2World 预训练

```
                          ┌─────────────────┐
                          │                 │
                          └─────────────────┘
```

### A2World-sim / A2World-policy 适配

```
     ┌─────────────────────────────┐
     │                             │
     └─────────────────────────────┘
```

---

## 5. Key Modules

### 5.1 Multi-View Interactive Diffusion World Model

#### Motivation

<!-- 为什么要用 diffusion + 多视角？ -->

#### 核心思想

<!-- DiT 架构，action 作为 condition 注入 -->

> **一句话总结：**

---

### 5.2 A2World-sim (History-Aware Autoregressive Simulator)

#### Motivation

#### 核心思想

<!-- 自回归 rollout，condition on history，预测 future frames -->

> **一句话总结：**

---

### 5.3 A2World-policy (MoE-like Video-Action Joint Model)

#### Motivation

#### 核心思想

<!-- 共享 attention + action 专用 denoising branch -->

> **一句话总结：**

---

## 6. Mathematical Formulation

### 符号说明

| 符号 | 含义 | 维度 |
|------|------|------|
| $o_t$ | 第 t 帧观测（多视角） | |
| $a_t$ | 动作 | |
| $l$ | 语言指令 | |
| $z_t$ | latent feature | |

### Action-Conditioned Video Diffusion

### Adapt to A2World-sim

### Adapt to A2World-policy

### 训练目标

---

## 7. Training Pipeline

```
─────────────────────────────────────────────────────────────
Algorithm: A2World Pretraining → Adaptation
─────────────────────────────────────────────────────────────
 1:  ...
 ...
─────────────────────────────────────────────────────────────
```

---

## 8. Inference / Planning

<!-- A2World-sim：long-horizon rollout → policy evaluation
A2World-policy：单步前向 → instruction-conditioned action -->

---

## 9. Limitations

<!--
- Diffusion 推理速度慢
- 跨 embodiment 泛化仍有边界
- 长 horizon rollout 误差累积
- 依赖大规模高质量操作数据
-->

---

## 10. Relationship

### 历史脉络

```
★ Dreamer / PlaNet / RSSM              ← Latent dynamics world models
    │                                    从零训练，无先验迁移
    ▼
★ Video Pretraining (VideoPoet, ...)   ← Text→Video 生成
    │                                    缺少 action 条件信号
    ▼
★ Latent Action World Models           ← Motus, AdaWorld, LaWAM
    │                                    用 latent action 做预训练
    ▼
★ A2World (ECCV 2026)                  ← 本文
    Action→Video 预训练 + Sim/Policy 双适配
```

### 承上：对 Dreamer/PlaNet 的改进

| 维度 | Dreamer/PlaNet | A2World |
|------|---------------|---------|
| 训练范式 | 从零训练 | 大规模预训练 → 迁移 |
| 模型架构 | RSSM (latent Gaussian) | Diffusion Transformer |
| 数据需求 | 单任务交互数据 | 大规模跨 embodiment 数据 |
| 目标 | 单任务规划/控制 | 可迁移 dynamics prior |

### 启下：对后续工作的影响

<!-- Foundation World Model 方向；Sim-to-Real 的 simulator 路线 -->

---

## 11. Personal Insights

### 这篇论文为什么重要？

### 最有启发性的设计

### 历史定位

> 一句话评价：
