# Dream to Control: Learning Behaviors by Latent Imagination

> **一句话总结：** Dreamer 首次提出了**在潜空间通过想象学习行为**的范式——用 Actor-Critic 替代在线规划，将价值梯度和 TD(λ) 通过可微世界模型反向传播训练策略，在 20 个视觉控制任务上以更少数据超越 SOTA。

## 1. Paper Information

- **标题**: Dream to Control: Learning Behaviors by Latent Imagination
- **作者**: Danijar Hafner, Timothy Lillicrap, Jimmy Ba, Mohammad Norouzi
- **发表**: ICLR 2020 (Oral)
- **链接**: [arXiv](https://arxiv.org/abs/1912.01603) | [Code (TF1)](https://github.com/google-research/dreamer) | [Code (TF2)](https://github.com/danijar/dreamer) | [Project Page](https://danijar.com/project/dreamer/) | [Blog](https://research.google/blog/introducing-dreamer-scalable-reinforcement-learning-using-world-models/) | [Video](https://www.youtube.com/watch?v=BDxRNnhPTlU)

---

## 2. Motivation

### 2.1 Background

<!-- Model-Based vs Model-Free 的 tradeoff；PlaNet 的 CEM 规划有什么局限？为什么需要 Actor-Critic？ -->

### 2.2 Goal

<!-- 核心研究问题：如何利用世界模型，在潜空间高效学习长期视界的行为？ -->

### 2.3 核心洞察

<!-- 关键洞察：价值梯度可以通过可微世界模型反向传播到策略网络；Actor-Critic 替代 CEM 可突破视界限制 -->

---

## 3. Core Ideas

### ① RSSM 潜空间世界模型

<!-- 沿用 PlaNet 的 RSSM，在潜空间学习环境动力学。相比 World Models 的分离式 VAE+RNN，RSSM 统一端到端训练。 -->

### ② Actor-Critic 潜空间想象

<!-- 不进行在线规划，而是分别在潜空间学习 Actor（策略网络）和 Critic（价值网络），在想象轨迹上训练。 -->

### ③ 价值梯度反向传播

<!-- 核心贡献：将 TD(λ) 的 analytic gradient 通过可微世界模型反向传播，直接优化 Actor。 -->

### ④ 三大组件并行交替

<!-- Dynamics Learning ↔ Behavior Learning ↔ Environment Interaction 三者交替执行，可并行。 -->

---

## 4. Overall Framework

### 数据流（Inference Path）

<!--
Encoder: 观测 o_t → 潜表示
RSSM:  h_t, s_t 双路径
Actor:  潜状态 → 动作
Decoder: 潜状态 → 重建观测（仅训练用）
Reward & Value: 潜状态 → 奖励和长期价值（训练 Actor-Critic 用）
-->

```
             ┌────────────────┐
             │                │
             └────────────────┘
```

### 三大组件交替

```
      ┌─────────────────────────────┐
      │                             │
      └─────────────────────────────┘
```

---

## 5. Key Modules

### 5.1 World Model (RSSM)

#### Motivation

#### 核心思想

> **一句话总结：**

---

### 5.2 Actor (Action Model)

#### Motivation

#### 核心思想

> **一句话总结：**

---

### 5.3 Critic (Value Model)

#### Motivation

#### 核心思想

> **一句话总结：**

---

### 5.4 Reward Model & Continue Model

<!-- Dreamer 除了奖励模型，还引入了 Continue Model 预测 episode termination 概率 -->

---

## 6. Mathematical Formulation

### 符号说明

| 符号 | 含义 | 维度 |
|------|------|------|
| $o_t$ | 像素观测 | $64 \times 64 \times 3$ |
| $a_t$ | 动作 | $\mathbb{R}^{\vert\mathcal{A}\vert}$ |
| $r_t$ | 奖励 | $\mathbb{R}$ |
| $h_t$ | 确定性隐状态（GRU） | $\mathbb{R}^{200}$ |
| $s_t$ | 随机隐状态 | $\mathbb{R}^{30}$ |
| $z_t$ | 拼接特征 $[h_t, s_t]$ | $\mathbb{R}^{230}$ |

### Dynamics Learning Loss

<!--
Represenation Model:   enc(o_t)
Transition Model:      s_t ~ q(s_t | h_t)
Reward Model:          r_t ~ p(r_t | h_t, s_t)
Continue Model:        c_t ~ p(c_t | h_t, s_t)

Loss: 重建 + 奖励 + KL + continue
-->

### Behavior Learning: Actor-Critic in Latent Space

<!--
Imagination Horizon H
λ-returns: V_λ(s_τ)
Actor Loss:   maximize V_λ
Critic Loss:  regress V_λ
-->

### 总体 Loss 汇总

---

## 7. Training Pipeline

```
─────────────────────────────────────────────────────────────
Algorithm: Dreamer Training
─────────────────────────────────────────────────────────────
 1:  ...
 ...
─────────────────────────────────────────────────────────────
```

---

## 8. Inference / Planning

<!-- Dreamer 的推理非常简洁：只需一次 Actor 前向 → 执行动作，不再需要 CEM 在线搜索 -->

---

## 9. Limitations

<!--
- 世界模型误差累积：想象 rollout 越长，误差越大，策略会 exploit 模型漏洞
- Gaussian 潜变量假设：对离散、多模态环境表达不足（DreamerV2 用 Categorical 解决）
- 视觉复杂度受限：encoder/decoder 架构简单（CNN），复杂 3D 场景表现有限
-->

---

## 10. Relationship

### 历史脉络

```
★ PlaNet (ICML 2019)                  ← Hafner et al.
    RSSM + CEM 在线潜空间规划
        │
        ▼
★ Dreamer (ICLR 2020)                ← 本文
    RSSM + Actor-Critic 潜空间想象
    价值梯度反向传播
        │
        ├──→ DreamerV2 (NeurIPS 2020)
        │     Categorical RSSM 替代 Gaussian
        │     首次 Atari 纯 model-based 超越 model-free
        │
        ├──→ DreamerV3 (NeurIPS 2022)
        │     通用 agent，固定超参数
        │     Minecraft 获取钻石（首个纯 model-based）
        │
        └──→ TD-MPC / IRIS / TWM / STORM
              后续潜空间 model-based 方法
```

### 承上：对 PlaNet 的改进

| 维度 | PlaNet | Dreamer |
|------|--------|---------|
| 决策方式 | CEM 在线规划（每步 I×J 次 rollout） | Actor 单次前向 |
| 长期视界 | H=12 有限视界 | λ-returns + value function 估计无限视界 |
| 计算效率 | 每步 ~10ms | 每步 ~1ms |
| 策略 | 无显式策略 | 显式策略网络（Actor） |
| 价值估计 | 无 | Critic 网络 |

### 启下：对后续工作的影响

<!-- Dreamer 系列、TD-MPC 等 -->

---

## 11. Personal Insights

### 这篇论文为什么重要？

### 最有启发性的设计

### 历史定位

> 一句话评价：
