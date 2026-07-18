# Learning Latent Dynamics for Planning from Pixels

> PlaNet 首次提出了**可用于规划（Planning）的潜在世界模型（Latent World Model）**。通过 RSSM 学习环境动力学，并利用 CEM 在潜在空间进行在线规划，大幅提升了 Model-Based Reinforcement Learning 的样本效率，是 **Dreamer 系列和现代 World Model 的重要奠基工作**。

## 1. Paper Information

- **标题**: Learning Latent Dynamics for Planning from Pixels
- **作者**: Danijar Hafner, Timothy Lillicrap, Ian Fischer, Ruben Villegas, David Ha, Honglak Lee, James Davidson
- **发表**: ICML 2019
- **链接**: [arXiv](https://arxiv.org/abs/1811.04551) | [Code](https://github.com/google-research/planet) | [Project Page](https://planetrl.github.io/) | [Blog](https://ai.googleblog.com/2019/02/introducing-planet-deep-planning.html)

---

## 2. Motivation

### 2.1 Background

传统强化学习主要采用 **Model-Free RL**，存在两个核心问题：

- 样本效率极低，需要大量环境交互；
- 从高维图像直接学习控制策略十分困难。

另一方面，早期 World Models（2018）虽然能够学习环境模型，但更多用于生成或模拟，并未形成真正有效的规划框架。

### 2.2 Goal

PlaNet 希望回答一个问题：

> **能否直接从图像学习一个潜在世界模型，并在潜在空间完成规划，而无需预测未来图像？**

其核心思想是：

> **Planning in Latent Space instead of Pixel Space.**

### 2.3 核心洞察

> 一个真正可用于规划的动力学模型，必须能**从任意时间点开始准确预测未来多步的奖励**。这意味着：
> - 隐状态必须同时包含**确定性记忆**（追踪历史上下文）和**随机性路径**（建模不可约的随机性）；
> - 训练目标不能只是单步重建 + 单步 KL，必须强制模型具备多步预测能力（latent overshooting）；
> - 规划应该在潜空间中完成，避免解码图像的计算开销。

PlaNet 正是基于这些洞察，将 World Models 的分离组件统一为一个**端到端变分训练**的 RSSM，并用 CEM 规划替代进化策略。

---

## 3. Core Ideas

PlaNet 的核心贡献可以总结为四点：

### ① RSSM：学习稳定的潜在动力学

结合 **Deterministic State（GRU）** 与 **Stochastic State（Latent Variable）**，同时建模长期记忆和环境随机性。

---

### ② Latent Planning

规划不再发生在像素空间，而是在低维潜在空间 Rollout。

相比预测未来图像：

- 计算更快
- 更稳定
- 更适合控制任务

---

### ③ Latent Overshooting

训练时不仅约束一步预测，还约束多步预测，使长期 Rollout 更稳定，减少误差累积。

---

### ④ CEM Online Planning

利用 Cross Entropy Method 在潜在空间搜索未来动作序列，每一步重新规划，实现 MPC（Model Predictive Control）。

---

## 4. Overall Framework

PlaNet 整体上是一个交替系统（Alternating System），在**模型训练**（从历史数据中学习）和**数据采集**（用当前模型做决策）之间不断循环。

### 数据流（Inference Path）

```
Observation（64×64×3）
      │
      ▼
 CNN Encoder ─────────────────── 像素 → 特征向量
      │
      ▼
┌──────────────────────────────┐
│          RSSM                │
│                              │
│  ┌────────────────────┐      │
│  │ Deterministic Path │      │  GRU: h_t = f(h_{t-1}, s_{t-1}, a_{t-1})
│  │   (GRU, 200 dim)   │      │  追踪长期依赖（如物体离开视野后仍记住）
│  └────────┬───────────┘      │
│           │                  │
│           ▼                  │
│  ┌────────────────────┐      │
│  │  Stochastic Path    │      │  s_t ~ p(s_t | h_t) 或 q(s_t | h_t, enc(o_t))
│  │  (Gaussian, 30 dim) │      │  建模环境随机性
│  └────────┬───────────┘      │
│           │                  │
│      z_t = (h_t, s_t)        │
└──────────┬───────────────────┘
           │
     ┌─────┴─────┐
     │           │
     ▼           ▼
 Decoder    Reward Model
 p(o|h,s)    p(r|h,s)
（仅训练用）   （训练+规划用）
```

### 训练与采集交替

```
     ┌─────────────────────────────────────────────┐
     │  收集种子数据 (随机策略, 5 episodes)          │
     └──────────────────┬──────────────────────────┘
                        ▼
     ┌─────────────────────────────────────────────┐
     │  Phase 1: 模型训练                            │
     │  从数据集 D 采样序列块 → RSSM 前向 → 计算      │
     │  ELBO + Latent Overshooting Loss → 梯度更新   │
     │  重复 C=100 步                                │
     └──────────────────┬──────────────────────────┘
                        ▼
     ┌─────────────────────────────────────────────┐
     │  Phase 2: 数据采集 (用当前模型)               │
     │  编码当前帧 → CEM 潜空间规划最优动作序列 →     │
     │  执行第一步 → 存 (o, a, r) 到 D               │
     │  跑完一整个 episode                           │
     └──────────────────┬──────────────────────────┘
                        │
                        ▼
                回到 Phase 1（循环）
```

整个系统包含四部分：**Encoder**（图像→特征）、**RSSM**（潜空间动力学）、**Reward Model**（奖励预测）和 **Planner/CEM**（动作搜索）。其中 Decoder 仅在训练时提供重建监督信号，规划阶段完全不使用。

---

## 5. Key Modules

### 5.1 RSSM（Recurrent State Space Model）

#### Motivation

传统 RNN 无法准确建模环境随机性，而普通状态空间模型又缺乏长期记忆。

RSSM 将二者结合：

```
Latent State
    │
 ┌──┴──┐
 │     │
h_t   s_t
```

其中：

- h：Deterministic State（长期记忆，比如曾经检出过一辆车，但是被遮挡了，其实它是存在的）
- s：Stochastic State（随机因素，比如只有一帧图像，一辆车处在同一个位置，但是它可能静止，正在加速，正在减速等）

#### 核心思想

每一步维护两个状态：

- GRU 保存历史信息；
- Latent Variable 建模环境不确定性。

最终状态：

```
z_t = (h_t, s_t)
```

> **一句话总结：**
>
> RSSM 是 PlaNet 最重要的贡献，也是 Dreamer 系列一直沿用的核心结构。

---

### 5.2 Latent Overshooting

**问题**：标准的单步 ELBO 训练中，梯度从 $p(s_t \mid h_t)$ 流向后验 $q(s_{t-1})$ 后就被截断了，**从不链式传播多个时间步**。这意味着模型从未被要求做多步开环预测——而规划时恰恰需要从当前状态 rollout 未来 H 步。

**核心思想**：除了单步 KL，还要求模型在所有时间尺度上（d=2, 3, ..., D）都能准确预测：

```
标准单步训练:     t → t+1（梯度不传播更远）
Latent Overshooting:
                  t → t+2 ← 用先验展开 2 步，与后验做 KL 一致性
                  t → t+3 ← 展开 3 步，KL 一致性
                  ...
                  t → t+D ← 展开 D 步（D=50，即 chunk 长度）
```

具体做法：对每个距离 d，用先验 $p$ 做 d 步开环展开得到 $\hat{s}_{t}$，然后与基于完整观测的后验 $q(s_t \mid o_{\le t})$ 做 KL 约束。对 d>1 的后验做 **stop-gradient**，确保先验主动"追赶"后验，而非后验被弱先验拉偏。

> **一句话总结：**
>
> Latent Overshooting 将"规划时需要的多步预测能力"直接加入训练目标，是 PlaNet 相比普通 RSSM 的关键改进。

---

### 5.3 Reward Model

RSSM 输出 Latent State 后，同时预测：

```
Latent
   │
 ┌─┴─────┐
 │        │
Decoder Reward
```

Reward Prediction 的意义：

- 无需真实环境即可评估动作；
- 支持在线规划。

> **一句话总结：**
>
> Reward Model 将环境交互转化为模型内部评估，是 Planning 的基础。

---

### 5.4 Cross Entropy Method（CEM）

PlaNet 不训练 Policy Network（考虑world model正在学习，如如果再次学习policy，policy会适应模型误差钻漏洞），而是在每一步：

**初始化**：$q(a_{t:t+H}) = \mathcal{N}(0, I)$

**每轮迭代**：
1. 从 $q$ 采样 $J$ 个候选动作序列；
2. 对每个候选，用 RSSM 潜空间 rollout 计算累计预测奖励：$\hat{R}^{(j)} = \sum_{\tau=t}^{t+H} \mu_r(s_\tau^{(j)})$；
3. 选出 top-K 个候选（最大 $\hat{R}$）；
4. 用 top-K 的均值和方差**拟合新的** $q(a_{t:t+H})$；
5. 重复 I 轮。

**执行**：取 $q(a_t)$ 的均值作为本步动作，然后 receding horizon 重规划。

最终执行：

```
Only First Action
```

随后重新规划。

这是经典 MPC 思想。

> **一句话总结：**
>
> CEM 将世界模型真正用于决策，而不仅仅用于预测。

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

### RSSM 生成过程

RSSM 定义了一个时序潜变量模型，每一步先更新确定性状态 $h_t$，再从 $h_t$ 生成随机状态 $s_t$，最后从 $(h_t, s_t)$ 重建观测并预测奖励：

$$
\begin{aligned}
\text{确定性转移:}\quad & h_t = f(h_{t-1}, s_{t-1}, a_{t-1}) \\
\text{随机先验:}\quad & s_t \sim p(s_t \mid h_t) = \mathcal{N}\big(\mu_\theta(h_t), \sigma_\theta^2(h_t)\big) \\
\text{后验（编码器）:}\quad & s_t \sim q(s_t \mid o_{\le t}, a_{<t}) = \mathcal{N}\big(\mu_\phi(h_t, \text{enc}(o_t)), \sigma_\phi^2(\cdot)\big) \\
\text{观测模型（解码器）:}\quad & o_t \sim p(o_t \mid h_t, s_t) \\
\text{奖励模型:}\quad & r_t \sim p(r_t \mid h_t, s_t)
\end{aligned}
$$

### 训练目标：ELBO（单步）

对整个序列的证据下界包含三项：

$$
\ln p(o_{1:T}, r_{1:T} \mid a_{1:T}) \ge \sum_{t=1}^T \Big[
\underbrace{\mathbb{E}_{q}[\ln p(o_t \mid h_t, s_t)]}_{\text{图像重建}}
+ \underbrace{\mathbb{E}_{q}[\ln p(r_t \mid h_t, s_t)]}_{\text{奖励预测}}
- \underbrace{\beta \cdot D_{\text{KL}}\big(q(s_t \mid o_{\le t}) \,\|\, p(s_t \mid h_t)\big)}_{\text{KL 正则化}}
\Big]
$$

其中 KL 项让后验 $q$ 接近仅基于过去动作的先验 $p$，防止 $s_t$ 编码过多冗余信息。

**Free nats 技巧**：将 KL 项设为 $\max(\text{KL}, 3\ \text{nats})$，前 3 nats 不计入 loss。这给编码器一定的"自由"去编码信息，防止训练初期强迫后验匹配一个还很差的先验。

### Latent Overshooting（多步变分目标）

单步 ELBO 的缺陷：梯度只流经单步 $p(s_t \mid h_t)$，**从不链式传播**，模型从未被训练做多步开环预测。

Latent Overshooting 显式地对所有距离 $d=2..D$ 添加多步 KL 约束：

对距离 $d$，定义从 $t-d$ 开始的 d 步先验展开：

$$
p_d(s_t \mid o_{\le t-d}, a_{<t}) = \mathbb{E}_{q(s_{t-d})}\Big[\prod_{\tau=t-d+1}^t p(s_\tau \mid h_\tau)\Big]
$$

最终的多步训练目标（加权平均所有距离的 bound）：

$$
\boxed{\mathcal{L}_{\text{LO}} = \frac{1}{D} \sum_{d=1}^D \mathcal{L}_d}
$$

其中 $D$ 为最大 overshooting 距离（论文取 $D=50$，等于序列 chunk 长度）。实践上，对 $d>1$ 的后验做 **stop-gradient**，确保多步先验主动"追赶"后验，而不是后验被弱先验拉偏。

### 总体 Loss 汇总

$$
\mathcal{L} = \underbrace{\mathcal{L}_{\text{recon}}}_{\text{重建 MSE}} + \underbrace{\mathcal{L}_{\text{reward}}}_{\text{奖励 MSE}} + \underbrace{\mathcal{L}_{\text{KL}} \big(\text{clipped at 3 nats}\big)}_{\text{单步 KL}} + \underbrace{\sum_{d=2}^D \beta \cdot \text{KL}^{(d)}}_{\text{多步 Overshooting}}
$$

---

## 7. Training Pipeline

PlaNet 的训练在两个阶段之间交替：**模型学习**（在已有数据上优化 RSSM）和**数据采集**（用当前模型 + CEM 规划与环境交互）。

```
─────────────────────────────────────────────────────────────
Algorithm: PlaNet Training (Alternating)
─────────────────────────────────────────────────────────────
 1:  初始化空数据集 D
 2:  用随机策略采集 S=5 个 seed episodes 存入 D
 3:
 4:  for episode = 1..TotalEpisodes do
 5:      // ─── Phase 1: 模型训练（在 D 上优化 C=100 步）───
 8:      for step = 1..C do
 9:          {o_t, a_t, r_t}_{1:L} ~ D    // 采样 B=50 条 L=50 的序列块
10:          h_0, s_0 = 0                  // 初始化 GRU 隐状态和随机状态
11:
12:          for t = 1..L do
13:              // 后验：编码器 + GRU → 当前随机状态
14:              s_t ~ q(s_t | h_{t-1}, enc(o_t))
15:              // 先验：仅从过去预测（用于 KL 正则）
16:              ŝ_t ~ p(ŝ_t | h_{t-1})
17:              // 确定性路径：GRU 递推
18:              h_t = GRU(h_{t-1}, s_{t-1}, a_{t-1})
19:          end for
20:
21:          // 单步损失
22:          ℒ_recon  = MSE(dec(s_{1:L}), o_{1:L})          // 图像重建
23:          ℒ_reward = MSE(rwd(s_{1:L}), r_{1:L})          // 奖励预测
24:          ℒ_kl     = max(KL(q(s_t) ‖ p(ŝ_t)), 3 nats)   // 单步 KL + free nats
25:
26:          // Latent Overshooting：多步 KL 一致性
27:          for d = 2..D do                               // D = 50
28:              ŝ_{t-d+1..t} ← open_loop_predict(h_{t-d}, a_{t-d..t})  // d 步先验展开
29:              ℒ_kl += β · KL(q(s_t).detach() ‖ p(ŝ_t))  // 后验 stop-grad
30:          end for
31:
32:          ℒ = ℒ_recon + ℒ_reward + ℒ_kl
33:          θ ← Adam(ℒ, θ, lr=1e-3)
34:      end for
35:
36:      // ─── Phase 2: 数据采集（用 CEM 规划决策）───
37:      o_1 = env.reset()
38:      h_1, s_1 = 0
39:
40:      for t = 1..EpisodeLength do
41:          s_t ~ q(s_t | h_t, enc(o_t))           // 编码当前状态
42:          a_{t..t+H} ← CEM(s_t, H=12, I=10, J=1000, K=100)  // 潜空间规划
43:          o_{t+1}, r_t, done = env.step(a_t + 𝒩(0, 0.3))   // 执行 + 探索噪声
44:          h_{t+1} = GRU(h_t, s_t, a_t)
45:          D ← (o_t, a_t, r_t)
46:          if done then break
47:      end for
48:  end for
─────────────────────────────────────────────────────────────
```

---

## 8. Inference / Planning

PlaNet 的推理阶段完全由 CEM 规划主导，没有显式的策略网络。

### CEM Latent Planning 算法

```
Algorithm: CEM Latent Planning
──────────────────────────────────────────────
输入: 当前状态信念 q(s_t | o_≤t, a_<t)
超参: H=12 (规划视界), I=10 (迭代次数),
      J=1000 (候选数), K=100 (保留数)

1. 初始化动作序列分布 q(a_{t:t+H}) ← N(0, I)
2. for i = 1..I do
3.   for j = 1..J do
4.     采样动作序列: a^{(j)}_{t:t+H} ~ q(a_{t:t+H})
5.     在潜空间中 rollout:
6.       for τ = t..t+H-1:
7.         s^{(j)}_{τ+1} ~ p(s | s^{(j)}_τ, a^{(j)}_τ)   ← 先验展开
8.         r^{(j)}_{τ+1} = mean(p(r | s^{(j)}_{τ+1}))
9.     累计奖励: R^{(j)} = Σ r^{(j)}_τ
10.  选出 top-K 候选: K = argsort({R^{(j)}})[:K]
11.  更新分布: q(a_{t:t+H}) ← fit_Gaussian({a^{(k)}_{t:t+H} for k ∈ K})
12. end: for
13. 执行 a_t = mean(q(a_t))   ← 只执行第一步
14. 收到 o_{t+1} 后 → 重规划 (回到 step 1)
```

### CEM vs Actor-Critic

| 维度 | PlaNet (CEM) | 后续 Dreamer (Actor-Critic) |
|------|-------------|----------------------|
| 决策方式 | 每步在线优化 (I×J 次 rollout) | 前向推理策略网络 (1 次 forward) |
| 计算开销 | 每步 ~10ms (GPU) | 每步 ~1ms |
| 视界处理 | 有限视界 (H=12)，之外不考虑 | 通过 value function 估计长期回报 |
| 对模型精度的依赖 | 高——所有决策依赖 rollout | 较低——策略可渐进改善 |
| 适用场景 | 短/中视界、模型较准的环境 | 长视界、高维动作空间 |

---

## 9. Limitations

PlaNet 在数据效率上取得了突破，但存在以下局限：

1. **规划视界有限（H=12）**：CEM 只能考虑有限步的未来奖励，对于需要极长视界的任务（如导航、复杂操作）效果不佳。超参数 H 的选择需要权衡：H 越大，CEM 搜索空间指数增长，计算开销也线性上升；
2. **CEM 计算成本高**：每步需要 I × J 次潜空间 rollout（默认 10 × 1000 = 10000 次），尽管在 GPU 上可批量化，但仍比一次策略网络前向贵 2–3 个数量级。这限制了 PlaNet 在高帧率场景（如实时机器人控制）中的部署；
3. **未解决探索问题**：使用简单的 $\mathcal{N}(0, 0.3)$ 探索噪声。在稀疏奖励环境中，随机探索效率低下；
4. **无显式长期价值估计**：CEM 只优化视界 H 内的累计奖励，不学习 value function。对于视界外的重要事件（如"先拿钥匙再开门"），H=12 的规划根本看不到；

这些局限直接引领了后续 Dreamer 系列的发展方向。

---

## 10. Relationship

### 历史脉络

```
World Models (2018)               ← Ha & Schmidhuber
    V (VAE) + M (MDN-RNN) + C (CMA-ES)
    分离式训练，非端到端
        │
        ▼
★ PlaNet (ICML 2019)              ← 本文
    RSSM（确定性 + 随机性统一路径）
    Latent Overshooting 多步变分训练
    CEM 潜空间规划
        │
        ├──→ Dreamer (ICLR 2020)
        │     Actor-Critic 替代 CEM
        │     用 value gradients 优化策略
        │     扩展到 20+ 任务
        │
        ├──→ DreamerV2 (NeurIPS 2020)
        │     Categorical RSSM 替代 Gaussian
        │     首次 Atari 纯 model-based 超越 model-free
        │
        ├──→ DreamerV3 (NeurIPS 2022)
        │     通用 agent，固定超参数
        │     Minecraft 获取钻石（首个纯 model-based）
        │
        ├──→ Genie (ICLR 2024)
        │     Video 预训练的 Foundation World Model
        │
        └──→ Modern Embodied World Models
              RT-2, UniSim, etc.
```

### 承上：对 World Models 的改进

| 维度 | World Models | PlaNet |
|------|-------------|--------|
| 架构 | VAE + MDN-RNN (分离) | RSSM (统一端到端) |
| 随机性建模 | 仅在 MDN 输出层用 Gaussian 混合 | 每条时间步的随机状态 $s_t$ |
| 确定性路径 | GRU 在 M 组件内部 | GRU 是**第一级路径**，随机状态是第二级 |
| 训练目标 | 分别优化 VAE loss 和 MDN NLL | 联合 ELBO + Latent Overshooting |
| 策略优化 | CMA-ES 黑箱进化 | CEM 在线规划（无策略网络） |

### 启下：对后续工作的影响

- **Dreamer (2019)**：保留 RSSM，但用 Actor-Critic 替代 CEM。核心洞察"规划器本身就是一种策略"——在潜空间中学习显式策略和价值函数，避免每步执行 CEM 的高计算开销。同时可以用 TD 估计视界外回报，突破 H=12 的限制；
- **RSSM 成为标准组件**：从 Dreamer 系列到 DayDreamer、IRIS、TWM（Transformer World Model），"deterministic + stochastic"双路径设计成为潜空间世界模型的事实标准架构；
- **Latent Overshooting** 启发了后续对"多步预测一致性"的研究，如 DreamerV2 的 KL balancing、TD-MPC 的 latent consistency loss；
- **潜空间规划范式**的建立——不生成像素、不在真实环境中 rollout，仅凭准确的世界模型 + 规划器完成控制。这一思想直接延续到 Dreamer 系列乃至 Genie 等 foundation world model 工作中。

---

## 11. Personal Insights

### 这篇论文为什么重要？

PlaNet 在 2019 年最重要的贡献不是刷榜——它的绝对分数并**没有**碾压当时的最优 model-free 方法——而是**证明了"从像素学可规划模型"这个方向可行**。

在 PlaNet 之前，model-based RL 在像素域有两个怀疑论论点：
1. "学到能用的模型太难了，误差积累后规划不如直接学策略"；
2. "即使能学到模型，CEM 这种规划器的计算开销也太大了"。

PlaNet 用 RSSM + Latent Overshooting 反驳了第一点，用"纯潜空间规划"反驳了第二点。它在 cheetah 上 500 个 episode 超过了 D4PG 100k 个 episode 的最终性能，这个数据效率差距是**无可辩驳的**。

### RSSM 的设计之美

从架构角度看，RSSM 的"deterministic + stochastic"双路径设计非常优雅。World Models 的 MDN-RNN 把 GRU 和 MDN 平行堆叠，本质上是一个黑箱。RSSM 则明确了两者的分工：GRU 负责记住过去，$s_t$ 负责建模当下不确定性。这就像卡尔曼滤波中的确定性状态转移 + 随机过程噪声——RSSM 相当于是**非线性、可学习的卡尔曼滤波**。

这也解释了为什么 RSSM 的消融实验中"缺任何一个路径都失败"：没有确定性路径，模型无法在物体离开视野后记住它（cartpole 推车离开画面）；没有随机性路径，模型无法表达"我不确定"而必须假装确定，长期预测会收敛到模糊均值。

### Latent Overshooting 的深层含义

Latent overshooting 表面上是训练技巧，实际上是**对"模型应该预测到什么"的根本性思考**。

标准的时序 ELBO 训练的是"给定历史，预测下一帧"，也就是推理性预测（inference）。但规划需要的是"给定任意历史起点，预测未来多步"，也就是生成性预测（generation）。Latent overshooting 将生成性预测纳入了训练目标，使得 $p(s_t \mid h_t)$ 先验不仅是一个正则器，实际上成为了一个**可 rollout 的生成模型**。

这一思想影响深远：DreamerV2 的 KL balancing（分别优化先验和后验的 KL 项）本质上是在处理同样的问题——让先验主动"追赶"后验，而不是被动地做正则器。到后来的 TD-MPC、IRIS，所有基于 latent dynamics 的工作都要面对这个 fundamental tension。

### 历史定位

从 2024 年回头看，PlaNet 的 CEM 规划路线后来被 Dreamer 的 Actor-Critic 路线大幅超越（数据效率、任务广度、计算效率都是 Dreamer 胜出），但这不妨碍 PlaNet 的历史地位：

- 它奠定了 RSSM 作为 latent world model 的**标准架构**；
- 它确立了"模型应在所有时间尺度上被训练"这一 latent overshooting 原则；
- 它是 Dreamer 系列的直接技术起点（Dreamer = RSSM + Actor-Critic）；
- 在没有显式策略、没有 pixel generation 的约束下，它完成了连续控制任务——这一"纯规划"范式的成功，至今仍有启发意义（思考：如果我们有足够好的世界模型，还需要策略网络吗？）。

一句话评价：**PlaNet 用最纯粹的方式回答了"如果有一个完美的世界模型，我们应该怎么用？"——而 Dreamer 则回答了"在模型不完美时，怎么用它学策略"。两者缺一不可。**
