# Recurrent World Models Facilitate Policy Evolution

- **标题**: Recurrent World Models Facilitate Policy Evolution
- **作者**: David Ha, Jürgen Schmidhuber
- **发表**: NIPS 2018 (Oral)
- **链接**: [arXiv](https://arxiv.org/abs/1809.01999) | [Code](https://github.com/hardmaru/WorldModelsExperiments) | [Project Page](https://worldmodels.github.io)

---

## 1. Paper Summary

提出 **World Model** 范式，将强化学习环境建模拆解为三个独立训练的组件：

| 组件 | 功能 | 算法 |
|------|------|------|
| **V (Vision)** | 像素 → 潜编码 | VAE |
| **M (Memory)** | 潜编码 → 未来预测 | MDN-RNN |
| **C (Controller)** | 潜编码 + 隐状态 → 动作 | CMA-ES 进化策略 |

核心流程：随机策略采集数据 → V 学习潜空间 → M 学习动态预测 → C 在 M 的"梦境"中进化 → 迁移回真实环境。

关键成果：首次展示**完全在模型内部训练策略**（不接触真实环境），且成功泛化到真实环境（CarRacing）；在 Doom 和 Atari 多个环境达到当时最优。

---

## 2. Motivation

- **DRL 样本效率极低**：需要数百万到数十亿时间步才能收敛，实际应用成本高
- **MBRL 的模型误差累积**：学到的环境模型一旦有偏差，基于梯度的策略优化会放大这些误差，导致策略在真实环境中失效
- **端到端训练不稳定**：同时学习感知、记忆、决策三个高度不同时间尺度的模块，联合训练难度大
- **之前的 work 不完整**：虽有 RNN-based 环境建模研究，但缺乏感知压缩的关键步骤，也没有展示"完全在模型内部训练策略"的能力

核心洞察：既然人类可以在大脑中"想象"并预演行动后果，AI 也应该先学会预测世界，再在其中练习决策。

---

## 3. Method

### 3.1 整体架构

```
                      ┌──────────────────┐
   x_t (像素) ──────→ │  Vision (VAE)     │ ──→ z_t (潜编码)
                      └──────────────────┘
                              │
                              ↓
                      ┌──────────────────┐
   z_t + a_t + h_t ──→ │  Memory (MDN-RNN) │ ──→ h_{t+1}, p(z_{t+1}|...)
                      └──────────────────┘
                              │
                              ↓
                      ┌──────────────────┐
   z_t + h_t ────────→ │  Controller (C)  │ ──→ a_t (动作)
                      └──────────────────┘
```

### 3.2 训练流程

```
Step 1 — Data Collection:
    Random Policy → Rollout N 条轨迹 → 存储 (x_t, a_t, r_t) 序列

Step 2 — Train V:
    对每帧 x_t:  ConvEncoder(x_t) → μ_t, σ_t → z_t ~ N(μ, σ²) → ConvDecoder(z_t) → x̂_t
    Loss: 重建损失 (MSE/Bernoulli) + β · KL[N(μ,σ²) ‖ N(0,I)]

Step 3 — Train M:
    对每步 (z_t, a_t, h_t):  GRU(z_t, a_t, h_t) → h_{t+1} → MDN(h_{t+1}) → π(z_{t+1})
    Loss: -log Σ_k w_k · N(z_{t+1} | μ_k, σ_k²)   (负对数似然)

Step 4 — Train C (Dream):
    for generation in 1..G:
        z_0 = V(x_0), h_0 = 0
        for t in 1..T:
            a_t = C(z_t, h_t)
            z_{t+1} ~ M(z_t, a_t, h_t)   ← 在梦中采样
            h_{t+1} = M.GRU(z_t, a_t, h_t)
        fitness = Σ r_t
    CMA-ES 根据 fitness 更新 C 的权重
```

### 3.3 组件详解

**V (VAE)**: 将 64×64×3 像素压缩到 ~32 维潜变量。架构：ConvEncoder(4层 Conv) → FC → μ, σ → reparameterize → FC → ConvDecoder(4层 DeConv)。

**M (MDN-RNN)**: GRU 隐状态 256 维；MDN 输出 5 个 Gaussian 混合（每个 5 组分 × 32 维均值 + 32 维方差 + 1 个权重）。权重的 softmax、方差的 exp 确保正值。

**C (Controller)**: 单层线性层 + tanh，将 z_t 和 h_t 拼接后映射到动作空间。参数数量 ~1000，用 CMA-ES 以 population size = 64 进化。

---

## 4. Mathematical Formulation

### 符号表

| 符号 | 含义 | 维度 |
|------|------|------|
| $x_t$ | t 时刻观测（像素帧） | $64 \times 64 \times 3$ |
| $z_t$ | VAE 编码的潜变量 | $\mathbb{R}^{32}$ |
| $a_t$ | 动作 | $\mathbb{R}^{\vert\mathcal{A}\vert}$ |
| $h_t$ | RNN 隐状态 | $\mathbb{R}^{256}$ |
| $w_k, \mu_k, \sigma_k$ | MDN 第 k 个混合组分参数 | — |

### 4.1 VAE 损失

$$
\mathcal{L}_{\text{VAE}} = \underbrace{\mathbb{E}_{q(z_t|x_t)}[\log p(x_t|z_t)]}_{\text{重建损失}} \;-\; \beta \cdot \underbrace{D_{\text{KL}}\big(q(z_t|x_t) \,\|\, \mathcal{N}(0, I)\big)}_{\text{KL 散度}}
$$

重建项衡量解码质量（MSE 或 Bernoulli 交叉熵），KL 项约束潜空间分布接近标准正态，$\beta$ 控制正则强度（实验中 $\beta = 0.5$）。

### 4.2 MDN-RNN 损失

输出 $K$ 个 Gaussian 混合的加权和：

$$
p(z_{t+1} \mid z_t, a_t, h_t) = \sum_{k=1}^K \pi_k(z_t, a_t, h_t) \cdot \mathcal{N}\big(z_{t+1} \mid \mu_k(z_t, a_t, h_t), \sigma_k^2(z_t, a_t, h_t)\big)
$$

其中 $\pi_k$ 是权重（softmax 归一化），$\mu_k$ 是均值，$\sigma_k$ 是标准差（exp 确保正值）。

训练目标为负对数似然：

$$
\mathcal{L}_{\text{MDN}} = -\sum_{t=1}^T \log \sum_{k=1}^K \pi_k^{(t)} \cdot \frac{1}{\sqrt{(2\pi)^D |\Sigma_k^{(t)}|}} \exp\left(-\frac{1}{2}(z_{t+1} - \mu_k^{(t)})^\top \Sigma_k^{(t)-1} (z_{t+1} - \mu_k^{(t)})\right)
$$

在实际实现中，$\sigma_k$ 通常假设为对角协方差，上式可化简为：

$$
\mathcal{L}_{\text{MDN}} = -\sum_{t=1}^T \log \sum_{k=1}^K \pi_k^{(t)} \cdot \prod_{d=1}^D \frac{1}{\sqrt{2\pi}\,\sigma_{k,d}^{(t)}} \exp\left(-\frac{(z_{t+1,d} - \mu_{k,d}^{(t)})^2}{2\sigma_{k,d}^{(t)2}}\right)
$$

**直观理解**：MDN 相当于让 RNN 输出多个"假设"（每个 Gaussian 是一个假设），然后让真实数据去选择哪个假设最合理。

### 4.3 CMA-ES 策略优化

Controller C 的权重参数 $\theta$ 通过 CMA-ES 优化，最大化期望累积奖励：

$$
\theta^* = \arg\max_\theta \mathbb{E}_{\tau \sim p_M(\tau | \theta)}\left[\sum_{t=1}^T r_t\right]
$$

其中轨迹 $\tau = (z_1, a_1, r_1, ..., z_T, a_T, r_T)$ **完全在 M 模型生成的梦境中采样**。CMA-ES 是一种黑箱优化算法，维护多元正态分布 $\mathcal{N}(\theta_{\text{mean}}, \sigma^2 C)$，迭代更新均值和协方差矩阵，不依赖梯度。

### 4.4 执行策略（ rollout）

在梦境中采样时，每次从 MDN 采样 $z_{t+1}$：

$$
z_{t+1} \sim \sum_{k=1}^K \pi_k \cdot \mathcal{N}(\mu_k, \sigma_k^2)
$$

这个过程引入了随机性，使得同一个起点可以产生多样化的轨迹，有助于 Controller 学到鲁棒策略。

---

## 5. Implementation

官方实现（TensorFlow）：[github.com/hardmaru/WorldModelsExperiments](https://github.com/hardmaru/WorldModelsExperiments)

### 关键代码结构

```
world_models/
├── vae.py           # VAE: ConvEncoder + Decoder, 训练 vae.py
├── rnn.py           # MDN-RNN: GRU + MDN, 训练 rnn.py
├── controller.py    # Controller: 线性策略 + CMA-ES
├── environment.py   # 环境封装 (CarRacing, Doom, etc.)
├── dream.py         # 梦境 rollout: 用 M 生成虚拟轨迹
├── train.py         # 主训练循环
└── utils.py         # 数据处理、可视化等
```

### 核心训练循环（伪代码）

```python
# Step 1: 数据采集
dataset = collect_rollouts(env, policy=random, n_episodes=10000)

# Step 2: 训练 VAE
vae = VAE(latent_dim=32, beta=0.5)
vae.train(dataset.frames, epochs=50, batch_size=32)

# Step 3: 编码数据 + 训练 MDN-RNN
z_dataset = vae.encode(dataset.frames)
mdn_rnn = MDNRNN(hidden_dim=256, num_gaussians=5)
mdn_rnn.train((z_dataset, dataset.actions), epochs=20, batch_size=32)

# Step 4: 梦境中进化 Controller
controller = Controller(input_dim=32+256, output_dim=n_actions)
cma_es = CMAES(controller.parameters, population_size=64)

for generation in range(1000):
    # 在梦境中评估每个个体
    for individual in cma_es.ask():
        controller.load(individual)
        total_reward = dream_rollout(env_init, vae, mdn_rnn, controller, steps=1000)
        individual.fitness = total_reward
    cma_es.tell()

# Step 5: 真实环境测试
test(env, vae, mdn_rnn, controller)
```

## 6. Evolution

### 历史脉络

```
Robot Learning (2013)   ← Hamster 等早期的 generative model + RL
        │
        ▼
μGAN (2016)             ← Ha 前作：生成模型辅助 RL
        │
        ▼
★ World Models (2018)   ← 本文：V + M + C 三组件范式
        │
        ├──→ PlaNet (2019)         ← 潜空间 + 确定性 + 随机路径合并
        ├──→ Dreamer (2019)       ← 潜空间 Actor-Critic 替代进化策略
        ├──→ DreamerV2 (2020)     ← RSSM + categorical VAE
        ├──→ DreamerV3 (2022)     ← 大规模通用 agent
        └──→ DayDreamer (2022)    ← 真实机器人上的 world model
```

### 承上

- 受早期 **μGAN** 启发：首次验证了用生成模型辅助 RL 的可能性，但当时只做了图像生成，没有完整的时序建模
- 受 **RNN 世界模型** 传统影响：Schmidhuber 1990 年就提出了"建模世界然后规划"的理念，本文用深度学习将此想法落地

### 启下

- **PlaNet**（Hafner et al., 2019）在 M 组件上做了关键改进：引入确定性 + 随机路径合并（RSSM），解决 MDN 在高维连续空间中的建模困难；用规划代替 Controller
- **Dreamer**（Hafner et al., 2019）进一步用 Actor-Critic（基于梯度）替代 CMA-ES（黑箱进化），大幅提升训练效率
- **DreamerV2/V3** 延续了 RSSM + Actor-Critic 路线，逐步扩展到大规模通用 agent
- 这篇论文标志着 **"潜空间世界模型 + 梦境训练"** 范式的诞生，是后续 world model 研究的奠基之作
