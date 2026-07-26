# 世界模型学习

World Models 相关论文的阅读笔记与公式推导。

## 仓库结构

```
.
├── README.md                ← 本页（论文清单 + 结构说明）
├── papers/
│   ├── _template/           ← 新建论文时复制此目录
│   │   └── README.md        ← 模板
│   │
│   ├── 01-world-models/     ← World Models (Ha & Schmidhuber, NIPS 2018)
│   │   ├── Recurrent World Models Facilitate Policy Evolution.md
│   │   └── Recurrent World Models Facilitate Policy Evolution.pdf
│   ├── 02-planet/           ← PlaNet (Hafner et al., ICML 2019)
│   │   ├── Learning Latent Dynamics for Planning from Pixels.md
│   │   └── Learning Latent Dynamics for Planning from Pixels.pdf
│   ├── 03-dreamer v1/       ← Dreamer (Hafner et al., ICLR 2020)
│   │   ├── Dream to Control_ Learning Behaviors by Latent Imagination.md
│   │   └── Dream to Control_ Learning Behaviors by Latent Imagination.pdf
│   └── 04-a2world/          ← A2World (Huang et al., ECCV 2026)
│       ├── Learning Transferable Dynamics Priors from Action to World Modeling.md
│       └── Learning Transferable Dynamics Priors from Action to World Modeling.pdf
```

每篇论文一个子目录，笔记和 PDF 均以论文标题命名。

---

## 论文清单

| # | 论文 | 状态 |
|---|------|------|
| 1 | [Recurrent World Models Facilitate Policy Evolution](papers/01-world-models/Recurrent%20World%20Models%20Facilitate%20Policy%20Evolution.md) — Ha & Schmidhuber, NIPS 2018 | ✅ 已完成 |
| 2 | [Learning Latent Dynamics for Planning from Pixels](papers/02-planet/Learning%20Latent%20Dynamics%20for%20Planning%20from%20Pixels.md) — Hafner et al., ICML 2019 | ✅ 已完成 |
| 3 | [Dream to Control: Learning Behaviors by Latent Imagination](papers/03-dreamer%20v1/Dream%20to%20Control_%20Learning%20Behaviors%20by%20Latent%20Imagination.md) — Hafner et al., ICLR 2020 | 📝 待读 |
| 4 | [Learning Transferable Dynamics Priors from Action to World Modeling](papers/04-a2world/Learning%20Transferable%20Dynamics%20Priors%20from%20Action%20to%20World%20Modeling.md) — Huang et al., ECCV 2026 | 📝 待读 |

> 状态: 📝 待读 / 🔍 阅读中 / ✅ 已完成

---

## 更新日志

| 日期 | 内容 |
|------|------|
| 2026-07-08 | 初始化仓库结构，创建笔记模板 |
| 2026-07-08 | 添加第一篇论文：Recurrent World Models Facilitate Policy Evolution（Ha & Schmidhuber, NIPS 2018），完成笔记 |
| 2026-07-18 | 添加第二篇论文：Learning Latent Dynamics for Planning from Pixels（Hafner et al., ICML 2019），完成笔记 |
| 2026-07-18 | 添加第三篇论文：Dream to Control: Learning Behaviors by Latent Imagination（Hafner et al., ICLR 2020），创建骨架 |
| 2026-07-18 | 添加第四篇论文：Learning Transferable Dynamics Priors from Action to World Modeling（Huang et al., ECCV 2026），创建骨架 |
