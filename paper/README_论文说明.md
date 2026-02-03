# JD-PSHAQ 论文说明

## 论文结构概览

```
JD_PSHAQ_Paper.tex
├── 1. Abstract (摘要)
│   └── 问题、方法、贡献、结果的精炼概括
│
├── 2. Introduction (引言)
│   ├── 背景：Web测试的重要性
│   ├── Challenge 1: 信度分配模糊
│   ├── Challenge 2: 探索-利用困境
│   ├── Key Insight: 金融-RL类比
│   └── Contributions: 四大贡献点
│
├── 3. Background & Related Work (背景与相关工作)
│   ├── RL for Web Testing
│   ├── Exploration Strategies
│   ├── Jump Diffusion in Finance
│   └── Shapley Value in MARL
│
├── 4. Theoretical Framework (理论框架) ⭐核心
│   ├── 4.1 Bellman → HJB 连续时间极限
│   ├── 4.2 Q-Value 作为随机过程建模
│   ├── 4.3 参数估计（矩方法）
│   ├── 4.4 探索红利 = 期权价值
│   ├── 4.5 单边惩罚（Sortino）的合理性
│   └── 4.6 完整映射表
│
├── 5. JD-PSHAQ Algorithm (算法)
│   ├── 5.1 架构概览 (CTDE)
│   ├── 5.2 奖励整形公式
│   ├── 5.3 Shapley 值估计
│   ├── 5.4 训练流程（伪代码）
│   └── 5.5 计算复杂度分析
│
├── 6. Experiments (实验)
│   ├── RQ1: 总体性能对比
│   ├── RQ2: 消融实验
│   ├── RQ3: JD假设验证
│   └── RQ4: 参数敏感性
│
├── 7. Discussion (讨论)
│   ├── Why it works (三个机制)
│   ├── Limitations (三个局限性)
│   │   ├── Poisson 假设
│   │   ├── 离散-连续近似
│   │   └── 可扩展性
│   └── Threats to Validity
│
├── 8. Conclusion (结论)
│
└── Appendix (附录)
    ├── 定理证明
    ├── 实现细节
    └── 补充实验
```

## 需要你补充的内容

### 1. 实验数据（最关键）

论文中所有 `--` 和 `[实验数据待填充]` 的位置都需要跑完实验后填入真实数据：

```latex
% Table 3: RQ1 主实验结果
IQL & 156±23 & 3.2±1.1 & 2345±312 & 15k \\
VDN & 178±31 & 4.1±1.3 & 2567±287 & 12k \\
...
JD-PSHAQ & 234±28 & 6.8±1.5 & 3456±298 & 8k \\
```

### 2. 图表

需要生成的图表：
- `fig:architecture` - 系统架构图
- Q值轨迹图（展示跳跃）
- 分布拟合图（正态 vs MJD）
- 参数敏感性曲线
- 消融实验柱状图

### 3. Benchmark 信息

```latex
% Table 1: 测试网站信息
\begin{tabular}{lccc}
Website & Domain & States & Known Bugs \\
GitHub & Social & ~500 & 12 \\
...
\end{tabular}
```

## 理论亮点（可重点强调）

### 1. Bellman-HJB 对应（第3.1节）

这是全文的理论基石。说明了为什么可以用连续时间SDE来建模离散RL：

$$\text{Bellman: } V(s) = \max_a [r + \gamma \mathbb{E}[V(s')]]$$
$$\downarrow \text{ as } \Delta t \to 0$$
$$\text{HJB: } \rho V = \max_a [r + \mathcal{L}^a V]$$

### 2. 探索红利 = 期权价值（第3.4节）

核心创新点。将期权定价中的"波动率溢价"转化为RL中的"探索红利"：

$$B_i = \beta \cdot (\sigma_i + \lambda_i \cdot |J|)$$

**金融直觉**：波动率越高，期权越值钱（因为有更大的上涨潜力）
**RL直觉**：不确定性越高，探索价值越大（因为可能有未发现的好状态）

### 3. Sortino 单边惩罚（第3.5节）

解决了"波动率悖论"——为什么我们喜欢向上波动但讨厌向下波动：

$$\text{Sortino} = \frac{\mathbb{E}[r]}{\sqrt{\mathbb{E}[\min(r, 0)^2]}}$$

只惩罚负回报，不惩罚正回报（发现Bug）。

## 审稿人可能的问题及回应

### Q1: "为什么不直接用 UCB 或 RND？"

**回应**：UCB 基于访问计数，RND 基于预测误差，都是"静态"的不确定性度量。JD-PSHAQ 提供了"动态"的、基于时间序列的不确定性估计，能捕捉Q值演化的temporal pattern。

### Q2: "Poisson 假设太强了"

**回应**：见 Discussion 6.2。我们在 RQ3 中验证了经验分布与理论分布的拟合度。未来可扩展到 Hawkes Process。

### Q3: "Shapley 值计算太贵"

**回应**：我们使用梯度近似（Lovász 扩展），复杂度 O(N) 而非 O(2^N)。见 Algorithm 1 第18行。

### Q4: "实验只在 Web Testing 上做，泛化性？"

**回应**：JD-PSHAQ 的理论框架是通用的，可应用于任何具有"稀疏高价值事件"的MARL场景（游戏、机器人探索等）。

## 投稿建议

### 目标会议/期刊

| 方向 | 会议 | 截稿日期 |
|------|------|---------|
| SE | ICSE, FSE, ASE, ISSTA | 查看CFP |
| AI/ML | NeurIPS, ICML, ICLR | 查看CFP |
| MARL | AAMAS | 查看CFP |

### 页数要求

- IEEE 格式：10-12页（不含参考文献）
- ACM 格式：10页（含参考文献）

当前论文框架约 8 页，加上实验图表后约 11-12 页。

## 编译方法

```bash
# 方法1: pdflatex
pdflatex JD_PSHAQ_Paper.tex
bibtex JD_PSHAQ_Paper
pdflatex JD_PSHAQ_Paper.tex
pdflatex JD_PSHAQ_Paper.tex

# 方法2: Overleaf
# 直接上传 .tex 文件
```

## 文件清单

```
paper/
├── JD_PSHAQ_Paper.tex      # 主论文
├── README_论文说明.md       # 本文件
├── figures/                # 待创建：存放图片
│   ├── architecture.pdf
│   ├── q_trajectory.pdf
│   └── ...
└── IEEEtran.cls           # IEEE 模板（如需要）
```
