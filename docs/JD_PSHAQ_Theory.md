# JD-PSHAQ: Jump Diffusion Portfolio-SHAQ
## 基于默顿跳跃扩散过程的多智能体 Web 测试信度分配算法

---

# 1. 核心建模理念 (Core Philosophy)

## 1.1 问题背景

Web 测试的奖励分布具有显著的**非高斯特性**：
- **稀疏性 (Sparsity)**: 99% 时间获得微小奖励
- **突发性 (Jumpiness)**: 偶发重大发现（Bug）导致价值跳跃

传统 RL 算法（DQN, QMIX）假设连续平滑的奖励分布，无法正确建模这种"尖峰厚尾"特性。

## 1.2 金融建模类比 (The Grand Mapping)

### 核心概念区分

在强化学习中，存在两个关键的价值函数：

| RL 概念 | 数学定义 | 金融类比 | 本质区别 |
|---------|----------|----------|----------|
| **状态价值** $V(s)$ | $V(s) = \max_a Q(s,a)$ | 资产价格 $S_t$ | 纯粹的"持有价值"，与动作无关 |
| **动作价值** $Q(s,a)$ | $Q(s,a) = \mathbb{E}[\sum_{t=0}^{\infty} \gamma^t r_t \mid s, a]$ | **期权/策略价值** | 执行特定动作后的条件期望 |

### 完整映射表

| Web Testing (RL) | Quantitative Finance | Symbol | 数学含义 |
|------------------|---------------------|--------|----------|
| Agent | 交易策略 (Trading Strategy) | - | 具有不确定回报的投资主体 |
| **$V(s) = \max_a Q(s,a)$** | **资产价格 (Asset Price)** | $S_t$ | 状态的内在价值（最优动作下的期望） |
| **$Q(s,a)$** | **美式期权价值 (American Option)** | $C(S,K,\tau)$ | 执行动作 $a$ 后的条件期望收益 |
| $r$ (即时奖励) | 股息/现金流 (Dividend) | $D_t$ | 即时收益流 |
| 累积奖励 $\sum r$ | 投资组合净值 (NAV) | $\sum D_t$ | 历史收益累计 |
| Q-Value 变化率 | 对数收益率 (Log Return) | $r_t = \ln(Q_t/Q_{t-1})$ | 连续复利增长率 |
| 常规探索 | 扩散波动 (Diffusion) | $\sigma$ | 连续高斯波动（日常操作） |
| 发现 Bug | 跳跃过程 (Jump) | $\lambda, J$ | 离散突发事件（稀有发现） |
| 信度分配 | 投资组合权重 | $w_i$ | 资源分配比例 |
| 探索红利 | 期权时间价值 (Time Value) | $\mathcal{B}$ | 对未来不确定性的定价 |
| 单边惩罚 | 下偏矩 (LPM) / Sortino | $\sigma_{down}$ | 只惩罚下行风险 |
| Performance Score | Alpha (超额收益) | $\alpha$ | 剔除市场平均后的独立贡献 |

### 为什么 $Q(s,a)$ 更像期权而非股票？

1. **条件性 (Contingency):** $Q(s,a)$ 是在执行动作 $a$ 的**条件下**的期望，类似于期权是在**行权条件下**的收益
2. **选择权 (Optionality):** Agent 可以选择不同的 $a$，就像期权持有者可以选择行权或放弃
3. **时间价值 (Time Value):** 未来探索的潜力 = 期权的时间价值

$$
Q(s,a) = \underbrace{\mathbb{E}[r \mid s,a]}_{\text{内在价值 (Intrinsic)}} + \underbrace{\gamma \mathbb{E}[V(s') \mid s,a]}_{\text{时间价值 (Time Value)}}
$$

---

# 2. 状态空间建模：默顿跳跃扩散 (Merton Jump Diffusion)

## 2.1 为什么建模 Q 值而非 V 值？

在 JD-PSHAQ 中，我们选择对 **动作价值 $Q(s,a)$** 而非状态价值 $V(s)$ 进行跳跃扩散建模，原因如下：

| 建模对象 | 优点 | 缺点 |
|----------|------|------|
| $V(s)$ | 与资产价格直接对应 | 丢失动作信息，无法区分策略差异 |
| $Q(s,a)$ | 保留动作条件性，可追踪策略演化 | 需要更精细的参数估计 |

**关键洞察：** 在 Web 测试中，同一页面状态 $s$ 下，不同动作 $a$（点击不同按钮）的价值差异巨大。$Q(s,a)$ 的跳跃反映了"发现正确的动作路径"，而非仅仅"到达新状态"。

## 2.2 随机微分方程 (SDE)

第 $i$ 个 Agent 的动作价值 $Q_i(s_t, a_t)$ 的时间演化服从以下 SDE：

$$
\frac{dQ_i(t)}{Q_i(t)} = \underbrace{(\mu_i - \lambda_i k_i) dt}_{\text{Drift (漂移项)}} + \underbrace{\sigma_i dW_i(t)}_{\text{Diffusion (扩散项)}} + \underbrace{(Y_i - 1) dN_i(t)}_{\text{Jump (跳跃项)}}
$$

**参数解释：**

| 符号 | 名称 | 金融含义 | Web Testing 含义 |
|------|------|----------|------------------|
| $Q_i(t)$ | 动作价值 | **期权价值** | 执行当前策略后的条件期望 |
| $\mu_i$ | 漂移率 (Drift) | 期权 Theta 衰减 + Delta 增益 | Agent 的长期学习效率 |
| $\sigma_i$ | 扩散波动率 | 隐含波动率 (IV) | 常规探索的不确定性 |
| $W_i(t)$ | 布朗运动 | 随机游走 | 日常操作的随机波动 |
| $\lambda_i$ | 跳跃强度 | 跳跃期权的跳跃频率 | Bug 发现/新路径频率 |
| $N_i(t)$ | 泊松过程 | 跳跃计数器 | 重大发现事件计数 |
| $Y_i$ | 跳跃幅度 | 跳跃导致的期权价值变化比例 | 发现带来的价值倍增 |
| $k_i$ | 补偿项 | 风险中性调整 | $k_i = \mathbb{E}[Y_i - 1]$ |

**与期权定价的深层联系：**

在 Black-Scholes 框架中，期权价值 $C$ 满足：
$$
C = S \cdot N(d_1) - K e^{-rT} N(d_2)
$$

当标的 $S$ 发生跳跃时，$C$ 也会发生非线性跳跃。类似地，当 Agent 发现新的高价值动作时，$Q(s,a)$ 会发生跳跃。

## 2.2 跳跃幅度分布

假设跳跃幅度服从**对数正态分布**：

$$
\ln Y_i \sim \mathcal{N}(\mu_{J,i}, \delta_{J,i}^2)
$$

其中：
- $\mu_{J,i}$: 跳跃对数均值
- $\delta_{J,i}$: 跳跃对数标准差

**补偿项计算：**

$$
k_i = \mathbb{E}[Y_i - 1] = e^{\mu_{J,i} + \frac{\delta_{J,i}^2}{2}} - 1
$$

## 2.3 泊松分布的理论依据

**小值定律 (Law of Rare Events):**

当试验次数 $n$ (操作步数) 很大，成功概率 $p$ (发现 Bug) 很小时：

$$
\lim_{n \to \infty, np \to \lambda} \binom{n}{k} p^k (1-p)^{n-k} = \frac{\lambda^k e^{-\lambda}}{k!}
$$

即二项分布弱收敛于泊松分布，这证明了用泊松过程建模 Bug 发现是数学最优选择。

---

# 3. HJB 方程与 Black-Scholes 的同源性

## 3.1 强化学习的 HJB 方程 (连续时间 MDP)

对于连续时间控制过程：
$$
dS_t = \mu(S_t, a_t)dt + \sigma(S_t, a_t)dW_t
$$

最优价值函数 $V^*(s, t)$ 满足 **Hamilton-Jacobi-Bellman 方程**：

$$
\frac{\partial V}{\partial t} + \max_{a} \left\{ \mu \frac{\partial V}{\partial s} + \frac{1}{2}\sigma^2 \frac{\partial^2 V}{\partial s^2} + r(s,a) \right\} - \rho V = 0
$$

## 3.2 金融的 Black-Scholes 方程

对于欧式期权 $V(S, t)$，当标的资产服从几何布朗运动时：

$$
\frac{\partial V}{\partial t} + rS \frac{\partial V}{\partial S} + \frac{1}{2}\sigma^2 S^2 \frac{\partial^2 V}{\partial S^2} - rV = 0
$$

## 3.3 同源性分析

| HJB 方程项 | Black-Scholes 项 | 物理含义 |
|-----------|-----------------|----------|
| $\frac{\partial V}{\partial t}$ | $\Theta$ (Theta) | 时间衰减 |
| $\mu \frac{\partial V}{\partial s}$ | $\Delta$ (Delta) | 一阶敏感度（漂移贡献） |
| $\frac{1}{2}\sigma^2 \frac{\partial^2 V}{\partial s^2}$ | $\Gamma$ (Gamma) | 二阶敏感度（波动贡献） |
| $-\rho V$ | $-rV$ | 折现/贴现项 |

**结论：** Bellman Error 本质上就是 BS 方程中 Greeks 的残差。

## 3.4 JD-PSHAQ 的 PIDE 扩展

引入跳跃项后，价值函数满足 **Partial Integro-Differential Equation (PIDE)**：

$$
\underbrace{\frac{\partial V}{\partial t} + \mathcal{L}_{diff}V}_{\text{扩散部分}} + \underbrace{\lambda \mathbb{E}[V(S \cdot J, t) - V(S, t)]}_{\text{跳跃积分项}} - rV = 0
$$

其中微分算子：
$$
\mathcal{L}_{diff}V = (\mu - \lambda k)S\frac{\partial V}{\partial S} + \frac{1}{2}\sigma^2 S^2 \frac{\partial^2 V}{\partial S^2}
$$

**理论意义：** 积分项显式捕捉了 Web 测试中的非局部跳跃，这是标准 DQN（假设连续路径）无法建模的。

---

# 4. 风险度量：JD-Sortino Ratio (单边惩罚)

## 4.1 为什么需要单边惩罚？

### 分布特性分析

Web 测试回报分布是 **极度右偏 (Right-Skewed)** 的：
- **左尾 (下行):** 较短，最差就是 0 分
- **右尾 (上行):** 非常长，可能发现重大 Bug 拿高分

### Sharpe Ratio 的缺陷

标准方差惩罚所有波动：
$$
\text{Var}(X) = \mathbb{E}[(X - \mu)^2]
$$

大的正向收益会增大方差，导致 Sharpe 降低。**这在 RL 中是荒谬的——你会因为 Agent 表现太好而惩罚它。**

## 4.2 下偏矩 (Lower Partial Moment, LPM)

定义最低可接受回报 $MAR$ (Minimum Acceptable Return)，下偏矩为：

$$
\text{LPM}_n = \int_{-\infty}^{MAR} (MAR - x)^n f(x) dx
$$

对于 $n=2$（二阶下偏矩）：

$$
\sigma_{down}^2 = \mathbb{E}\left[\min(0, r - MAR)^2\right]
$$

## 4.3 JD-Sortino Ratio 公式

在跳跃扩散模型下，总下行风险分解为：

$$
\sigma_{i,down}^2 = \underbrace{\sigma_{i,diff}^2 \cdot \mathbb{I}_{\{\mu_i < MAR\}}}_{\text{扩散下行风险}} + \underbrace{\lambda_i^{-} \cdot \mathbb{E}[(J_{down})^2]}_{\text{负向跳跃风险}}
$$

**JD-Sortino 比率：**

$$
\text{JD-Sortino}_i = \frac{\mathbb{E}[r_i] - MAR}{\sqrt{\sigma_{i,down}^2 + \epsilon}}
$$

**关键性质：** 只惩罚负向跳跃 $\lambda^-$，忽略正向跳跃 $\lambda^+$（发现 Bug 是好事！）

## 4.4 与金融的对应

| JD-PSHAQ 概念 | 金融概念 | 公式 |
|---------------|----------|------|
| 下行风险 | Value at Risk (VaR) | $P(L > VaR) = \alpha$ |
| JD-Sortino | Sortino Ratio | $\frac{R_p - MAR}{\sigma_{down}}$ |
| 只惩罚负跳跃 | Put Option Pricing | 只为下行风险付费 |

---

# 5. 探索激励：期权定价理论

## 5.1 探索即期权 (Exploration as an Option)

探索未知状态就像持有**看涨期权 (Call Option)**：
- 失败：损失有限（时间成本）
- 成功：收益无限（发现重大 Bug）

根据 Merton (1976) 期权定价，高波动率资产的期权价值更高。

## 5.2 探索红利公式

**完整 Merton 公式（用于理论分析）：**

$$
\mathcal{C}_{Merton} = \sum_{n=0}^{\infty} \frac{e^{-\lambda'\tau}(\lambda'\tau)^n}{n!} \cdot \mathcal{C}_{BS}(S, K, r_n, \sigma_n, \tau)
$$

其中：
- $\lambda' = \lambda(1 + k)$: 调整后跳跃强度
- $\sigma_n^2 = \sigma^2 + n\sigma_J^2/\tau$: 调整后波动率

**v3.0 简化公式（用于实际计算）：**

$$
\mathcal{B}_i = \beta \cdot \left( \lambda_i^+ \cdot \mathbb{E}[Y^+] \cdot \tau + 0.5 \cdot \sigma_i \sqrt{\tau} \right)
$$

**物理含义：** 
- 第一项：正向跳跃潜力（发现 Bug 的期望价值）
- 第二项：扩散上行潜力（常规探索的期望收益）

## 5.3 加法奖励重构

**v3.0 核心改动：** 使用加法而非乘法

$$
R_{i,final} = R_{i,base} + \beta \cdot \mathcal{B}_i
$$

**梯度稳定性证明：**

设损失函数 $L$，则梯度：
$$
\nabla L \propto R_{final} \cdot \nabla Q
$$

- **乘法 ($R = w \cdot R_{base}$):** 若 $w$ 波动大，梯度爆炸/消失
- **加法 ($R = R_{base} + \mathcal{B}$):** 即使 $\mathcal{B}$ 波动，不破坏原始结构

---

# 6. 权重分配与 Shapley Value

## 6.1 Shapley Value 的博弈论定义

在合作博弈 $(N, v)$ 中，Agent $i$ 的 Shapley Value：

$$
\phi_i(v) = \sum_{S \subseteq N \setminus \{i\}} \frac{|S|!(|N|-|S|-1)!}{|N|!} \left[ v(S \cup \{i\}) - v(S) \right]
$$

**含义：** 按所有排列顺序，计算 $i$ 加入团队时的边际贡献期望。

## 6.2 SHAQ 的 Lovász 扩展近似

计算真正的 Shapley 需要 $2^N$ 次运算。SHAQ 利用 **Lovász 扩展** 的性质：

如果价值函数 $V_{tot}(Q_1, ..., Q_n)$ 是凸的，则：

$$
\phi_i(s) \approx \frac{\partial V_{tot}(Q)}{\partial Q_i}
$$

即：**Shapley Value ≈ 联合 Q 值对个体 Q 值的梯度**

## 6.3 JD-PSHAQ 的融合架构

$$
\text{Final Weight}_i = \underbrace{\phi_i}_{\text{SHAQ: 过去贡献}} + \underbrace{\beta \cdot \text{Option}(\sigma_i, \lambda_i)}_{\text{JD: 未来潜力}}
$$

| 组件 | 来源 | 度量维度 | 含义 |
|------|------|----------|------|
| $\phi_i$ | SHAQ (Shapley) | 一阶矩 (均值) | "谁刚才干得好" |
| $\mathcal{B}_i$ | JD (Option) | 二阶矩 (波动) | "谁接下来可能干得好" |

## 6.4 Softmax + Temperature 机制

**综合权重计算：**

$$
score_i = \alpha \cdot \text{JD-Sortino}_i + \beta \cdot \text{Option}_i + \gamma \cdot \text{InfoPremium}_i
$$

$$
w_i^{raw} = \max(0, \phi_i + \epsilon) \cdot \exp\left(\frac{score_i - \max_j score_j}{\tau}\right)
$$

$$
w_i = \frac{w_i^{raw}}{\sum_j w_j^{raw}}
$$

**温度参数 $\tau$ 的作用：**
- $\tau \to \infty$: 权重趋于均匀（保守探索）
- $\tau \to 0$: 权重趋于 one-hot（赢者通吃）

---

# 7. 理论优势证明

## 7.1 对稀疏奖励的敏感性

**命题：** 当奖励服从重尾分布时，JD-PSHAQ 收敛于帕累托最优，而标准 RL 是次优的。

**证明思路：**
1. 标准 RL 优化 $\mathbb{E}[R]$，样本均值被噪声掩盖
2. JD-PSHAQ 分离 $W_t$（连续）和 $N_t$（跳跃）
3. 高 $\lambda$ Agent 获得更高权重
4. 梯度更新：$\Delta\theta \propto \lambda$，而非 $\Delta\theta \propto \frac{1}{N}\sum r$

## 7.2 避免探索退化

**命题：** 即使 $\epsilon \to 0$，JD-PSHAQ 仍能自适应增加探索。

**证明：**
1. 新状态出现 → Q 值跳跃
2. Tracker 检测到 → $\lambda_t, \sigma_t$ 增大
3. 探索红利 $\mathcal{B}_i \propto \sqrt{\sigma^2 + \lambda(\cdot)}$ 增大
4. $R_{final}$ 增大 → 自然吸引 Agent 探索

---

# 8. 完整工作流程

```
┌─────────────────────────────────────────────────────────────┐
│                    JD-PSHAQ v3.0 Workflow                    │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐  │
│  │   Observe    │───▶│   Estimate   │───▶│   Evaluate   │  │
│  │  Q(s,a), r   │    │  μ, σ, λ     │    │Sortino, Option│  │
│  └──────────────┘    └──────────────┘    └──────────────┘  │
│         │                   │                   │           │
│         ▼                   ▼                   ▼           │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐  │
│  │ JD Tracker   │    │Method of     │    │JD-Sortino    │  │
│  │ Update       │    │Moments       │    │Calculation   │  │
│  └──────────────┘    └──────────────┘    └──────────────┘  │
│                                                              │
│                          ▼                                   │
│                ┌──────────────────┐                         │
│                │   Weight Calc    │                         │
│                │  w_i = Softmax   │                         │
│                │  (φ_i × exp(s/τ))│                         │
│                └──────────────────┘                         │
│                          │                                   │
│                          ▼                                   │
│                ┌──────────────────┐                         │
│                │ Reward Shaping   │                         │
│                │R_final = R + βB  │                         │
│                └──────────────────┘                         │
│                          │                                   │
│                          ▼                                   │
│                ┌──────────────────┐                         │
│                │   DQN Update     │                         │
│                │  θ ← θ - α∇L    │                         │
│                └──────────────────┘                         │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

---

# 9. 符号汇总表

| 符号 | 定义 | 范围/单位 |
|------|------|-----------|
| $Q_i(s,a)$ | Agent $i$ 的动作价值函数（类比期权价值） | $\mathbb{R}^+$ |
| $V_i(s)$ | Agent $i$ 的状态价值函数（类比资产价格） | $\mathbb{R}^+$ |
| $\mu_i$ | 漂移率 | $[-1, 1]$ |
| $\sigma_i$ | 扩散波动率 | $[0, 1]$ |
| $\lambda_i$ | 跳跃强度 | $[0, 1]$ |
| $\lambda_i^+$ | 正向跳跃强度 | $[0, 1]$ |
| $\lambda_i^-$ | 负向跳跃强度 | $[0, 1]$ |
| $Y_i$ | 跳跃幅度 | LogNormal |
| $\phi_i$ | Performance Score | $[0, 1]$ |
| $\mathcal{B}_i$ | 探索红利 | $[0, 5]$ |
| $w_i$ | 最终权重 | $[0, 1], \sum w_i = 1$ |
| $\tau$ | 温度参数 | $[0.1, 10]$ |
| $MAR$ | 最低可接受回报 | 0 |

---

# 10. 参考文献

1. Merton, R.C. (1976). "Option Pricing When Underlying Stock Returns Are Discontinuous". *Journal of Financial Economics*.

2. Kou, S.G. (2002). "A Jump-Diffusion Model for Option Pricing". *Management Science*.

3. Wang, J. et al. (2020). "SHAQ: Incorporating Shapley Value Theory into Multi-Agent Q-Learning". *NeurIPS*.

4. Sortino, F.A. & Van der Meer, R. (1991). "Downside risk". *Journal of Portfolio Management*.

5. Black, F. & Scholes, M. (1973). "The Pricing of Options and Corporate Liabilities". *Journal of Political Economy*.

---

**Document Version:** v3.0  
**Last Updated:** 2026-02-03  
**Author:** JD-PSHAQ Research Team
