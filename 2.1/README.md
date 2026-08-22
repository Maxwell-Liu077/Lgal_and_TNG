# Phase 2.1：AGN 双重抑制与低红移冷气体循环研究设计

## 1. 研究说明

Phase 2.1 是 Phase 1.8 的延续，但不再只比较相邻快照端点的直接空间供给率，
而是同时记录中央冷库的直接进出、中央原位冷却/加热及其延迟吸积或反馈贡献。

## 2. 样本设计

### 2.1 基本选择

- 模拟：TNG50-1。
- 选择快照：snap99。
- 星系类型：FoF 中心星系，即 `GroupFirstSub`，并要求 `SubhaloFlag=True`。
- 恒星质量：沿用 Phase 1.8 的
  `SubhaloMassInRadType[:, 4]`，即 $2R_{\star,1/2}$ 内的恒星质量。
- 质量范围：
  $8.5\leq\log_{10}(M_\star/M_\odot)\leq11.5$。
- 抽样上限：每个最终质量箱最多 50 个；候选少于 50 个时全部保留。
- 固定随机种子：`202608`。
- 抽样顺序必须确定且可复现；候选在随机抽样前按 `SubfindID` 排序。

### 2.2 质量箱

最终箱边界为：

```text
[8.50, 8.75), [8.75, 9.00), ...,
[10.75, 11.00), [11.00, 11.50]
```

最后两个原始 0.25 dex 箱
`[11.00, 11.25)` 与 `[11.25, 11.50]` 合并为一个 0.5 dex 箱。
必须先合并候选池，再从合并后的 `[11.00, 11.50]` 中抽取最多 50 个；不能先
分别抽取最多 50 个再合并，否则最终箱可能含 100 个对象。

所有内部边界采用左闭右开，只有最后一个边界包含 11.50。样本文件必须记录
每箱候选数、抽样数、随机种子、质量定义和候选排序方式。

### 2.3 主支追踪

从 snap99 对象沿 SubLink 主祖支追踪到 snap90。每个快照使用该主祖对象当时
的中心、FoF 晕和 $R_{200c}$，不能固定使用 snap99 的中心或半径。

snap94 或 snap95 缺失、中心或 $R_{200c}$ 非物理时拒绝该晕；其他历史快照
若无法读取则保留锚点事件，并将受影响 tracer 标记为 `unresolved`，计入 `other`。

## 3. 时间设计

TNG50-1 官方输出时间为：

| 快照 | 红移 | 宇宙年龄 [Gyr] | 在本研究中的作用 |
|---:|---:|---:|---|
| 90 | 0.11 | 12.337 | 进入者及原位冷却者的前史起点 |
| 94 | 0.06 | 12.993 | 事件窗口左端、离开队列起点 |
| 95 | 0.05 | 13.127 | 事件窗口右端、进入队列终点 |
| 99 | 0.00 | 13.803 | 离开者及原位加热者的后验终点与样本选择快照 |

由此得到：

| 区间 | 时间长度 [Gyr] | 作用 |
|---|---:|---|
| snap90 $→$ 94 | 0.656 | 进入者及原位冷却者前史 |
| snap94 $→$ 95 | 0.134 | 唯一流率测量窗口 |
| snap95 $→$ 99 | 0.676 | 离开者及原位加热者后验 |
| snap90 $→$ 99 | 1.466 | 最近气体循环环境 |

## 4. 统一空间、相态与载体定义

### 4.1 空间区域

每个快照使用周期性边界距离，并区分中央空间区域、中央冷库及中央外部区域：

- 中央区域：相对当时主祖 `SubhaloPos`，$r<0.1R_{200c}$；
- 中央冷库：位于中央区域且满足第 5.2 节冷气体定义的气体；
- 中央热气体：位于中央区域且满足第 5.2 节热气体定义的气体；
- 外晕：位于中央区域之外，且相对 `GroupPos` 满足 $r<R_{200c}$；
- 晕外：相对 `GroupPos` 满足 $r\geq R_{200c}$；
- 卫星：绑定到非中心 Subfind 子晕，单独标记，不并入平滑外晕。

### 4.2 热/冷相态

沿用 Phase 1.8：

- 冷气体：`SFR > 0`，或 `SFR <= 0` 且 $\log_{10}(T/{\rm K})<4.5$；
- 热气体：`SFR <= 0` 且 $\log_{10}(T/{\rm K})\geq4.5$。

### 4.3 tracer 状态向量

每个 `TracerID` 在每个快照保存相互正交的状态字段：

```text
host_state    = main_central / satellite / other_halo / unbound / unresolved
radial_state  = inner / outer_halo / beyond_r200c / unresolved
phase_state   = cold / hot / unresolved
carrier_state = gas / star / wind / black_hole / unresolved
```

## 5. AGN 影响检验

### 5.1 归一化冷却半径

对 snap94 和 snap95 分别计算现有 Henriques 等温热晕模型输出：

$$
x_{\rm cool,s}=\frac{r_{\rm cool,s}}{R_{200c,s}},
\qquad s\in\{94,95\}.
$$

对两个端点的值进行算术平均作为分析对象。

### 5.2 AGN 强度定义

$$
\dot M_{\rm BH}=k_{\rm AGN}
\left(\frac{M_{\rm hot}}{10^{11}M_\odot}\right)
\left(\frac{M_{\rm BH}}{10^8M_\odot}\right),
$$

$$
\dot E_{\rm radio}=0.1\dot M_{\rm BH}c^2,
\qquad
\dot M_{\rm heat,H15}=\frac{2\dot E_{\rm radio}}{V_{200c}^2}.
$$

定义 AGN 相对强度

$$
\mathcal A_{\rm SAM}
=\log_{10}\left(\frac{\dot M_{\rm heat,H15}}{M_{\rm hot}/t_{\rm dyn}}\right),
$$

其中
$$
t_{\rm dyn} = \frac{R_{200c}}{V_{200c}}
$$

### 5.3 AGN 强度分类

在每个最终恒星质量箱内部，分别按选定 AGN 强度排序：

- Q1：最低 0--25%；
- Q2：25--50%；
- Q3：50--75%；
- Q4：最高 75--100%。

50 个对象不能严格四等分，因此使用人数差不超过 1 的稳定分组（12/13 个）。

## 6. snap94 → 95 中央冷库事件

### 6.1 冷库指示量

令 $C_s(j)=1$ 表示 tracer $j$ 在快照 $s$ 位于主祖中央冷气体库，
$C_s(j)=0$ 表示不在。

### 6.2 进入

对于 $C_{94}=0\to C_{95}=1$ 的进入事件，按 snap94 来源分为：

- 外晕冷：空间冷模供给；
- 外晕热：空间热模供给并冷却；
- 晕外：跨晕快速供给；
- 中央热：原位冷却；
- star/wind/BH/unresolved/卫星/其他星系 → 中央冷：统一记为其他。

#### 回溯

将进入事件的气体 tracer 回溯至 snap90：

首先筛查出前史中涉及恒星、BH、卫星星系或其他明确载体，或者由于状态缺失而无法可靠判断其中央冷库历史的 tracer，统一记为 `other`，并在后续互斥分类的母集中排除 `other`。`other` 仍保留在完整事件账本、总质量率和成分比例中。

- 对于 snap 94 在中心外的 tracer（外晕冷、外晕热、晕外）可划分为：
  1. `first-in`：在可追踪的 snap90--94 期间从未成为中央冷气体，并于 snap95 首次被观测为中央冷气体；
  2. `recycled-in-a`：在可追踪前史中至少一次成为中央冷气体，随后离开中央冷库，并于 snap95 再次成为中央冷气体；与 1 互斥；

- 对于 snap 94 在中心的 tracer（中央热）可划分为：
  1. `stay-in`：snap90--94 始终位于中央区域，没有观测到中央外 → 中央内的空间进入事件；
  2. `single-in`：snap90--94 期间只发生过一次中央外 → 中央内的转变；进入之前的相态无要求，进入之后至 snap94 需要始终为热气体。例如，snap90--92 位于中央外，snap93--94 为中央热气体；
  3. `recycled-in-b`：在排除 `other` 后，除 `stay-in` 和 `single-in` 之外的有效中央热事件；与 1、2 互斥。

### 6.3 离开

对于 $C_{94}=1\to C_{95}=0$ 的事件，按 snap95 去向分为：

- 外晕冷：冷空间流出；
- 外晕热：离开并被加热；
- 中央热：原位加热；
- 晕外：跨晕吹出；
- star/wind/BH/卫星/其他星系/unresolved：统一记为其他；

#### 追踪

首先筛查出后验历史中涉及恒星、BH、卫星星系或其他明确载体，或者由于状态缺失而无法可靠判断其中央冷库历史的 tracer，统一记为 `other`，并在后续互斥分类的母集中排除 `other`。`other` 仍保留在完整事件账本、总质量率和成分比例中。

将离开事件的气体 tracer 追踪至 snap99：
1. `stay-out`：在 snap96--99 期间始终未重新成为中央冷气体；
2. `recycled-out`：在排除 `other` 后，于 snap96--99 至少一次重新成为中央冷气体；与 1 互斥；

## 7. 速率计算

所有速率均以 snap94 → snap95 为窗口，统一使用
$$
\Delta t_{94,95}=t_{95}-t_{94}
$$
作为时间分母。`other` 从 first/recycled/stay/single 的有效分类母集中排除，
但保留在总事件率、成分比例和闭合质量中。

### 7.1 总事件率

进入全集定义为 $C_{94}=0\to C_{95}=1$，离开全集定义为
$C_{94}=1\to C_{95}=0$。这里的 `outer` 是中央冷气体库之外，不局限于某个空间相态。

$$
\dot M_{\rm total,in}
=
\frac{M(C_{94}=0\rightarrow C_{95}=1)}{\Delta t_{94,95}},
\qquad
\dot M_{\rm total,out}
=
\frac{M(C_{94}=1\rightarrow C_{95}=0)}{\Delta t_{94,95}}.
$$

### 7.2 进入分类与闭合

$$
\dot M_{\rm first,in}
= \frac{M(\mathrm{first{-}in})}{\Delta t_{94,95}},
\qquad
\dot M_{\rm recycled,in,a}
= \frac{M(\mathrm{recycled{-}in{-}a})}{\Delta t_{94,95}},
$$
$$
\dot M_{\rm stay,in}
= \frac{M(\mathrm{stay{-}in})}{\Delta t_{94,95}},
\qquad
\dot M_{\rm single,in}
= \frac{M(\mathrm{single{-}in})}{\Delta t_{94,95}},
$$
$$
\dot M_{\rm recycled,in,b}
= \frac{M(\mathrm{recycled{-}in{-}b})}{\Delta t_{94,95}},
\qquad
\dot M_{\rm other,in}
= \frac{M(\mathrm{other\ entering\ events})}{\Delta t_{94,95}}.
$$

进入事件严格闭合为：

$$
\dot M_{\rm total,in}
= \dot M_{\rm first,in}
+ \dot M_{\rm recycled,in,a}
+ \dot M_{\rm stay,in}
+ \dot M_{\rm single,in}
+ \dot M_{\rm recycled,in,b}
+ \dot M_{\rm other,in}.
$$

### 7.3 离开分类与闭合

$$
\dot M_{\rm stay,out}
= \frac{M(\mathrm{stay{-}out})}{\Delta t_{94,95}},
\qquad
\dot M_{\rm recycled,out}
= \frac{M(\mathrm{recycled{-}out})}{\Delta t_{94,95}},
$$
$$
\dot M_{\rm other,out}
= \frac{M(\mathrm{other\ leaving\ events})}{\Delta t_{94,95}}.
$$

离开事件严格闭合为：

$$
\dot M_{\rm total,out}
= \dot M_{\rm stay,out}
+ \dot M_{\rm recycled,out}
+ \dot M_{\rm other,out}.
$$

为保留已有变量名称，定义中央冷库总反馈作用率为：

$$
\dot M_{\rm feedback}=\dot M_{\rm total,out}.
$$

该量是中央冷库总移除率；`other` 中可能包含恒星形成、风、BH或卫星通道，不能全部解释为AGN反馈。

### 7.4 组合定义

$$
\dot M_{\rm first,in}^{\rm eff}
= \dot M_{\rm first,in} + \dot M_{\rm single,in},
$$
$$
\dot M_{\rm recycled,in}^{\rm eff}
= \dot M_{\rm recycled,in,a} + \dot M_{\rm recycled,in,b},
$$
$$
\dot M_{\rm supply}=\dot M_{\rm total,in},
\qquad
\dot M_{\rm pf,in}
= \dot M_{\rm first,in}^{\rm eff} + \dot M_{\rm stay,out}.
$$

每个质量箱的供给100%组成使用
$(\dot M_{\rm first,in}^{\rm eff},\dot M_{\rm recycled,in}^{\rm eff},
\dot M_{\rm stay,in},\dot M_{\rm other,in})$；离开100%组成使用
$(\dot M_{\rm stay,out},\dot M_{\rm recycled,out},\dot M_{\rm other,out})$。

## 8. 计算契约

- tracer 扫描沿用 1.8 的快照内分块策略：每个 HDF5 chunk 是一个独立任务，默认使用 64 个进程；snap94/95 的 `ParentID→TracerID`、snap90--99 的 `TracerID→ParentID` 和 parent 粒子记录查询均采用同一调度；
- 每个 tracer/particle chunk 单独原子缓存，任务中断后只重算尚未完成的 chunk；完整快照结果另存为合并缓存；
- groupcat 的 `GroupFirstSub`、`GroupNsubs`、`SubhaloLenType` 及 MPB 所需的子晕字段按快照一次读取，并缓存为 `data/interim/cache/catalogues/*.npz`；
- 默认使用 1 个 worker 构建逐晕状态，避免多个大 FoF 同时驻留；tracer 阶段仍独立使用 64 核；
- 星体状态构建只读取实际需要的 `ParticleIDs`；
- 样本、90--99 MPB、逐快照状态、tracer-parent 映射和事件账本分别缓存；
- fingerprint 包含模拟名、snap范围、事件锚点、质量箱、随机种子、状态定义、`other`规则、分类规则、tracer权重和schema版本；
- 合法零结果写入缓存，读取失败不能写成零；所有缓存采用原子写入。

## 9. 可视化

1. 主图：按 $\mathcal A_{\rm H15}$ 四分位的四条 $r_{\rm cool}/R_{200c}$ 中位线，副图：围绕中位数的 1 $\sigma$ 离散度；
2. 中央冷库总供给来源：$\dot M_{\rm first,in}^{\rm eff}$、$\dot M_{\rm recycled,in}^{\rm eff}$、$\dot M_{\rm stay,in}$ 和 `other` 的 100% 堆叠柱形图；
3. 中央冷库总反馈作用去向：$\dot M_{\rm stay,out}$、$\dot M_{\rm recycled,out}$ 和 `other` 的 100% 堆叠柱形图；
4. 主图：反馈前吸积率估计 $\dot M_{\rm pf,in}$ 与 $\dot M_{\rm cool,iso}$ 的中位线；副图展示围绕中位数的 1 $\sigma$ 离散度，需要采用两种归一化方式：$M_{\rm hot}$ 和 $M_{\rm stellar}$。

数值产品包括样本/MPB JSON、逐晕 NPZ/CSV、逐快照事件 HDF5、分箱和AGN四分位
NPZ，以及包含定义、有效数、`other`比例和闭合残差的元数据 JSON。事件 HDF5
同时保存 `TracerID`、事件方向、锚点/源/目标状态、完整状态序列和缺失掩码。

命令行入口为 `scripts/run_analysis.py`；可通过 `--base-path`、
`--cooling-table-dir`、`--cache-dir`、`--output-dir`、`--figure-dir`、
`--state-workers`、`--state-backend`、`--rebuild-sample` 和 `--quiet` 覆盖运行路径、
并行策略与输出行为。state 构建默认保持单 worker；tracer 扫描会在每个快照
内部将全部 HDF5 chunks 分派到最多 64 个进程。
