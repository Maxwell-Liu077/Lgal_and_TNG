## H15：Supernova Feedback

SN feedback 主要分为：

1. **Reheating**：将冷气体重新加热并转移到 hot gas reservoir。
2. **Ejection**：将 halo 的热气体喷出到 ejected gas reservoir。

### 1.1 参数表

| 参数 | 含义 | H15 best fit | 单位 |
|---|---|---:|---|
| $\epsilon$ | Mass-loading efficiency normalization | 2.6 | 无量纲 |
| $V_{\mathrm{reheat}}$ | Reheating characteristic velocity | 480 | km/s |
| $\beta_1$ | Reheating velocity slope | 0.72 | 无量纲 |
| $\eta_{\mathrm{SN}}$ | SN energy coupling / ejection efficiency normalization | 0.62 | 无量纲 |
| $V_{\mathrm{eject}}$ | Ejection characteristic velocity | 100 | km/s |
| $\beta_2$ | Ejection velocity slope | 0.80 | 无量纲 |
| $V_{\mathrm{SN}}$ | SN energy normalization | 630 | km/s |
| $\gamma$ | Reincorporation timescale normalization | $3.0\times10^{10}$ | yr |

### 1.2 SN Reheating Rate

Reheating mass-loading efficiency（Eq. S19）：

$$
\boxed{\epsilon_{\mathrm{disk}}=\epsilon\left[0.5+\left(\frac{V_{\max}}{V_{\mathrm{reheat}}}\right)^{-\beta_1}\right]}
$$

未经能量上限修正的 reheating rate（Eq. S18）：

$$
\dot M_{\mathrm{reheat,nominal}}=\epsilon_{\mathrm{disk}}\,\mathrm{SFR}
$$

考虑 SN 能量预算后：

$$
\boxed{
\dot M_{\mathrm{reheat}}=
\min\left[\epsilon_{\mathrm{disk}},\epsilon_{\mathrm{halo}}\frac{V_{\mathrm{SN}}^2}{V_{200c}^2}\right]\mathrm{SFR}
}
$$

其中 $\epsilon_{\mathrm{halo}}$ 由下一节计算。

### 1.3 SN Ejection Rate

Halo feedback efficiency（Eq. S17）：

$$
\boxed{\epsilon_{\mathrm{halo}}=\eta_{\mathrm{SN}}\left[0.5+\left(\frac{V_{\max}}{V_{\mathrm{eject}}}\right)^{-\beta_2}\right]}
$$

SN 可用于反馈的能量注入率（由 Eq. S16 写成速率形式）：

$$
\boxed{\dot E_{\mathrm{SN}}=\frac12\epsilon_{\mathrm{halo}}\,\mathrm{SFR}\,V_{\mathrm{SN}}^2}
$$

Reheating 消耗能量的速率：

$$
\dot E_{\mathrm{reheat}}=\frac12\dot M_{\mathrm{reheat}}V_{200c}^2
$$

由剩余能量计算 ejection rate（Eq. S20）：

$$
\boxed{
\dot M_{\mathrm{eject}}=
\max\left[\epsilon_{\mathrm{halo}}\frac{V_{\mathrm{SN}}^2}{V_{200c}^2}-\epsilon_{\mathrm{disk}},\ 0\right]\mathrm{SFR}
}
$$

当 SN 能量不足以支持 nominal reheating 时，实际 reheating 被限制为能量预算允许的最大值，且 $\dot M_{\mathrm{eject}}=0$。

**所需物理量：**

| 物理量 | 含义 | 单位 |
|---|---|---|
| $\mathrm{SFR}$ | Star formation rate | $M_\odot/\mathrm{yr}$ |
| $V_{\max}$ | Halo maximum circular velocity | km/s |
| $V_{200c}$ | Halo virial velocity | km/s |

---

## 二、H15：AGN Feedback

### 2.1 参数表

| 参数 | 含义 | H15 best fit | 单位 |
|---|---|---:|---|
| $k_{\mathrm{AGN}}$ | Radio-mode accretion normalization | $5.3\times10^{-3}$ | $M_\odot/\mathrm{yr}$ |
| $\eta_{\mathrm{radio}}$ | Radio-mode heating efficiency | 0.1 | 无量纲 |
| $f_{\mathrm{BH}}$ | Quasar-mode BH growth efficiency | 0.041 | 无量纲 |
| $V_{\mathrm{BH}}$ | Quasar-mode characteristic velocity | 750 | km/s |

### 2.2 Radio-mode BH Accretion Rate（Eq. S24）

$$
\boxed{
\dot M_{\mathrm{BH}}=k_{\mathrm{AGN}}
\left(\frac{M_{\mathrm{hot}}}{10^{11}M_\odot}\right)
\left(\frac{M_{\mathrm{BH}}}{10^{8}M_\odot}\right)
}
$$

代入 H15 参数：

$$
\dot M_{\mathrm{BH}}=5.3\times10^{-3}
\left(\frac{M_{\mathrm{hot}}}{10^{11}M_\odot}\right)
\left(\frac{M_{\mathrm{BH}}}{10^{8}M_\odot}\right)
M_\odot/\mathrm{yr}
$$

### 2.3 AGN Heating Rate（Eq. S25）

$$
\boxed{\dot E_{\mathrm{AGN}}=\eta_{\mathrm{radio}}\dot M_{\mathrm{BH}}c^2}
$$

其中 $\eta_{\mathrm{radio}}=0.1$，$c$ 为光速。

### 2.4 AGN Cooling Suppression Rate（Eq. S26）

$$
\boxed{
\dot M_{\mathrm{cool,eff}}=
\max\left[\dot M_{\mathrm{cool}}-\frac{2\dot E_{\mathrm{AGN}}}{V_{200c}^2},\ 0\right]
}
$$

根据该式推导出的被抑制冷却速率为：

$$
\boxed{
\dot M_{\mathrm{cool,suppressed}}=
\min\left[\dot M_{\mathrm{cool}},\frac{2\dot E_{\mathrm{AGN}}}{V_{200c}^2}\right]
}
$$

**所需物理量：**

| 物理量 | 含义 | 单位 |
|---|---|---|
| $M_{\mathrm{hot}}$ | Hot gas reservoir mass | $M_\odot$ |
| $M_{\mathrm{BH}}$ | Central black hole mass | $M_\odot$ |
| $V_{200c}$ | Halo virial velocity | km/s |
| $\dot M_{\mathrm{cool}}$ | AGN 抑制之前的 cooling rate | $M_\odot/\mathrm{yr}$ |

---

## 三、H20：AGN Feedback

### 3.1 参数表

| 参数 | H15 | H20 best fit | 单位 |
|---|---:|---:|---|
| $k_{\mathrm{AGN}}$ | $5.3\times10^{-3}$ | **$2.5\times10^{-3}$** | $M_\odot/\mathrm{yr}$ |
| $\eta_{\mathrm{radio}}$ | 0.1 | **0.1** | 无量纲 |
| $f_{\mathrm{BH}}$ | 0.041 | **0.066** | 无量纲 |
| $V_{\mathrm{BH}}$ | 750 | **700** | km/s |

### 3.2 Radio-mode BH Accretion Rate（Eq. S28）

$$
\boxed{
\dot M_{\mathrm{BH}}=
2.5\times10^{-3}
\left(\frac{M_{\mathrm{hot}}}{10^{11}M_\odot}\right)
\left(\frac{M_{\mathrm{BH}}}{10^8M_\odot}\right)
M_\odot/\mathrm{yr}
}
$$

### 3.3 AGN Heating Rate（Eq. S29）

$$
\boxed{\dot E_{\mathrm{AGN}}=0.1\dot M_{\mathrm{BH}}c^2}
$$

### 3.4 AGN Cooling Suppression Rate（Eq. S30）

$$
\boxed{
\dot M_{\mathrm{cool,eff}}=
\max\left[\dot M_{\mathrm{cool}}-\frac{2\dot E_{\mathrm{AGN}}}{V_{200c}^2},\ 0\right]
}
$$

推导出的被抑制冷却速率：

$$
\boxed{
\dot M_{\mathrm{cool,suppressed}}=
\min\left[\dot M_{\mathrm{cool}},\frac{2\dot E_{\mathrm{AGN}}}{V_{200c}^2}\right]
}
$$

所需模拟物理量与 H15 AGN 相同：$M_{\mathrm{hot}}$、$M_{\mathrm{BH}}$、$V_{200c}$、$\dot M_{\mathrm{cool}}$。

---