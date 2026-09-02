# Transformer 控制倒立摆 · 项目说明

> 用 Transformer 神经网络作为倒立摆（Cart-Pole）的控制器，通过模仿学习训练，并配套交互式可视化网页与完整技术文档。

---

## 1. 项目概述

本项目将倒立摆稳定控制问题形式化为**序列到动作的映射**：以最近 8 步状态 `[x, ẋ, θ, θ̇]` 为输入，用 Transformer 输出当前控制力。训练采用**模仿学习**（Behavior Cloning），以 LQR 为专家教师。

**完整链路**：物理建模 → LQR 专家 → 数据收集 → 监督训练 → 权重导出 → 网页部署 → 文档沉淀。

**技术栈**：
- 训练：Python 3.9+ + **纯 NumPy**（手写 Transformer 前向/反向传播，无深度学习框架）
- 网页：纯 HTML/CSS/JS（Canvas 动画，单文件自包含）
- 文档：LaTeX（ctex + TikZ）+ reportlab（PDF）

---

## 2. 文件清单

### 2.1 核心交付物

| 文件 | 大小 | 作用 |
|------|------|------|
| `transformer_pendulum.html` | 123.5 KB | **最终交互式网页**（内嵌权重，双击即用） |
| `transformer_cartpole.tex` | 53.1 KB | **LaTeX 技术文档**（理论+公式+TikZ图+训练指南） |
| `transformer_cartpole.pdf` | 689.8 KB | **PDF 技术文档**（可直接阅读/打印） |
| `train_transformer_controller.py` | 16.3 KB | **训练脚本**（单文件自包含，六阶段流程） |

### 2.2 训练产物

| 文件 | 大小 | 作用 |
|------|------|------|
| `transformer_weights.json` | 190.9 KB | 完整模型权重（float64，含位置编码） |
| `weights_compact.json` | 84.7 KB | 紧凑权重（float32，去 PE，内嵌进网页） |

### 2.3 生成脚本

| 文件 | 大小 | 作用 |
|------|------|------|
| `generate_pdf.py` | 37.4 KB | PDF 生成脚本（reportlab + matplotlib mathtext） |
| `transformer_pendulum_template.html` | 38.6 KB | 网页模板（含 `__WEIGHTS_JSON__` 占位符） |

### 2.4 前序任务产物（背景相关）

| 文件 | 作用 |
|------|------|
| `inverted_pendulum.html` | LQR 控制倒立摆网页（本项目的起点，对比基准） |
| `Transformer实现指南_MATLAB与Python.md` | 更早的 Transformer 实现指南（MATLAB + Python） |

---

## 3. 快速开始

### 3.1 查看交互式网页（无需任何依赖）

直接双击打开 `transformer_pendulum.html`。网页包含：

- **倒立摆实时动画**（Canvas，含力箭头、角度标注）
- **多头注意力热力图**（4 头 × 8×8，红框=当前决策关注）
- **注意力诊断面板**（自动统计集中度/跨度/最近3步占比/头相似度 + 调参建议）
- **输入敏感度分析**（扰动法因果归因，与注意力对照展示"注意力≠因果"）
- **实时计算链**（输入序列 → 前向 → 控制力）
- **原理 / 伪代码 / 流程图** 三栏文档
- 可切换 **Transformer / LQR** 控制器对比，调初始角度/扰动/速度

### 3.2 重新训练模型

```bash
pip install numpy          # 唯一依赖
python train_transformer_controller.py
```

脚本按六阶段顺序自动执行：

1. **LQR 增益求解**（离散 Riccati 迭代，打印 K）
2. **梯度检查**（数值 vs 解析梯度，验证手写反向传播）
3. **数据收集**（LQR 专家 + 扰动，约 5.8 万样本）
4. **监督训练**（Adam + MSE，60 epoch）
5. **闭环测试**（θ₀=0.1/0.2/0.3 各 800 步）
6. **导出权重**（生成 `transformer_weights.json`）

### 3.3 编译文档

- **PDF 版**：直接打开 `transformer_cartpole.pdf`
- **LaTeX 版**：将 `transformer_cartpole.tex` 上传到 [Overleaf](https://www.overleaf.com)，**编译器选 XeLaTeX**（文档用了 ctex 中文宏包 + TikZ 图）

---

## 4. 文件依赖关系

```
train_transformer_controller.py ──► transformer_weights.json
                                          │
                              （压缩 float32、去 PE）
                                          ▼
                                   weights_compact.json
                                          │
                          （注入模板占位符 __WEIGHTS_JSON__）
                                          ▼
transformer_pendulum_template.html ──► transformer_pendulum.html（最终网页）

transformer_cartpole.tex ──（手写 LaTeX）──► 可编译为 PDF
generate_pdf.py ──（reportlab）──► transformer_cartpole.pdf
```

**说明**：`transformer_pendulum.html` 已内嵌权重，独立可用；`transformer_cartpole.tex` 与 `transformer_cartpole.pdf` 内容对应（前者公式排版更专业，后者开箱即用）。

---

## 5. 核心成果

### 5.1 理论

| 章节 | 内容 |
|------|------|
| 倒立摆建模 | 质点杆模型 → 拉格朗日方程 → 非线性动力学 → 泰勒展开线性化 → A/B 矩阵 → 可控性分析（det C = g²/M⁴l⁴） |
| LQR | 庞特里亚金极小值原理 → CARE 方程 → 离散 Riccati 迭代 |
| Transformer | 缩放点积注意力（√d_k 方差分析）→ 多头 → 位置编码 → FFN + 残差 + LayerNorm |
| 编码 | 纯 NumPy 手写前向 + 反向传播（含 LayerNorm/注意力梯度），数值梯度检查验证 |

### 5.2 实验数据

| 指标 | 结果 |
|------|------|
| LQR 增益 K | `[-2.954, -4.735, -46.339, -9.163]` |
| 闭环稳定性 | θ₀=0.1/0.2/0.3 rad 均稳定（峰值 5.7°/11.4°/17.1°） |
| 扰动恢复 | 1.2 rad/s 脉冲，约 3 s 恢复（峰值 5.9°） |
| 训练 | 5.8 万样本 / 60 epoch / loss 0.92 → 10⁻³ |
| 注意力诊断 | 最近 3 步占比 69%，头相似度 54%（健康） |
| 敏感度（当前步） | `[1.98, 3.57, 31.19, 6.87]`（θ 最大，结构同 LQR） |

### 5.3 关键结论

1. **Transformer 学到了 LQR 策略**——当前步敏感度结构与 LQR 增益一致（θ 权重最大）。
2. **Transformer 区别于 LQR**——历史步敏感度非零（利用历史），LQR 历史步贡献严格为零。
3. **注意力 ≠ 因果**——注意力分布相对分散，但敏感度分析显示真实贡献集中在当前步的 θ、θ̇，网页通过两图并排直观展示这一边界。

---

## 6. 网页架构（`transformer_pendulum.html`）

单文件自包含，内部结构：

- **物理层**：非线性动力学 + 四阶 Runge-Kutta 积分
- **控制层**：Transformer 前向推理（嵌入 → 位置编码 → 多头注意力 → FFN → 输出头）；LQR 作对比
- **可视化层**：
  - Canvas 动画（倒立摆 + 力箭头）
  - 注意力热力图（相关性）
  - 注意力诊断（统计 + 调参建议）
  - 敏感度分析（因果归因，扰动法）
- **文档层**：原理 / 伪代码 / 流程图（Tab 切换）

---

## 7. 复现与调参提示

- **环境**：Python 3.9+、NumPy；网页无依赖；LaTeX 需 XeLaTeX 编译器。
- **超参数**：集中在训练脚本顶部（物理参数 G/M/m/L、LQR 权重 Q/R、网络结构 D_MODEL/NUM_HEADS/D_FF/T、训练 epochs/batch/lr）。
- **注意力调参**：网页诊断面板会根据注意力模式自动建议（如"最近 3 步占比 > 70% → T 可减小到 4~5"），但注意力是**诊断信号**而非优化目标，最终以闭环稳定性为准。
- **已知边界**：模仿学习的性能受限于专家（LQR 是线性的），误差累积可用 DAgger 修正；注意力可解释性有限，严格归因需梯度方法。

---

## 8. 常见问题

| 问题 | 解决 |
|------|------|
| 网页无法加载权重 | `transformer_pendulum.html` 已内嵌权重，直接双击即可；若用模板需先注入 `weights_compact.json` |
| LaTeX 编译报 `red!70!80` 颜色错误 | 已修复为 `red!70!black`，重新下载最新 `.tex` |
| LaTeX 报 `Missing $ inserted` | 已修复（函数名反引号 → `\texttt{}`），重新下载最新 `.tex` |
| hyperref 警告（PDF 书签） | 已用 `\texorpdfstring` 修复，非致命 |
| LaTeX 编译失败 | 确认编译器为 **XeLaTeX**（ctex 需要），非 pdfLaTeX |
