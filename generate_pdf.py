# -*- coding: utf-8 -*-
"""用 reportlab + matplotlib mathtext 生成《基于 Transformer 的倒立摆控制》详细版 PDF。"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from io import BytesIO
from PIL import Image as PILImage

from reportlab.lib.pagesizes import A4
from reportlab.lib.units import cm
from reportlab.lib import colors
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer, Image,
                                Table, TableStyle, Preformatted)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY

# ---------------- 字体 ----------------
FONT_DIR = 'C:/Windows/Fonts/'
pdfmetrics.registerFont(TTFont('YaHei', FONT_DIR + 'msyh.ttc'))
pdfmetrics.registerFont(TTFont('YaHeiBold', FONT_DIR + 'msyhbd.ttc'))
pdfmetrics.registerFont(TTFont('Mono', FONT_DIR + 'consola.ttf'))

ACCENT = colors.HexColor('#2D5F8A')
styles = getSampleStyleSheet()

S = {}
S['title'] = ParagraphStyle('t', fontName='YaHeiBold', fontSize=22, leading=30,
                            alignment=TA_CENTER, textColor=colors.HexColor('#1A1A1A'))
S['subtitle'] = ParagraphStyle('st', fontName='YaHei', fontSize=12, leading=18,
                               alignment=TA_CENTER, textColor=colors.HexColor('#555555'))
S['h1'] = ParagraphStyle('h1', fontName='YaHeiBold', fontSize=15.5, leading=21,
                         textColor=ACCENT, spaceBefore=16, spaceAfter=7)
S['h2'] = ParagraphStyle('h2', fontName='YaHeiBold', fontSize=12.5, leading=17,
                         textColor=colors.HexColor('#1A1A1A'), spaceBefore=10, spaceAfter=5)
S['body'] = ParagraphStyle('b', fontName='YaHei', fontSize=10, leading=16,
                           alignment=TA_JUSTIFY, spaceAfter=5)
S['abstract'] = ParagraphStyle('ab', fontName='YaHei', fontSize=9.5, leading=15,
                               alignment=TA_JUSTIFY, textColor=colors.HexColor('#333333'),
                               leftIndent=10, rightIndent=10, spaceAfter=8)
S['code'] = ParagraphStyle('c', fontName='Mono', fontSize=8, leading=10.8,
                           textColor=colors.HexColor('#1a1a1a'), backColor=colors.HexColor('#f4f6f8'),
                           borderPadding=5, leftIndent=3, rightIndent=3, spaceAfter=6)
S['ref'] = ParagraphStyle('r', fontName='YaHei', fontSize=9, leading=13, spaceAfter=3)
S['caption'] = ParagraphStyle('cap', fontName='YaHei', fontSize=8.5, leading=12,
                              textColor=colors.HexColor('#666666'), alignment=TA_CENTER, spaceAfter=8)

def render_math(expr, fontsize=12.5, dpi=200):
    fig = plt.figure(dpi=dpi)
    fig.text(0.5, 0.5, f'${expr}$', fontsize=fontsize, ha='center', va='center')
    buf = BytesIO()
    fig.savefig(buf, format='png', dpi=dpi, bbox_inches='tight', pad_inches=0.03, transparent=True)
    plt.close(fig)
    buf.seek(0)
    img = PILImage.open(buf)
    w, h = img.size
    buf.seek(0)
    return Image(buf, width=w * 72 / dpi, height=h * 72 / dpi)

def M(expr, fs=12.5):
    return [Spacer(1, 3), render_math(expr, fs), Spacer(1, 3)]

def matrix_table(rows, col_w=1.3 * cm, fs=9):
    data = [[Paragraph(c, ParagraphStyle('m', fontName='YaHei', fontSize=fs,
              alignment=TA_CENTER)) for c in row] for row in rows]
    t = Table(data, colWidths=[col_w] * len(rows[0]), rowHeights=[0.5 * cm] * len(rows))
    t.setStyle(TableStyle([
        ('GRID', (0, 0), (-1, -1), 0.4, colors.HexColor('#cccccc')),
        ('LINEBEFORE', (0, 0), (0, -1), 1.4, colors.HexColor('#1a1a1a')),
        ('LINEAFTER', (-1, 0), (-1, -1), 1.4, colors.HexColor('#1a1a1a')),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
    ]))
    return t

def body(txt):
    return Paragraph(txt, S['body'])

# ---------------- matplotlib 画图 ----------------
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle, Rectangle

def _fig_to_image(fig, dpi=160):
    buf = BytesIO()
    fig.savefig(buf, format='png', dpi=dpi, bbox_inches='tight', pad_inches=0.08, transparent=False)
    plt.close(fig)
    buf.seek(0)
    img = PILImage.open(buf)
    w, h = img.size
    buf.seek(0)
    return Image(buf, width=w * 72 / dpi, height=h * 72 / dpi)

def draw_cartpole():
    fig, ax = plt.subplots(figsize=(7.2, 3.4))
    ax.set_xlim(-4.3, 4.3); ax.set_ylim(-1.2, 3.6)
    ax.set_aspect('equal'); ax.axis('off')
    # 地面
    ax.plot([-4.2, 4.2], [0, 0], color='#333', lw=2)
    for i in range(-4, 5):
        ax.plot([i, i], [0, -0.08], color='#999', lw=1)
    # 小车
    ax.add_patch(Rectangle((-0.9, 0), 1.8, 0.7, fc='#cfe4f7', ec='#2D5F8A', lw=2))
    ax.add_patch(Circle((-0.6, -0.12), 0.16, fc='#333'))
    ax.add_patch(Circle((0.6, -0.12), 0.16, fc='#333'))
    # 竖直参考
    ax.plot([0, 0], [0.7, 3.1], '--', color='#bbb', lw=1)
    # 杆
    ax.plot([0, 1.15], [0.7, 3.0], color='#c0392b', lw=3)
    ax.plot([0], [0.7], 'o', color='#333', ms=5)
    ax.plot([1.15], [3.0], 'o', color='#c0392b', ms=7)
    # 角度弧
    ax.annotate('', xy=(0.44, 1.38), xytext=(0, 1.5),
                arrowprops=dict(arrowstyle='->', color='#333', lw=1.2))
    ax.text(0.35, 1.62, r'$\theta$', fontsize=13)
    # 力
    ax.annotate('', xy=(2.2, 0.35), xytext=(0.95, 0.35),
                arrowprops=dict(arrowstyle='->', color='#e67e22', lw=2.5))
    ax.text(1.55, 0.55, '$F$', fontsize=13, color='#e67e22')
    # 坐标
    ax.annotate('', xy=(1.6, -0.7), xytext=(0, -0.7),
                arrowprops=dict(arrowstyle='->', color='#333', lw=1.2))
    ax.text(1.7, -0.9, '$x$', fontsize=13)
    # 标注
    ax.text(0, 0.32, '$M$', fontsize=13, color='#2D5F8A', ha='center')
    ax.text(1.15, 3.15, '$m$', fontsize=13, color='#c0392b')
    ax.text(0.7, 1.75, '$l$', fontsize=13)
    ax.annotate('', xy=(1.15, 2.35), xytext=(1.15, 2.95),
                arrowprops=dict(arrowstyle='->', color='#333', lw=1.2))
    ax.text(1.35, 2.6, '$mg$', fontsize=13)
    return _fig_to_image(fig)

def draw_loop():
    fig, ax = plt.subplots(figsize=(9.5, 2.6))
    ax.set_xlim(0, 10); ax.set_ylim(0, 3.4); ax.axis('off')
    def box(x, y, text, color='#2D5F8A'):
        ax.add_patch(FancyBboxPatch((x, y), 1.8, 1.0, boxstyle='round,pad=0.06',
                     fc='#eef4fa', ec=color, lw=1.8))
        ax.text(x+0.9, y+0.5, text, ha='center', va='center', fontsize=9)
    box(0.2, 1.4, 'state sequence\n$X_t\\in\\mathbb{R}^{T\\times4}$')
    box(2.2, 1.4, 'Transformer\ncontroller $\\pi_\\phi$')
    box(4.2, 1.4, 'control\nforce $u_t$')
    box(6.2, 1.4, 'cart-pole\nnonlinear dynamics')
    box(8.2, 1.4, 'new state\n$z_{t+1}$')
    for x in [2.0, 4.0, 6.0, 8.0]:
        ax.annotate('', xy=(x+0.18, 1.9), xytext=(x, 1.9),
                    arrowprops=dict(arrowstyle='->', color='#2D5F8A', lw=1.8))
    # 反馈
    ax.plot([9.1, 9.1, 1.1, 1.1], [1.4, 0.6, 0.6, 1.36], color='#c0392b', lw=1.8)
    ax.annotate('', xy=(1.1, 1.38), xytext=(1.1, 0.6),
                arrowprops=dict(arrowstyle='->', color='#c0392b', lw=1.8))
    ax.text(5.1, 0.35, 'update sliding window (keep last T steps)', ha='center', fontsize=9, color='#c0392b')
    return _fig_to_image(fig)

def draw_transformer():
    fig, ax = plt.subplots(figsize=(6.4, 5.6))
    ax.set_xlim(0, 7); ax.set_ylim(0, 8.6); ax.axis('off')
    steps = [
        ('input seq $X\\in\\mathbb{R}^{T\\times4}$', '#2D5F8A', 'plain'),
        ('linear embedding $XW_e+b_e$ + positional encoding PE', '#666', 'fill'),
        ('multi-head self-attention (h=4 heads)', '#2D5F8A', 'plain'),
        ('residual + LayerNorm', '#666', 'fill'),
        ('FFN: ReLU$(xW_1+b_1)W_2+b_2$', '#2D5F8A', 'plain'),
        ('residual + LayerNorm', '#666', 'fill'),
        ('take last position $H_2[T-1]$ + linear head', '#c0392b', 'plain'),
        ('control force $u_t\\in\\mathbb{R}$', '#c0392b', 'plain'),
    ]
    y = 8.0
    ys = []
    for text, color, kind in steps:
        fc = '#eef4fa' if color == '#2D5F8A' else ('#f6f6f6' if kind == 'fill' else '#fbeaea')
        ax.add_patch(FancyBboxPatch((0.8, y-0.38), 4.8, 0.72, boxstyle='round,pad=0.05',
                     fc=fc, ec=color, lw=1.5))
        ax.text(3.2, y, text, ha='center', va='center', fontsize=8.5)
        ys.append(y)
        y -= 1.02
    # 箭头
    for i in range(len(ys)-1):
        ax.annotate('', xy=(3.2, ys[i+1]+0.42), xytext=(3.2, ys[i]-0.4),
                    arrowprops=dict(arrowstyle='->', color='#333', lw=1.3))
    # 残差虚线
    ax.annotate('', xy=(5.75, ys[3]+0.3), xytext=(5.75, ys[1]-0.3),
                arrowprops=dict(arrowstyle='->', color='#2D5F8A', lw=1.2, linestyle='--'))
    ax.annotate('', xy=(5.75, ys[5]+0.3), xytext=(5.75, ys[3]-0.3),
                arrowprops=dict(arrowstyle='->', color='#2D5F8A', lw=1.2, linestyle='--'))
    ax.text(5.95, (ys[1]+ys[3])/2, 'residual', fontsize=8, color='#2D5F8A', rotation=90, va='center')
    # 维度标注
    dims = ['', '$(T,32)$', '$(T,32)$', '', '$(T,32)\\to(T,64)\\to(T,32)$', '', '$(32)\\to(1)$', '']
    for i, d in enumerate(dims):
        if d:
            ax.text(0.6, ys[i], d, fontsize=8, color='#888', va='center', ha='right')
    return _fig_to_image(fig)

doc = SimpleDocTemplate('transformer_cartpole.pdf', pagesize=A4,
                        leftMargin=2.1 * cm, rightMargin=2.1 * cm,
                        topMargin=2.0 * cm, bottomMargin=2.0 * cm,
                        title='基于 Transformer 的倒立摆控制')

story = []
story.append(Spacer(1, 26))
story.append(Paragraph('基于 Transformer 的倒立摆控制', S['title']))
story.append(Spacer(1, 6))
story.append(Paragraph('系统建模 · 最优控制 · 深度学习控制 · 完整理论与编码实现', S['subtitle']))
story.append(Spacer(1, 16))

# 摘要
story.append(Paragraph('摘　要', S['h1']))
story.append(Paragraph(
    '本文完整阐述用 <b>Transformer</b> 神经网络控制倒立摆（Cart–Pole）的理论与实现，覆盖四个层面：'
    '<b>(i) 系统建模</b>——从质点杆模型出发，经拉格朗日方程严格推导非线性动力学，再由一阶泰勒展开得到状态空间线性化，'
    '并分析开环稳定性与可控性；<b>(ii) 最优控制</b>——由庞特里亚金极小值原理导出 LQR 的 Riccati 方程，给出离散迭代求解；'
    '<b>(iii) 深度学习</b>——讲解 Transformer 的缩放点积注意力、多头注意力、位置编码、前馈与残差连接及其数学依据；'
    '<b>(iv) 编码实现</b>——以纯 NumPy 手写物理仿真、LQR 求解、Transformer 前向/反向传播与训练循环，并逐行说明。'
    '实验表明，模仿学习训练的 Transformer 控制器能稳定维持平衡，输出与 LQR 专家高度一致。', S['abstract']))
story.append(Spacer(1, 4))

# ============ 1 引言 ============
story.append(Paragraph('1　引言', S['h1']))
story.append(body('倒立摆系统由可水平运动的小车与铰接其上的摆杆组成，控制目标为施加水平力 F，使摆杆维持在竖直向上的'
    '<b>不稳定平衡点</b>附近，同时小车回归原点。系统集<b>欠驱动</b>（一个输入、两个自由度）、<b>非线性</b>、'
    '<b>开环不稳定</b>于一身，是控制理论的经典验证平台。'))
story.append(body('控制方法分两类：<b>基于模型</b>（PID、LQR、MPC）需精确建模、可解释但依赖线性化；'
    '<b>数据驱动</b>（强化学习、模仿学习）直接从数据学习策略，无需精确模型但可解释性较弱。'))
story.append(body('Transformer 凭借<b>多头自注意力</b>在序列建模上取得巨大成功，其"对序列任意位置长程依赖直接建模"的能力'
    '与控制问题高度契合：控制器本质是"观测历史到当前动作"的序列决策器。本文将倒立摆控制形式化为序列决策任务，'
    '用 Transformer 作为策略网络，经模仿学习训练。'))

# ============ 2 建模 ============
story.append(Paragraph('2　倒立摆系统建模', S['h1']))
story.append(Paragraph('2.1　系统描述、坐标系与假设', S['h2']))
story.append(body('广义坐标取小车位移 x（向右为正）与摆杆偏角 θ（相对竖直向上，顺时针为正）。'
    '<b>关键假设（质点杆模型）</b>：摆杆质量 m 集中于距铰点 l 处，忽略杆转动惯量。该假设使非线性动力学与线性化'
    '<b>严格自洽</b>——可精确退化为教科书标准形式。参数：M=1.0 kg，m=0.1 kg，l=0.5 m，g=9.81 m/s²。'))
story.append(draw_cartpole())
story.append(Paragraph('图 1　倒立摆系统示意图：小车质量 M、摆杆质量 m（质点）、半长 l、偏角 θ、水平控制力 F。', S['caption']))

story.append(Paragraph('2.2　运动学：质心位置与速度', S['h2']))
story.append(body('杆质心坐标 (x + l·sinθ, l·cosθ)，速度为 (ẋ + l·θ̇·cosθ, −l·θ̇·sinθ)。'))

story.append(Paragraph('2.3　拉格朗日量：动能与势能', S['h2']))
story.append(body('动能由小车平动与杆质心运动构成，展开并利用 cos²θ+sin²θ=1：'))
for e in [r'T=\frac{1}{2}M\dot{x}^2+\frac{1}{2}m\left[(\dot{x}+l\dot{\theta}\cos\theta)^2+(l\dot{\theta}\sin\theta)^2\right]',
          r'T=\frac{1}{2}(M+m)\dot{x}^2+ml\dot{x}\dot{\theta}\cos\theta+\frac{1}{2}ml^2\dot{\theta}^2']:
    story += M(e)
story.append(body('势能（竖直向下为零势能点）：'))
story += M(r'V=mgl\cos\theta')
story.append(body('拉格朗日量 L = T − V。'))

story.append(Paragraph('2.4　欧拉–拉格朗日方程', S['h2']))
story.append(body('对广义坐标 q ∈ {x, θ}，有 d/dt(∂L/∂q̇) − ∂L/∂q = Q_q，其中 Q_x=F、Q_θ=0。对 x 方向：'))
story += M(r'(M+m)\ddot{x}+ml\ddot{\theta}\cos\theta-ml\dot{\theta}^2\sin\theta=F')
story.append(body('对 θ 方向（−mlẋθ̇sinθ 两项相消，除以 ml）：'))
story += M(r'\ddot{x}\cos\theta+l\ddot{\theta}-g\sin\theta=0')

story.append(Paragraph('2.5　显式化：解出 θ̈ 与 ẍ', S['h2']))
story.append(body('由上两式消元（先由第一式解出 ẍ，代入第二式），得仿真所用的完整非线性动力学：'))
for e in [r'\ddot{\theta}=\frac{g\sin\theta-\cos\theta\,\frac{F+ml\dot{\theta}^2\sin\theta}{M+m}}{l\left(1-\frac{m\cos^2\theta}{M+m}\right)}',
          r'\ddot{x}=\frac{F+ml\,(\dot{\theta}^2\sin\theta-\ddot{\theta}\cos\theta)}{M+m}']:
    story += M(e, fs=13)

story.append(Paragraph('2.6　平衡点线性化：一阶泰勒展开', S['h2']))
story.append(body('直立平衡点 θ=0、θ̇=0、ẋ=0、u=0。一阶泰勒展开 ż ≈ A(z−z*) + Bu，其中 A=∂f/∂z、B=∂f/∂u。'))
story.append(body('记 θ̈ 的分子 N、分母 D：N = g·sinθ − cosθ·(u+mlθ̇²sinθ)/(M+m)，D = l·(1 − m·cos²θ/(M+m))。'
    '在平衡点 ∂N/∂θ=g、∂D/∂θ=0、D=lM/(M+m)、∂N/∂u=−1/(M+m)，由商法则得：'))
story += M(r'\frac{\partial\ddot{\theta}}{\partial\theta}=\frac{(M+m)g}{Ml},\qquad \frac{\partial\ddot{\theta}}{\partial u}=-\frac{1}{Ml}')
story.append(body('于是 θ̈ ≈ (M+m)g/(Ml)·θ − F/(Ml)。'))
story.append(body('<b>关键：ẍ 的线性化须代入线性化后的 θ̈。</b> 由 ẍ=(u+ml(θ̇²sinθ−θ̈cosθ))/(M+m)，'
    'θ̇²sinθ 项在平衡点偏导为零，但 θ̈cosθ 项同时依赖 θ 与 u（因 θ̈ 含 u）。代入 θ̈ 的线性式：'))
story += M(r'\ddot{x}=\frac{u-ml\ddot{\theta}}{M+m}=-\frac{mg}{M}\theta+\frac{1}{M}u')
story.append(body('由此得线性状态空间模型 ż = Az + Bu：'))

A_rows = [['0', '1', '0', '0'], ['0', '0', '−mg/M', '0'],
          ['0', '0', '0', '1'], ['0', '0', '(M+m)g/(Ml)', '0']]
B_rows = [['0'], ['1/M'], ['0'], ['−1/(Ml)']]
mt = Table([[matrix_table(A_rows, 1.5 * cm), Spacer(1, 0.4 * cm), matrix_table(B_rows, 1.0 * cm)]],
           colWidths=[None, None, None])
mt.setStyle(TableStyle([('VALIGN', (0, 0), (-1, -1), 'MIDDLE'), ('ALIGN', (0, 0), (-1, -1), 'CENTER')]))
story.append(mt)
story.append(Spacer(1, 4))

story.append(Paragraph('2.7　开环稳定性与可控性', S['h2']))
story.append(body('矩阵 A 的特征值为 {0, 0, +ω, −ω}，其中 ω=√((M+m)g/(Ml)) ≈ 4.65 rad/s。正实部特征值 +ω 表明开环不稳定。'))
story.append(body('可控性矩阵 C=[B, AB, A²B, A³B]，逐项计算后其行列式可解析求得 det(C) = g²/(M⁴l⁴) > 0，'
    '故 rank(C)=4，系统<b>完全可控</b>，保证 LQR 有解且闭环可稳定。'))

# ============ 3 LQR ============
story.append(Paragraph('3　LQR 最优控制', S['h1']))
story.append(Paragraph('3.1　问题形式化', S['h2']))
story += M(r'J=\int_0^{\infty}\left(z^{\top}Qz+u^{\top}Ru\right)dt')
story.append(body('<b>物理含义</b>：zᵀQz 惩罚状态偏离零（对角元越大、对应状态被拉回越"急"）；uᵀRu 惩罚控制能耗（R 越大越"省力"但响应慢）。'))

story.append(Paragraph('3.2　庞特里亚金极小值原理推导', S['h2']))
story.append(body('构造哈密顿函数（协态向量 λ）：'))
story += M(r'\mathcal{H}=z^{\top}Qz+u^{\top}Ru+\lambda^{\top}(Az+Bu)')
story.append(body('协态方程（对状态求偏导取负）：'))
story += M(r'\dot{\lambda}=-\frac{\partial\mathcal{H}}{\partial z}=-(2Qz+A^{\top}\lambda)')
story.append(body('最优性条件（对控制求偏导为零）：'))
story += M(r'\frac{\partial\mathcal{H}}{\partial u}=2Ru+B^{\top}\lambda=0\;\Rightarrow\; u=-\frac{1}{2}R^{-1}B^{\top}\lambda')
story.append(body('令 λ=2Pz（P 待求），得 u=−R⁻¹BᵀPz=−Kz，K=R⁻¹BᵀP。对 λ=2Pz 求导并与协态方程联立（稳态 Ṗ=0），'
    '消去 z 得连续代数 Riccati 方程（CARE）：'))
story += M(r'A^{\top}P+PA-PBR^{-1}B^{\top}P+Q=0')

story.append(Paragraph('3.3　离散化与迭代求解', S['h2']))
story.append(body('取 Ad=I+A·Δt、Bd=B·Δt，用离散 Riccati 迭代（收敛于唯一镇定解）：'))
for e in [r'K\leftarrow(R+B_d^{\top}PB_d)^{-1}B_d^{\top}PA_d',
          r'P\leftarrow Q+A_d^{\top}PA_d-A_d^{\top}PB_dK']:
    story += M(e)
story.append(body('取 Q=diag(1,0.1,50,0.1)、R=0.1、Δt=0.01s，迭代十余次收敛，得 '
    'K ≈ [−2.954, −4.735, −46.339, −9.163]。|K₃|=46.339 最大，控制强依赖角度；K₄=−9.163 提供角速度阻尼。'))
story.append(body('LQR 既是性能基准，也是模仿学习的专家教师：u_expert(z)=−K·z。'))

# ============ 4 Transformer ============
story.append(Paragraph('4　Transformer 理论基础', S['h1']))
story.append(Paragraph('4.1　注意力机制动机', S['h2']))
story.append(body('注意力源于键值查询检索类比：查询 q 按"与键的相似度"对值做加权聚合。序列中每个位置既是查询又是键值，'
    '即<b>自注意力</b>——每个位置审视整条序列、按相关性聚合信息。'))

story.append(Paragraph('4.2　缩放点积注意力', S['h2']))
story += M(r'\mathrm{Attention}(Q,K,V)=\mathrm{softmax}\left(\frac{QK^{\top}}{\sqrt{d_k}}\right)V')
story.append(body('<b>为什么除以 √d_k？</b> 设 q、k 各分量独立同分布（均值 0、方差 1），内积 q·k=Σqᵢkᵢ 的方差为 d_k，'
    '即量级 √d_k。若不缩放，d_k 增大时内积落入 softmax 饱和区致梯度消失；除以 √d_k 使方差回到 1，保证梯度稳定。'))
story.append(body('softmax 将分数化为非负、和为 1 的权重，实现可微的"软"加权。'))

story.append(Paragraph('4.3　多头注意力', S['h2']))
for e in [r'\mathrm{MultiHead}(Q,K,V)=\mathrm{Concat}(head_1,\dots,head_h)W^O',
          r'head_i=\mathrm{Attention}(QW_i^Q,KW_i^K,VW_i^V)']:
    story += M(e)
story.append(body('不同头在不同子空间学习不同关系（一个关注角度、另一个关注速度），提升表达能力。本文 h=4、每头 d_k=8。'))

story.append(Paragraph('4.4　位置编码', S['h2']))
for e in [r'\mathrm{PE}(pos,2i)=\sin(pos/10000^{2i/d_{model}})',
          r'\mathrm{PE}(pos,2i+1)=\cos(pos/10000^{2i/d_{model}})']:
    story += M(e)
story.append(body('不同频率正弦波使不同维度随位置以不同速率变化，允许模型学到相对位置并可外推到更长序列。'))

story.append(Paragraph('4.5　前馈网络、残差与层归一化', S['h2']))
for e in [r'\mathrm{FFN}(x)=\mathrm{ReLU}(xW_1+b_1)W_2+b_2',
          r"x'=\mathrm{LayerNorm}(x+\mathrm{Sublayer}(x))"]:
    story += M(e)
story.append(body('层归一化对每个位置特征维标准化：μ=Σxᵢ/d，σ²=Σ(xᵢ−μ)²/d，x̂ᵢ=(xᵢ−μ)/√(σ²+ε)，yᵢ=γᵢx̂ᵢ+βᵢ。'
    '残差缓解梯度消失，LayerNorm 稳定训练、加速收敛。'))

# ============ 5 控制器 ============
story.append(Paragraph('5　Transformer 作为控制器', S['h1']))
story.append(body('取最近 T 步状态构成输入序列 X_t=[z_{t−T+1},…,z_t]ᵀ ∈ R^{T×4}，策略输出 u_t=π_φ(X_t)。'
    '与 LQR 只看当前状态不同，Transformer 利用整段历史并自适应加权。控制闭环如下：'))
story.append(draw_loop())
story.append(Paragraph('图 2　Transformer 控制闭环：状态序列经策略网络输出控制力，作用于动力学，新状态滑入历史窗口。', S['caption']))
story.append(Paragraph('5.1　模仿学习', S['h2']))
story += M(r'\mathcal{L}(\phi)=\frac{1}{N}\sum_{i=1}^{N}\left(\pi_\phi(X_i)-u_{expert}(z_i)\right)^2')
story.append(body('数据收集以 LQR 专家驱动仿真、随机初始化并周期性注入扰动，使样本覆盖偏离平衡的状态，'
    '让策略学会恢复能力、缓解误差累积问题。'))

# ============ 6 架构 ============
story.append(Paragraph('6　网络架构与训练', S['h1']))
story.append(body('单层 Transformer 编码器块：序列长度 T=8，输入维度 4，嵌入维度 d_model=32，头数 h=4（d_k=8），'
    'FFN 维度 d_ff=64，输出维度 1。前向结构：'))
story.append(draw_transformer())
story.append(Paragraph('图 3　Transformer 控制器前向结构（虚线为残差跳过连接，侧边为维度标注）。', S['caption']))
for e in [r'H=XW_e+b_e+\mathrm{PE}',
          r'H_1=\mathrm{LayerNorm}(H+\mathrm{MultiHead}(H))',
          r'H_2=\mathrm{LayerNorm}(H_1+\mathrm{FFN}(H_1))',
          r'u=H_2[T-1]\,w_o+b_o']:
    story += M(e)
story.append(body('其中 H₂[T−1] 为输出序列最后一个时间步，经线性输出头映射为标量控制力。'
    '优化用 Adam（lr=2×10⁻³，batch=64），训练 60 epoch，数据约 5.8 万条，反向传播手写并经验证。'))

# ============ 7 编码 ============
story.append(Paragraph('7　编码实现详解（纯 NumPy）', S['h1']))

story.append(Paragraph('7.1　物理仿真：非线性动力学与 RK4', S['h2']))
story.append(Preformatted('''def derivatives(z, u):          # 对应式(1)(2)
    x, xd, th, thd = z
    s, c = np.sin(th), np.cos(th)
    denom = M + m
    thdd = (G*s - c*(u + m*L*thd*thd*s)/denom) / (L*(1 - m*c*c/denom))
    xdd = (u + m*L*(thd*thd*s - thdd*c)) / denom
    return np.array([xd, xdd, thd, thdd])

def rk4(z, u, dt):                # 四阶 Runge-Kutta
    k1 = derivatives(z, u)
    k2 = derivatives(z + k1*dt/2, u)
    k3 = derivatives(z + k2*dt/2, u)
    k4 = derivatives(z + k3*dt, u)
    return z + (k1 + 2*k2 + 2*k3 + k4)*dt/6''', S['code']))

story.append(Paragraph('7.2　LQR：线性化与离散 Riccati 迭代', S['h2']))
story.append(Preformatted('''def solve_lqr(Q_diag, R):
    A, B = build_AB()              # 式(A,B)
    Q = np.diag(Q_diag)
    Ad = np.eye(4) + A * DT        # Euler discretization
    Bd = B * DT
    P = Q.copy()
    for _ in range(2000):
        S  = R + Bd.T @ P @ Bd     # (R + Bd'PBd) 标量
        K  = (Bd.T @ P @ Ad) / S
        Pn = Q + Ad.T @ P @ Ad - Ad.T @ P @ Bd @ K
        if np.abs(Pn - P).max() < 1e-13: break
        P = Pn
    return (Bd.T @ P @ Ad) / (R + Bd.T @ P @ Bd)   # 1x4''', S['code']))
story.append(body('控制为标量，R+BdᵀPBd 为 1×1，求逆退化为除法。'))

story.append(Paragraph('7.3　位置编码与嵌入', S['h2']))
story.append(Preformatted('''def positional_encoding(T, d):
    pe = np.zeros((T, d))
    for t in range(T):
        for i in range(0, d, 2):
            pe[t, i]   = np.sin(t / 10000**(i/d))
            pe[t, i+1] = np.cos(t / 10000**(i/d))
    return pe
PE = positional_encoding(T, D_MODEL)   # (T, 32)''', S['code']))

story.append(Paragraph('7.4　多头注意力：前向', S['h2']))
story.append(Preformatted('''def attention_forward(h, p):
    B, T, d = h.shape
    Q = h @ p['Wq'] + p['bq']       # (B,T,32)
    K = h @ p['Wk'] + p['bk']
    V = h @ p['Wv'] + p['bv']
    Qh = Q.reshape(B, T, H, DK).transpose(0, 2, 1, 3)  # (B,4,T,8)
    Kh = K.reshape(B, T, H, DK).transpose(0, 2, 1, 3)
    Vh = V.reshape(B, T, H, DK).transpose(0, 2, 1, 3)
    scores = Qh @ Kh.transpose(0, 1, 3, 2) / np.sqrt(DK)  # (B,4,T,T)
    attn = softmax(scores, axis=-1)        # 沿键维归一
    ctx = (attn @ Vh).transpose(0, 2, 1, 3).reshape(B, T, d)
    out = ctx @ p['Wo'] + p['bo']
    return out, (h, Qh, Kh, Vh, scores, attn, ctx)''', S['code']))
story.append(body('reshape(B,T,H,DK).transpose(0,2,1,3) 将 32 维按 4 头切为 (B,4,T,8)。scores[q,k] 为查询位置 q 与键位置 k 的缩放内积。'))

story.append(Paragraph('7.5　多头注意力：反向', S['h2']))
story.append(Preformatted('''def attention_backward(dout, cache, p):
    h, Qh, Kh, Vh, scores, attn, ctx = cache[:7]
    dctx = dout @ p['Wo'].T
    dWo = np.einsum('bti,btj->ij', ctx, dout)   # sum_b ctx^T dout
    dctx = dctx.reshape(B, T, H, DK).transpose(0, 2, 1, 3)
    dattn = dctx @ Vh.transpose(0, 1, 3, 2)
    dVh = attn.transpose(0, 1, 3, 2) @ dctx
    dscores = softmax_backward(dattn, attn) / np.sqrt(DK)
    dQh = dscores @ Kh
    dKh = dscores.transpose(0, 1, 3, 2) @ Qh
    dQ = dQh.transpose(0, 2, 1, 3).reshape(B, T, d)
    dK = dKh.transpose(0, 2, 1, 3).reshape(B, T, d)
    dV = dVh.transpose(0, 2, 1, 3).reshape(B, T, d)
    dh = dQ @ p['Wq'].T + dK @ p['Wk'].T + dV @ p['Wv'].T
    dWq = np.einsum('bti,btj->ij', h, dQ)      # 同理 Wk,Wv,偏置
    return dh, (dWq, dbq, dWk, dbk, dWv, dbv, dWo, dbo)''', S['code']))

story.append(Paragraph('7.6　LayerNorm：前向与反向', S['h2']))
story.append(Preformatted('''def layernorm_forward(x, gamma, beta):
    mu = x.mean(-1, keepdims=True)
    var = x.var(-1, keepdims=True)
    xhat = (x - mu) / np.sqrt(var + EPS)
    return gamma * xhat + beta, (x, mu, var, xhat)

def layernorm_backward(dy, cache, gamma):
    x, mu, var, xhat = cache
    D = x.shape[-1]
    dxhat = dy * gamma
    dvar = (dxhat*(x-mu)).sum(-1,keepdims=True)*(-0.5)*(var+EPS)**(-1.5)
    dmu  = (dxhat*(-1/np.sqrt(var+EPS))).sum(-1,keepdims=True) \\
         + dvar*(-2*(x-mu)).sum(-1,keepdims=True)/D
    dx = dxhat/np.sqrt(var+EPS) + dvar*2*(x-mu)/D + dmu/D
    dgamma = (dy*xhat).sum(axis=(0,1))
    dbeta  = dy.sum(axis=(0,1))
    return dx, dgamma, dbeta''', S['code']))

story.append(Paragraph('7.7　前馈网络、整体前向与数据收集/训练', S['h2']))
story.append(Preformatted('''def forward(X, p):
    h = X @ p['We'] + p['be']           # linear embedding (T,4)->(T,32)
    h = h + PE                            # positional encoding
    attn_out, c1 = attention_forward(h, p)
    h1, _ = layernorm_forward(h + attn_out, p['g1'], p['b1ln'])
    ffn_out, c2 = ffn_forward(h1, p)      # ReLU(W1)+W2
    h2, _ = layernorm_forward(h1 + ffn_out, p['g2'], p['b2ln'])
    last = h2[:, -1, :]                   # last-position output
    u = last @ p['Wo_final'] + p['b_final']
    return u, (X, h, c1, c2, h1, h2, last)

# data collection: LQR expert + periodic perturbation
for ep in range(n_episodes):
    z = random_init_state(); hist = [z.copy()]
    for step in range(steps):
        u = lqr_u(hist[-1])               # expert action
        if step % 25 == 24:               # perturbation
            hist[-1][3] += uniform(-1.5, 1.5)
        z = rk4(hist[-1], u, DT)
        if abs(z[2]) > pi/4: break
        hist.append(z)
        if len(hist) >= T:
            Xs.append(hist[-T:]); ys.append(lqr_u(hist[-1]))

# training: MSE + Adam
for epoch in range(epochs):
    for Xb, yb in minibatch(X, y):
        u, cache = forward(Xb, p)
        loss = mean((u - yb) ** 2)
        grads = backward(2*(u-yb)/batch, cache, p)
        adam_step(p, grads)''', S['code']))

# ============ 8 实验 ============
story.append(Paragraph('8　实验结果', S['h1']))
story.append(body('训练损失由约 0.92 快速降至 10⁻³ 量级。闭环测试（800 步）结果：'))
cl = Table([['初始角度 θ₀', '结果', '峰值 |θ|max'],
            ['0.10 rad', '稳定', '5.7°'],
            ['0.20 rad', '稳定', '11.4°'],
            ['0.30 rad', '稳定', '17.1°']], colWidths=[3.0 * cm, 2.4 * cm, 2.6 * cm])
cl.setStyle(TableStyle([
    ('BACKGROUND', (0, 0), (-1, 0), ACCENT), ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
    ('FONTNAME', (0, 0), (-1, -1), 'YaHei'), ('FONTNAME', (0, 0), (-1, 0), 'YaHeiBold'),
    ('FONTSIZE', (0, 0), (-1, -1), 9.5), ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
    ('GRID', (0, 0), (-1, -1), 0.4, colors.HexColor('#cccccc')),
    ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#f0f4f8')]),
    ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'), ('TOPPADDING', (0, 0), (-1, -1), 4),
    ('BOTTOMPADDING', (0, 0), (-1, -1), 4)]))
story.append(cl)
story.append(Spacer(1, 5))
story.append(body('施加 1.2 rad/s 角速度脉冲后约 3 s 内恢复（峰值 5.9°）。Transformer 输出与 LQR 高度一致；'
    '注意力热力图显示控制器主要关注最近几步状态，符合倒立摆短时记忆特性。'))

# ============ 9 训练指南 ============
story.append(Paragraph('9　手动训练与运行指南', S['h1']))
story.append(Paragraph('9.1　环境准备', S['h2']))
story.append(body('仅依赖 Python 3.9+ 与 NumPy（本文用 Python 3.12、NumPy 2.x），无需 PyTorch/TensorFlow：'))
story.append(Preformatted('pip install numpy', S['code']))
story.append(Paragraph('9.2　一键运行与脚本内部执行顺序', S['h2']))
story.append(body('训练脚本单文件自包含，直接运行即可：'))
story.append(Preformatted('python train_transformer_controller.py', S['code']))
story.append(body('脚本按<b>六个阶段</b>顺序执行：'))
story.append(body('<b>① LQR 增益求解</b>（脚本顶层，导入即执行）——构建 A/B 矩阵、离散 Riccati 迭代，打印 '
    'K ≈ [−2.954, −4.735, −46.339, −9.163]。'))
story.append(body('<b>② 梯度检查</b>（grad_check）——中心差分数值梯度与手写解析梯度逐参数比对，'
    '验证反向传播正确（误差约 1e-9 ~ 1e-7；若出现 1e-2 量级说明反向传播有 bug，须先修正）。'))
story.append(body('<b>③ 数据收集</b>（collect_data）——LQR 专家 + 周期性扰动，生成约 5.8 万条 (状态序列, 动作) 样本。'))
story.append(body('<b>④ 监督训练</b>（train 内循环）——Adam 优化 MSE，60 epoch，损失从约 0.92 降至 1e-3 量级。'))
story.append(body('<b>⑤ 闭环测试</b>（closed_loop_test）——θ₀=0.1/0.2/0.3 各跑 800 步，报告稳定性。'))
story.append(body('<b>⑥ 导出权重</b>（export_weights）——参数与配置序列化为 JSON，供网页加载。'))
story.append(Paragraph('9.3　分步手动执行（可选）', S['h2']))
story.append(body('注释脚本 main 部分，逐阶段交互式执行：'))
story.append(Preformatted('''# 阶段一：LQR
from train_transformer_controller import K_LQR
print(K_LQR)
# 阶段二：梯度检查
from train_transformer_controller import grad_check
grad_check()
# 阶段三：数据收集
from train_transformer_controller import collect_data
X, y = collect_data(300, 200)      # (N, 8, 4)
# 阶段四：训练（内部含梯度检查+数据收集）
from train_transformer_controller import train
p = train()
# 阶段五：闭环测试
from train_transformer_controller import closed_loop_test
print(closed_loop_test(p, init_th=0.2))
# 阶段六：导出
from train_transformer_controller import export_weights
export_weights(p, 'my_weights.json')''', S['code']))
story.append(Paragraph('9.4　超参数调整', S['h2']))
story.append(body('物理参数 G/M/m/L（被控对象）；LQR 权重 solve_lqr([1.0,0.1,50.0,0.1], 0.1)（专家行为）；'
    '网络结构 D_MODEL/NUM_HEADS/D_FF/T（模型规模）；训练 n_episodes/steps/epochs/batch/lr（数据量与优化）。'))
story.append(Paragraph('9.5　常见问题', S['h2']))
story.append(body('梯度检查失败→反向传播有误（softmax 沿错维归一化、LayerNorm 缺 dμ/dvar 项、reshape/transpose 顺序错）；'
    '闭环发散→增加扰动或数据量、增大 θ 初始范围、引入 DAgger；网页加载失败→确认权重与 HTML 同目录或已内嵌。'))

# ============ 10 结论 ============
story.append(Paragraph('10　结论', S['h1']))
story.append(body('本文将倒立摆控制形式化为序列决策任务，用 Transformer 作控制器、以 LQR 为专家做模仿学习，'
    '实验证明纯 NumPy 手写的控制器能稳定维持平衡并对扰动鲁棒。优势：无需精确线性化即可拟合非线性策略、'
    '注意力提供可解释性、架构可扩展；局限：性能受限于专家与数据分布、误差累积需 DAgger 修正、推理计算量高于线性 LQR。'))

story.append(Paragraph('参考文献', S['h1']))
refs = ['[1] Vaswani, A., et al. (2017). Attention is All You Need. NeurIPS, 30.',
        '[2] Ogata, K. (2010). Modern Control Engineering (5th ed.). Prentice Hall.',
        '[3] Åström, K. J., & Murray, R. M. (2008). Feedback Systems. Princeton University Press.',
        '[4] Kwakernaak, H., & Sivan, R. (1972). Linear Optimal Control Systems. Wiley-Interscience.',
        '[5] Ross, S., Gordon, G., & Bagnell, D. (2011). A Reduction of Imitation Learning to No-Regret Online Learning. AISTATS.',
        '[6] Chen, L., et al. (2021). Decision Transformer: RL via Sequence Modeling. NeurIPS.']
for r in refs:
    story.append(Paragraph(r, S['ref']))

doc.build(story)
print('PDF generated: transformer_cartpole.pdf')
