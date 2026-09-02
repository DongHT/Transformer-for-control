# -*- coding: utf-8 -*-
"""
训练一个 Transformer 控制器用于倒立摆（Cart-Pole）稳定控制。
方法：模仿学习（Behavior Cloning），专家教师为 LQR。
纯 numpy 实现 Transformer 前向/反向传播，训练后导出权重 JSON 供网页端推理。
"""
import numpy as np
import json

np.random.seed(0)

# ================= 物理参数 =================
G, M, L = 9.81, 1.0, 0.5
m = 0.1
DT = 0.01

# ================= LQR 专家 =================
def build_AB():
    A = np.array([[0, 1, 0, 0],
                  [0, 0, -m * G / M, 0],
                  [0, 0, 0, 1],
                  [0, 0, (M + m) * G / (M * L), 0]])
    B = np.array([[0], [1 / M], [0], [-1 / (M * L)]])
    return A, B

def solve_lqr(Q_diag, R):
    A, B = build_AB()
    Q = np.diag(Q_diag)
    Ad = np.eye(4) + A * DT
    Bd = B * DT
    P = Q.copy()
    for _ in range(2000):
        S = R + Bd.T @ P @ Bd
        K = (Bd.T @ P @ Ad) / S
        Pn = Q + Ad.T @ P @ Ad - Ad.T @ P @ Bd @ K
        if np.abs(Pn - P).max() < 1e-13:
            break
        P = Pn
    S = R + Bd.T @ P @ Bd
    K = (Bd.T @ P @ Ad) / S
    return K.reshape(1, -1)

K_LQR = solve_lqr([1.0, 0.1, 50.0, 0.1], 0.1).reshape(-1)
print("LQR 增益 K =", np.round(K_LQR, 3))

def lqr_u(z):
    return float(-K_LQR @ z)

# ================= 倒立摆仿真 =================
def derivatives(z, u):
    x, xd, th, thd = z
    s, c = np.sin(th), np.cos(th)
    denom = M + m
    thdd = (G * s - c * (u + m * L * thd * thd * s) / denom) / (L * (1 - m * c * c / denom))
    xdd = (u + m * L * (thd * thd * s - thdd * c)) / denom
    return np.array([xd, xdd, thd, thdd])

def rk4(z, u, dt):
    k1 = derivatives(z, u)
    k2 = derivatives(z + k1 * dt / 2, u)
    k3 = derivatives(z + k2 * dt / 2, u)
    k4 = derivatives(z + k3 * dt, u)
    return z + (k1 + 2 * k2 + 2 * k3 + k4) * dt / 6

# ================= Transformer 网络 =================
INPUT_DIM = 4
D_MODEL = 32
NUM_HEADS = 4
D_K = D_MODEL // NUM_HEADS
D_FF = 64
T = 8
EPS = 1e-5

def positional_encoding(T_len, d):
    pe = np.zeros((T_len, d))
    pos = np.arange(T_len)[:, None]
    div = 10000 ** (np.arange(0, d, 2) / d)
    pe[:, 0::2] = np.sin(pos * div)
    pe[:, 1::2] = np.cos(pos * div)
    return pe

PE = positional_encoding(T, D_MODEL)

def init_params():
    p = {}
    p['We'] = np.random.randn(INPUT_DIM, D_MODEL) * 0.1
    p['be'] = np.zeros(D_MODEL)
    for name in ['Wq', 'Wk', 'Wv', 'Wo']:
        p[name] = np.random.randn(D_MODEL, D_MODEL) * np.sqrt(2.0 / D_MODEL)
    p['bq'] = np.zeros(D_MODEL); p['bk'] = np.zeros(D_MODEL)
    p['bv'] = np.zeros(D_MODEL); p['bo'] = np.zeros(D_MODEL)
    p['W1'] = np.random.randn(D_MODEL, D_FF) * np.sqrt(2.0 / D_MODEL)
    p['b1'] = np.zeros(D_FF)
    p['W2'] = np.random.randn(D_FF, D_MODEL) * np.sqrt(2.0 / D_FF)
    p['b2'] = np.zeros(D_MODEL)
    p['g1'] = np.ones(D_MODEL); p['b1ln'] = np.zeros(D_MODEL)
    p['g2'] = np.ones(D_MODEL); p['b2ln'] = np.zeros(D_MODEL)
    p['Wo_final'] = np.random.randn(D_MODEL, 1) * 0.1
    p['b_final'] = np.zeros(1)
    return p

def softmax(x, axis=-1):
    x = x - x.max(axis=axis, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=axis, keepdims=True)

def softmax_backward(dout, out):
    s = (dout * out).sum(axis=-1, keepdims=True)
    return out * (dout - s)

def layernorm_forward(x, gamma, beta):
    mu = x.mean(axis=-1, keepdims=True)
    var = x.var(axis=-1, keepdims=True)
    std = np.sqrt(var + EPS)
    xhat = (x - mu) / std
    y = gamma * xhat + beta
    return y, (x, mu, var, std, xhat)

def layernorm_backward(dy, cache, gamma):
    x, mu, var, std, xhat = cache
    D = x.shape[-1]
    dxhat = dy * gamma
    dvar = (dxhat * (x - mu)).sum(axis=-1, keepdims=True) * (-0.5) * (var + EPS) ** (-1.5)
    dmu = (dxhat * (-1.0 / std)).sum(axis=-1, keepdims=True) + dvar * (-2.0 * (x - mu)).sum(axis=-1, keepdims=True) / D
    dx = dxhat / std + dvar * 2.0 * (x - mu) / D + dmu / D
    dgamma = (dy * xhat).sum(axis=(0, 1))
    dbeta = dy.sum(axis=(0, 1))
    return dx, dgamma, dbeta

def attention_forward(h, p):
    B, T_len, d = h.shape
    Q = h @ p['Wq'] + p['bq']
    K = h @ p['Wk'] + p['bk']
    V = h @ p['Wv'] + p['bv']
    Qh = Q.reshape(B, T_len, NUM_HEADS, D_K).transpose(0, 2, 1, 3)
    Kh = K.reshape(B, T_len, NUM_HEADS, D_K).transpose(0, 2, 1, 3)
    Vh = V.reshape(B, T_len, NUM_HEADS, D_K).transpose(0, 2, 1, 3)
    scores = Qh @ Kh.transpose(0, 1, 3, 2) / np.sqrt(D_K)
    attn = softmax(scores, axis=-1)
    ctx = attn @ Vh
    ctx = ctx.transpose(0, 2, 1, 3).reshape(B, T_len, d)
    out = ctx @ p['Wo'] + p['bo']
    return out, (h, Qh, Kh, Vh, scores, attn, ctx, Q, K, V)

def attention_backward(dout, cache, p):
    h, Qh, Kh, Vh, scores, attn, ctx, Q, K, V = cache
    B, T_len, d = h.shape
    dctx = dout @ p['Wo'].T
    dWo = np.einsum('bti,btj->ij', ctx, dout)
    dbo = dout.sum(axis=(0, 1))
    dctx = dctx.reshape(B, T_len, NUM_HEADS, D_K).transpose(0, 2, 1, 3)
    dattn = dctx @ Vh.transpose(0, 1, 3, 2)
    dVh = attn.transpose(0, 1, 3, 2) @ dctx
    dscores = softmax_backward(dattn, attn) / np.sqrt(D_K)
    dQh = dscores @ Kh
    dKh = dscores.transpose(0, 1, 3, 2) @ Qh
    dQ = dQh.transpose(0, 2, 1, 3).reshape(B, T_len, d)
    dK = dKh.transpose(0, 2, 1, 3).reshape(B, T_len, d)
    dV = dVh.transpose(0, 2, 1, 3).reshape(B, T_len, d)
    dh = dQ @ p['Wq'].T + dK @ p['Wk'].T + dV @ p['Wv'].T
    dWq = np.einsum('bti,btj->ij', h, dQ)
    dWk = np.einsum('bti,btj->ij', h, dK)
    dWv = np.einsum('bti,btj->ij', h, dV)
    dbq = dQ.sum(axis=(0, 1))
    dbk = dK.sum(axis=(0, 1))
    dbv = dV.sum(axis=(0, 1))
    return dh, (dWq, dbq, dWk, dbk, dWv, dbv, dWo, dbo)

def ffn_forward(h, p):
    f1 = h @ p['W1'] + p['b1']
    a = np.maximum(0, f1)
    f2 = a @ p['W2'] + p['b2']
    return f2, (h, f1, a)

def ffn_backward(dout, cache, p):
    h, f1, a = cache
    dW2 = np.einsum('bti,btj->ij', a, dout)
    db2 = dout.sum(axis=(0, 1))
    da = dout @ p['W2'].T
    df1 = da * (f1 > 0)
    dW1 = np.einsum('bti,btj->ij', h, df1)
    db1 = df1.sum(axis=(0, 1))
    dh = df1 @ p['W1'].T
    return dh, (dW1, db1, dW2, db2)

def forward(X, p):
    h = X @ p['We'] + p['be']
    h = h + PE[:T]
    attn_out, c_attn = attention_forward(h, p)
    h1, c_ln1 = layernorm_forward(h + attn_out, p['g1'], p['b1ln'])
    ffn_out, c_ffn = ffn_forward(h1, p)
    h2, c_ln2 = layernorm_forward(h1 + ffn_out, p['g2'], p['b2ln'])
    last = h2[:, -1, :]
    u = last @ p['Wo_final'] + p['b_final']
    cache = (X, h, c_attn, c_ln1, h1, c_ffn, c_ln2, h2, last)
    return u, cache

def backward(du, cache, p):
    X, h, c_attn, c_ln1, h1, c_ffn, c_ln2, h2, last = cache
    g = {}
    g['Wo_final'] = last.T @ du
    g['b_final'] = du.sum(axis=0)
    dh2 = np.zeros_like(h2)
    dh2[:, -1, :] = du @ p['Wo_final'].T

    d_pre2, dg2, db2ln = layernorm_backward(dh2, c_ln2, p['g2'])
    g['g2'] = dg2; g['b2ln'] = db2ln
    dh1_f, (dW1, db1, dW2, db2) = ffn_backward(d_pre2, c_ffn, p)
    g['W1'], g['b1'], g['W2'], g['b2'] = dW1, db1, dW2, db2
    dh1 = d_pre2 + dh1_f  # 残差：h1 + ffn

    d_pre1, dg1, db1ln = layernorm_backward(dh1, c_ln1, p['g1'])
    g['g1'] = dg1; g['b1ln'] = db1ln
    dh_attn, attn_g = attention_backward(d_pre1, c_attn, p)
    g['Wq'], g['bq'], g['Wk'], g['bk'], g['Wv'], g['bv'], g['Wo'], g['bo'] = attn_g
    dh = d_pre1 + dh_attn  # 残差：h + attn

    g['We'] = np.einsum('bti,btj->ij', X, dh)
    g['be'] = dh.sum(axis=(0, 1))
    return g

# ================= 梯度检查 =================
def grad_check():
    p = init_params()
    B = 3
    X = np.random.randn(B, T, INPUT_DIM)
    y = np.random.randn(B, 1)
    u, cache = forward(X, p)
    du = 2 * (u - y) / B
    g = backward(du, cache, p)
    eps = 1e-6
    for name in p:
        w = p[name]
        gw = g[name]
        assert w.shape == gw.shape, f"{name}: {w.shape} vs {gw.shape}"
        w_flat = w.ravel()
        gw_flat = gw.ravel()
        idxs = np.random.choice(len(w_flat), size=min(4, len(w_flat)), replace=False)
        max_rel = 0.0
        for idx in idxs:
            old = w_flat[idx]
            w_flat[idx] = old + eps
            up, _ = forward(X, p)
            w_flat[idx] = old - eps
            um, _ = forward(X, p)
            w_flat[idx] = old
            lp = np.mean((up - y) ** 2)
            lm = np.mean((um - y) ** 2)
            num = (lp - lm) / (2 * eps)
            ana = gw_flat[idx]
            denom = max(abs(num), abs(ana), 1e-8)
            max_rel = max(max_rel, abs(num - ana) / denom)
        print(f"  {name:10s} shape={w.shape} 相对误差={max_rel:.2e}")
    return None

# ================= 数据收集（LQR 专家） =================
def collect_data(n_episodes, steps_per_episode):
    Xs, ys = [], []
    for ep in range(n_episodes):
        z = np.array([np.random.uniform(-0.5, 0.5),
                      np.random.uniform(-0.3, 0.3),
                      np.random.uniform(-0.25, 0.25),
                      np.random.uniform(-0.3, 0.3)])
        hist = [z.copy()]
        for step in range(steps_per_episode):
            u = lqr_u(hist[-1])
            # 周期性扰动，让数据覆盖偏离平衡的状态
            if step % 25 == 24:
                z_p = hist[-1].copy()
                z_p[3] += np.random.uniform(-1.5, 1.5)
                hist.append(z_p)
            z_next = rk4(hist[-1], u, DT)
            if abs(z_next[2]) > np.pi / 4 or abs(z_next[0]) > 5:
                break
            hist.append(z_next)
            if len(hist) >= T:
                Xs.append(np.array(hist[-T:]))
                ys.append(lqr_u(hist[-1]))
    X = np.array(Xs, dtype=np.float64)
    y = np.array(ys, dtype=np.float64).reshape(-1, 1)
    return X, y

# ================= Adam 优化器 =================
class Adam:
    def __init__(self, lr=1e-3):
        self.lr = lr
        self.m = {}
        self.v = {}
        self.t = 0
    def step(self, p, g):
        self.t += 1
        for name in p:
            if name not in self.m:
                self.m[name] = np.zeros_like(p[name])
                self.v[name] = np.zeros_like(p[name])
            self.m[name] = 0.9 * self.m[name] + 0.1 * g[name]
            self.v[name] = 0.999 * self.v[name] + 0.001 * g[name] ** 2
            m_hat = self.m[name] / (1 - 0.9 ** self.t)
            v_hat = self.v[name] / (1 - 0.999 ** self.t)
            p[name] -= self.lr * m_hat / (np.sqrt(v_hat) + 1e-8)

# ================= 训练 =================
def train():
    grad_check()
    X, y = collect_data(300, 200)
    print(f"训练数据: {X.shape[0]} 个样本, 形状 {X.shape}")
    p = init_params()
    opt = Adam(lr=2e-3)
    n = X.shape[0]
    batch = 64
    n_batches = n // batch
    epochs = 60
    for epoch in range(epochs):
        perm = np.random.permutation(n)
        total_loss = 0.0
        for b in range(n_batches):
            idx = perm[b * batch:(b + 1) * batch]
            Xb = X[idx]
            yb = y[idx]
            u, cache = forward(Xb, p)
            loss = np.mean((u - yb) ** 2)
            du = 2 * (u - yb) / batch
            g = backward(du, cache, p)
            opt.step(p, g)
            total_loss += loss
        if epoch % 5 == 0 or epoch == epochs - 1:
            print(f"Epoch {epoch:3d}  平均 Loss = {total_loss / n_batches:.6f}")
    return p

# ================= 闭环评估 =================
def closed_loop_test(p, horizon=800, init_th=0.2):
    z = np.array([0.0, 0.0, init_th, 0.0])
    hist = [z.copy() for _ in range(T)]
    max_th = 0.0
    for step in range(horizon):
        seq = np.array(hist[-T:]).reshape(1, T, INPUT_DIM)
        u, _ = forward(seq, p)
        u = float(u[0, 0])
        z = rk4(z, u, DT)
        max_th = max(max_th, abs(z[2]))
        if abs(z[2]) > np.pi / 4 or abs(z[0]) > 5:
            return False, step, max_th
        hist.append(z.copy())
    return True, horizon, max_th

# ================= 导出权重 =================
def export_weights(p, path):
    out = {}
    for k, v in p.items():
        out[k] = v.tolist()
    out['config'] = {'input_dim': INPUT_DIM, 'd_model': D_MODEL, 'num_heads': NUM_HEADS,
                     'd_ff': D_FF, 'T': T, 'eps': EPS, 'pe': PE.tolist()}
    with open(path, 'w') as f:
        json.dump(out, f)
    print(f"权重已导出: {path}")

if __name__ == '__main__':
    p = train()
    # 闭环测试
    for th0 in [0.1, 0.2, 0.3]:
        ok, steps, max_th = closed_loop_test(p, init_th=th0)
        print(f"闭环测试 θ0={th0}: {'稳定' if ok else f'失稳于第{steps}步'}, 最大|θ|={max_th * 180 / np.pi:.1f}°")
    export_weights(p, 'transformer_weights.json')
