import numpy as np
import matplotlib.pyplot as plt

# 線形回帰モデル
def linear_reg(x, w0, w1):
    y = w0 * x + w1
    
    return y

# 線形回帰モデルの平均二乗誤差(MSE)の計算
def mse_linear_reg(xs, ts, w0, w1):
    ys = linear_reg(xs, w0, w1)
    mse = np.mean((ts - ys) ** 2)
    
    return mse

# dw0, dw1の計算
def dw_linear_reg(xs, ts, w0, w1):
    ys = w0 * xs + w1
    dw0 = -2 * np.mean((ys - ts) * xs)
    dw1 = -2 * np.mean(ys - ts)
    
    return dw0, dw1

# グラフ表示設定
def plt_settings():
    plt.legend()
    plt.xlim(x_range)
    plt.ylim(t_range)
    plt.xlabel('x (temperature)')
    plt.ylabel('t (sales)')
    plt.grid(axis='both')

np.set_printoptions(precision=3, suppress=True)

config = {
    "font.size":14,
    "figure.figsize":(7, 4)
}
plt.rcParams.update(config)


n = 50
noise = 10
np.random.seed(1)

xs = np.random.rand(n) * 30
ts = (xs * 0.1) ** 4 + 1 * xs + 50 + np.random.randn(n) * noise
x_range = (0, 30)
t_range = (0, 200)

print('xs (temperature)', xs)
print('')
print('ts (sales)', ts)
print('')

# 勾配法による線形回帰学習
# 勾配法
alpha = 0.001
init_w0 = 1.0
init_w1 = 40.0
w0s = [init_w0]
w1s = [init_w1]
init_mse = mse_linear_reg(xs, ts, init_w0, init_w1)
mses = [init_mse]

for i in range(10000):
    dw0, dw1 = dw_linear_reg(xs, ts, w0s[-1], w1s[-1])
    next_dw0 = w0s[-1] + alpha * dw0
    next_dw1 = w1s[-1] + alpha * dw1
    w0s.append(next_dw0)
    w1s.append(next_dw1)
    
    next_mse = mse_linear_reg(xs, ts, w0s[-1], w1s[-1])
    mses.append(next_mse)

# 学習曲線表示
print(f'MMSE = {mses[-1]:.2f}')
print('The learning curve was output.\n')
plt.plot(mses)
plt.xlim(-100, 10000)
plt.ylim(150.0, 200.0)
plt.xlabel('steps')
plt.ylabel('MSE')
plt.grid(axis='both')
plt.show()

# グラフ表示
print(f'w0 = {w0s[-1]:.3f}, w1 = {w1s[-1]:.3f}, MMSE = {mses[-1]:.3f}')
print('The data and a linear regression model for it were output.\n')

xs_pred = np.linspace(x_range[0], x_range[1], 50)
ys_pred = linear_reg(xs_pred, w0s[-1], w1s[-1])

plt.scatter(xs, ts, label='data')
plt.plot(xs_pred, ys_pred, 'green', label='approximate straight line')
plt_settings()
plt.show()