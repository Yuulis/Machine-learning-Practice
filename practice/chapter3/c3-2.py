import numpy as np
import matplotlib.pyplot as plt
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense

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
np.random.seed(2)

xs = np.random.rand(n) * 30
ts = (xs * 0.1) ** 4 + 1 * xs + 50 + np.random.randn(n) * noise
x_range = (0, 30)
t_range = (0, 200)

# モデル生成
model = Sequential([
    Dense(32, activation='sigmoid', input_dim=1),
    Dense(1, activation='linear')
])
model.compile(optimizer='adam', loss='mse')

# 学習
history = model.fit(
    xs.reshape(-1, 1),
    ts.reshape(-1, 1),
    batch_size=len(xs),
    epochs=10000,
    verbose=0,
)

# 予測
xs_pred = np.linspace(x_range[0], x_range[1], 100)
ys_pred = model.predict(xs_pred.reshape(-1, 1))

# 学習曲線表示
print(f"MSE = {history.history['loss'][-1]:.3f}")
print('The learning curve was output.\n')
plt.plot(history.history['loss'])
plt.title('MSE')
plt.xlabel('epochs')
plt.grid(axis='both')
plt.show()

# グラフ表示
print('The data and a neural network model for it were output.\n')
plt.scatter(xs, ts, label='data')
plt.plot(xs_pred, ys_pred, 'green', label='neural network model')

plt_settings()
plt.show()
