import numpy as np
import matplotlib.pyplot as plt
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense
from tensorflow.python.keras.callbacks import EarlyStopping

# 2Dグラフ表示
def show_data2d():
    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(111, projection='3d')

    ax.scatter(xs_train[:, 0], xs_train[:, 1], ts_train[:, 0], s=50, label='train')
    ax.scatter(xs_test[:, 0], xs_test[:, 1], ts_test[:, 0], s=50, label='test')

    for i in range(len(ts)):
        plt.plot(
            [xs[i, 0], xs[i, 0]],
            [xs[i, 1], xs[i, 1]],
            [0, ts[i]], 'gray', alpha=0.2
        )
    ax.set_xlabel('$x_0$')
    ax.set_ylabel('$x_1$')
    ax.legend()

    ax.view_init(elev=60, azim=-30)
    return ax

np.set_printoptions(precision=3, suppress=True)

config = {
    "font.size":14,
    "figure.figsize":(7, 4)
}
plt.rcParams.update(config)


n = 100
n_train = 80
noise = 0.25
np.random.seed(10)

xs = np.random.rand(n, 2) @ np.array([[10, 0], [0, 6]])
ts = np.sin(xs[:, 0]) + np.cos(xs[:, 1]) + noise * np.random.randn(n) + 2

xs_train = xs[:n_train, :]
ts_train = ts[:n_train].reshape(-1, 1)
xs_test = xs[n_train:, :]
ts_test = ts[n_train:].reshape(-1, 1)

x_range = ((0, 10), (0, 6))

n_batch = len(xs)

# Early Stopping
early_stopping = EarlyStopping(
    monitor='val_loss',
    patience=1000,
)

# モデル生成
model = Sequential([
    Dense(32, activation='sigmoid', input_dim=2),
    Dense(32, activation='sigmoid'),
    Dense(1, activation='linear'),
])
model.compile(optimizer='adam', loss='mse')

# 学習
history = model.fit(
    xs_train,
    ts_train,
    validation_data=(xs_test, ts_test),
    batch_size=n_train,
    epochs=40000,
    callbacks=[early_stopping],
    verbose=0,
)

# 予測用の格子点計算
reso = 50
x0s = np.linspace(x_range[0][0], x_range[0][1], reso)
x1s = np.linspace(x_range[1][0], x_range[1][1], reso)
xx0s, xx1s = np.meshgrid(x0s, x1s)
xxs_pred = np.c_[xx0s.reshape(-1), xx1s.reshape(-1)]

# 予測
ys_pred = model.predict(xxs_pred)
yys_pred = ys_pred.reshape(xx0s.shape)

# 学習曲線表示
print(f"train_MSE = {history.history['loss'][-1]:.3f}")
print(f"test_MSE = {history.history['val_loss'][-1]:.3f}")
print('The learning curve was output.\n')

plt.plot(history.history['loss'], label='train')
plt.plot(history.history['val_loss'], label='test')
plt.title('MSE')
plt.xlabel('epochs')
plt.legend()
plt.grid(axis='both')
plt.show()

# 結果表示
print('The data was output as a 2D graph.')
ax = show_data2d()
ax.plot_surface(
    xx0s,
    xx1s,
    yys_pred,
    cmap='Greens',
    alpha=0.2,
    rstride=5,
    cstride=5,
    edgecolor='k',
)
ax.view_init(elev=60, azim=30)
plt.show()