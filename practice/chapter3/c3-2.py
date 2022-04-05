import numpy as np
import matplotlib.pyplot as plt
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense
from tensorflow.python.keras.callbacks import EarlyStopping

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
n_train = 30
noise = 10
np.random.seed(5)

xs = np.random.rand(n) * 30
ts = (xs * 0.1) ** 4 + 1 * xs + 50 + np.random.randn(n) * noise
xs_train = xs[:n_train].reshape(-1, 1)
ts_train = ts[:n_train].reshape(-1, 1)
xs_test = xs[n_train:].reshape(-1, 1)
ts_test = ts[n_train:].reshape(-1, 1)

x_range = (0, 30)
t_range = (0, 250)

n_batch = len(xs)

# Early Stopping
early_stopping = EarlyStopping(
    monitor='val_loss',
    patience=10,
)

# モデル生成
model = Sequential([
    Dense(32, activation='sigmoid', input_dim=1),
    # Dense(32, activation='relu', input_dim=1),
    Dense(1, activation='linear')
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

# 予測
xs_pred = np.linspace(x_range[0], x_range[1], 100)
ys_pred = model.predict(xs_pred.reshape(-1, 1))

# 学習曲線表示
print(f"train_MSE = {history.history['loss'][-1]:.3f}")
print(f"test_MSE = {history.history['val_loss'][-1]:.3f}")
print(f"min test_MSE : {np.argmin(history.history['val_loss']):d} epochs")
print('The learning curve was output.\n')

plt.plot(history.history['loss'], label='train')
plt.plot(history.history['val_loss'], label='test')
plt.title('MSE')
plt.xlabel('epochs')
plt.legend()
plt.grid(axis='both')
plt.show()

# グラフ表示
print('The data and a neural network model for it were output.\n')
plt.scatter(xs_train, ts_train, label='train')
plt.scatter(xs_test, ts_test, label='test')
plt.plot(xs_pred, ys_pred, 'green', label='neural network model')

plt_settings()
plt.show()
