import numpy as np
import matplotlib.pyplot as plt

np.set_printoptions(precision=3, suppress=True)
config = {
    "font.size":14,
    "figure.figsize":(7, 4)
}
plt.rcParams.update(config)

n = 30
noise = 5
np.random.seed(1)

xs = np.random.rand(n) * 30
ts = (xs * 0.1) ** 4 + xs * 1 + 50 + np.random.rand(n) * noise
x_range = (0, 30)
t_range = (30, 140)

print('xs (temperature)', xs)
print('')
print('ts (sales)', ts)


