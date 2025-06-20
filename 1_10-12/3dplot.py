from mpl_toolkits.mplot3d import Axes3D
import numpy as np

import matplotlib.pyplot as plt

# サンプルデータ（3点）
x1 = [1, 1, 2]
x2 = [1, 2, 1]
y  = [-1, 0, 1]

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

ax.scatter(x1, x2, y, c='r', marker='o')

ax.set_xlabel('x1')
ax.set_ylabel('x2')
ax.set_zlabel('y')
ax.set_title('3D Scatter Plot (3 points)')
# 平面の範囲を決める
x1_range = np.linspace(min(x1), max(x1), 10)
x2_range = np.linspace(min(x2), max(x2), 10)
X1, X2 = np.meshgrid(x1_range, x2_range)
a1, a2, b = 2, 1, -4
Y = a1 * X1 + a2 * X2 + b

# 平面を描画
ax.plot_surface(X1, X2, Y, alpha=0.5, color='blue')
plt.show()


# サンプルデータ（4点）
x1 = [1, 1, 2, 2]
x2 = [1, 2, 1, 2]
y  = [-1, 0, 1, 1]

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

ax.scatter(x1, x2, y, c='r', marker='o')

ax.set_xlabel('x1')
ax.set_ylabel('x2')
ax.set_zlabel('y')
ax.set_title('3D Scatter Plot (4 points)')
# 平面の範囲を決める
x1_range = np.linspace(min(x1), max(x1), 10)
x2_range = np.linspace(min(x2), max(x2), 10)
X1, X2 = np.meshgrid(x1_range, x2_range)
a1, a2, b = 1.5, 0.5, -2.75
Y = a1 * X1 + a2 * X2 + b

# 平面を描画
ax.plot_surface(X1, X2, Y, alpha=0.5, color='blue')
plt.show()