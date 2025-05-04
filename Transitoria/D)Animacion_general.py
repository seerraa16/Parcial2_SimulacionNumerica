# calor_2d_animacion.py
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy.sparse import diags, kron, identity, linalg

# Parámetros
L = 1.0
N = 20
alpha = 0.01
dx = L / (N + 1)
dt = 0.001
T_total = 0.1
steps = int(T_total / dt)

x = np.linspace(dx, L - dx, N)
y = np.linspace(dx, L - dx, N)
X, Y = np.meshgrid(x, y)

# Condición inicial
T0 = 100*np.sin(np.pi * X ) * np.sin(np.pi * Y)
T = T0.flatten()

r = alpha * dt / dx**2

# Matrices
Ax = diags([(1 + 2 * r) * np.ones(N), -r * np.ones(N - 1), -r * np.ones(N - 1)], [0, -1, 1])
Ay = diags([(1 + 2 * r) * np.ones(N), -r * np.ones(N - 1), -r * np.ones(N - 1)], [0, -1, 1])
Ay = Ay.tolil()
Ay[0, 1] = -2 * r
Ay[-1, -2] = -2 * r
Ay = Ay.tocsr()

I = identity(N)
A = kron(I, Ax) + kron(Ay, I)

# Preparar figura
fig, ax = plt.subplots()
contour = ax.contourf(X, Y, T0, 20, cmap='hot')
fig.colorbar(contour)

def update(frame):
    global T
    for _ in range(5):  # Avanza varios pasos por frame para que sea más rápido
        T = linalg.spsolve(A, T)
    ax.clear()
    T_plot = T.reshape((N, N))
    cont = ax.contourf(X, Y, T_plot, 20, cmap='hot')
    ax.set_title(f"Tiempo = {frame * 5 * dt:.3f} s")
    return cont.collections

ani = FuncAnimation(fig, update, frames=steps // 5, repeat=False)
plt.show()
