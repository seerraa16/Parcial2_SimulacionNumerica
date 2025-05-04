# calor_2d_implicito_sparse.py
import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import diags, kron, identity, linalg

# Parámetros del problema
L = 1.0
N = 20
alpha = 0.01
dx = L / (N + 1)
dt = 0.001
T_total = 0.1
steps = int(T_total / dt)

# Malla espacial
x = np.linspace(dx, L - dx, N)
y = np.linspace(dx, L - dx, N)
X, Y = np.meshgrid(x, y)

# Condición inicial
T0 = np.sin(np.pi * X / L) * np.sin(np.pi * Y / L)
T = T0.flatten()

# Método implícito: matriz sparse
r = alpha * dt / dx**2

# Matriz en x con Dirichlet
Ax = diags([(1 + 2 * r) * np.ones(N), -r * np.ones(N - 1), -r * np.ones(N - 1)], [0, -1, 1])

# Matriz en y con Neumann
Ay = diags([(1 + 2 * r) * np.ones(N), -r * np.ones(N - 1), -r * np.ones(N - 1)], [0, -1, 1])
Ay = Ay.tolil()
Ay[0, 1] = -2 * r
Ay[-1, -2] = -2 * r
Ay = Ay.tocsr()

# Matriz del sistema
I = identity(N)
A = kron(I, Ax) + kron(Ay, I)

# Solución temporal
for step in range(steps):
    T = linalg.spsolve(A, T)

# Mostrar resultado
T_final = T.reshape((N, N))

plt.figure(figsize=(6, 5))
plt.contourf(X, Y, T_final, 20, cmap='hot')
plt.colorbar(label='Temperatura')
plt.title("Solución 2D transitoria - Método Implícito + Sparse")
plt.xlabel("x")
plt.ylabel("y")
plt.show()
