import numpy as np
import matplotlib.pyplot as plt

# Parámetros
L = 1.0
Nx, Ny = 50, 50
dx = L / (Nx - 1)
dy = L / (Ny - 1)
alpha = 0.01
dt = 0.0005
Nt = 1000

x = np.linspace(0, L, Nx)
y = np.linspace(0, L, Ny)
X, Y = np.meshgrid(x, y)

T = np.sin(np.pi * X) * np.sin(np.pi * Y)

r_x = alpha * dt / dx**2
r_y = alpha * dt / dy**2

for n in range(Nt):
    Tn = T.copy()
    for i in range(1, Nx-1):
        for j in range(1, Ny-1):
            T[i, j] = Tn[i, j] + r_x * (Tn[i+1, j] - 2*Tn[i, j] + Tn[i-1, j]) + \
                                  r_y * (Tn[i, j+1] - 2*Tn[i, j] + Tn[i, j-1])
    T[0, :] = 0
    T[-1, :] = 0
    T[:, 0] = T[:, 1]
    T[:, -1] = T[:, -2]

plt.contourf(X, Y, T, cmap='hot')
plt.title("Temperatura - Método Explícito Transitorio")
plt.colorbar()
plt.show()
