import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import diags
from scipy.sparse.linalg import spsolve

# Parámetros
Nx, Ny = 20, 20
L = 1.0
dx = L / (Nx - 1)
dy = L / (Ny - 1)
alpha = 0.01
dt = 0.01
Nt = 100

N = Nx * Ny
r = alpha * dt / dx**2

# Inicial
T = np.zeros((Nx, Ny))
x = np.linspace(0, L, Nx)
y = np.linspace(0, L, Ny)
X, Y = np.meshgrid(x, y)
T = np.sin(np.pi * X) * np.sin(np.pi * Y)
T = T.flatten()

# Matriz coeficientes para Laplaciano 2D
main_diag = (1 + 4*r) * np.ones(N)
side_diag = -r * np.ones(N-1)
up_down_diag = -r * np.ones(N-Nx)

for i in range(1, Nx):
    side_diag[i * Ny - 1] = 0

diagonals = [main_diag, side_diag, side_diag, up_down_diag, up_down_diag]
offsets = [0, -1, 1, -Nx, Nx]
A = diags(diagonals, offsets, shape=(N, N)).tocsc()

# Tiempo
for n in range(Nt):
    T = spsolve(A, T)

T = T.reshape((Nx, Ny))
plt.contourf(X, Y, T, cmap='hot')
plt.title("Temperatura - Método Implícito Transitorio")
plt.colorbar()
plt.show()
