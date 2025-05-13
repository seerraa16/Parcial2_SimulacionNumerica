import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import lil_matrix
from scipy.sparse.linalg import splu

# Parámetros del dominio y malla
L      = 1.0
Nx, Ny = 50, 50
dx     = L / (Nx - 1)
dy     = L / (Ny - 1)

# Condiciones de contorno
T_bottom, T_top = 50.0, 10.0  # Temperaturas en la frontera

# Constantes para la Laplaciana
lamx = 1.0 / dx**2
lamy = 1.0 / dy**2

# Construcción de la matriz dispersa A para Laplace estacionario
N = Nx * Ny
A = lil_matrix((N, N))

# Función para convertir coordenadas (i, j) en un índice 1D
def idx(i, j):
    return i * Ny + j

# Construcción de la matriz A
for i in range(Nx):
    for j in range(Ny):
        k = idx(i, j)
        if j == 0:  # y=0: Dirichlet inferior
            A[k, k] = 1.0
        elif j == Ny - 1:  # y=L: Dirichlet superior
            A[k, k] = 1.0
        elif i == 0:  # x=0: Neumann izquierda
            A[k, k] = 1.0
            A[k, idx(i+1, j)] = -1.0
        elif i == Nx - 1:  # x=L: Neumann derecha
            A[k, k] = 1.0
            A[k, idx(i-1, j)] = -1.0
        else:
            # Interior: 5 puntos (coeficientes de Laplace)
            A[k, k]               = 2 * (lamx + lamy)
            A[k, idx(i+1, j)]     = -lamx
            A[k, idx(i-1, j)]     = -lamx
            A[k, idx(i, j+1)]     = -lamy
            A[k, idx(i, j-1)]     = -lamy

# Factorización LU
A = A.tocsc()  # Convertir la matriz a formato CSC para mayor eficiencia
lu = splu(A)  # Factorización LU dispersa

# Vector b con condiciones de frontera
b = np.zeros(N)
for i in range(Nx):
    for j in range(Ny):
        k = idx(i, j)
        if j == 0:
            b[k] = T_bottom  # Condición de Dirichlet en y=0
        elif j == Ny - 1:
            b[k] = T_top     # Condición de Dirichlet en y=L

# Resolver el sistema A * T = b usando la factorización LU
T_vec = lu.solve(b)
T = T_vec.reshape((Nx, Ny))  # Reshape al formato 2D de temperatura

# Gráfica
x = np.linspace(0, L, Nx)
y = np.linspace(0, L, Ny)
X, Y = np.meshgrid(x, y, indexing='ij')

plt.figure(figsize=(6,5))
cf = plt.contourf(X, Y, T, levels=50, cmap='inferno')
plt.colorbar(cf, label="Temperatura (°C)")
plt.title("Solución estacionaria (Laplace 2D)")
plt.xlabel("x")
plt.ylabel("y")
plt.tight_layout()
plt.show()
