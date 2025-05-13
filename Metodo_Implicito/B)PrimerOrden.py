import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import lil_matrix
from scipy.sparse.linalg import splu

# Parámetros del dominio y malla
L = 1.0
Nx, Ny = 50, 50
dx = L / (Nx - 1)
dy = L / (Ny - 1)
alpha = 0.01
dt = 0.01
tol = 1e-6  # tolerancia de convergencia
max_iter = 10000

# Índices lineales
def idx(i, j):
    return i * Ny + j

# Coeficientes
lamx = alpha * dt / dx**2
lamy = alpha * dt / dy**2
N = Nx * Ny

# Condiciones de frontera
T_bottom, T_top = 50.0, 10.0

# Matriz A para método implícito
A = lil_matrix((N, N))

for i in range(Nx):
    for j in range(Ny):
        k = idx(i, j)

        if j == 0 or j == Ny - 1:  # Dirichlet
            A[k, k] = 1.0
        elif i == 0:  # Neumann izquierda
            A[k, k] = 1 + lamx + 2*lamy
            A[k, idx(i+1, j)] = -lamx
            A[k, idx(i, j+1)] = -lamy
            A[k, idx(i, j-1)] = -lamy
        elif i == Nx - 1:  # Neumann derecha
            A[k, k] = 1 + lamx + 2*lamy
            A[k, idx(i-1, j)] = -lamx
            A[k, idx(i, j+1)] = -lamy
            A[k, idx(i, j-1)] = -lamy
        else:  # interior
            A[k, k] = 1 + 2*lamx + 2*lamy
            A[k, idx(i+1, j)] = -lamx
            A[k, idx(i-1, j)] = -lamx
            A[k, idx(i, j+1)] = -lamy
            A[k, idx(i, j-1)] = -lamy

A = A.tocsc()
lu = splu(A)

# Condición inicial: T(x, y, 0) = -50 * x * y
x = np.linspace(0, L, Nx)
y = np.linspace(0, L, Ny)
X, Y = np.meshgrid(x, y, indexing='ij')
T = (-50 * X * Y).reshape(N)

# Función para aplicar las condiciones de frontera Dirichlet
def aplicar_dirichlet(T_vec):
    for i in range(Nx):
        T_vec[idx(i, 0)] = T_bottom
        T_vec[idx(i, Ny-1)] = T_top
    return T_vec

T = aplicar_dirichlet(T)

# Iterar hasta que la solución se estabilice
for step in range(max_iter):
    T_old = T.copy()
    T = aplicar_dirichlet(T)
    T = lu.solve(T_old)
    diff = np.linalg.norm(T - T_old, ord=np.inf)
    if diff < tol:
        print(f"Convergencia alcanzada en {step} pasos.")
        break
else:
    print("No se alcanzó la convergencia.")

# Graficar solución estacionaria
T_mat = T.reshape(Nx, Ny)
plt.figure(figsize=(6,5))
cf = plt.contourf(X, Y, T_mat, levels=50, cmap='inferno')
plt.colorbar(cf, label="Temperatura (°C)")
plt.title("Solución estacionaria (Ecuación del Calor 2D)")
plt.xlabel("x")
plt.ylabel("y")
plt.tight_layout()
plt.show()
