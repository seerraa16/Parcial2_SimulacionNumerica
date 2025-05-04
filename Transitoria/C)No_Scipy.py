import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Parámetros
L = 1.0
Nx = 30
Ny = 30
dx = L / (Nx - 1)
dy = L / (Ny - 1)
dt = 0.001
Tmax = 10
Nt = int(Tmax / dt)
c2 = 0.01

# Mallado
x = np.linspace(0, L, Nx)
y = np.linspace(0, L, Ny)
X, Y = np.meshgrid(x, y)

# Condición inicial
T0 = -50 * X * Y
Tn = T0.flatten()
Tn1 = Tn.copy()

# Función para construir matriz Laplaciana densa
def laplace_dense(N, h, neumann=False):
    M = np.zeros((N, N))
    for i in range(N):
        M[i, i] = -2
        if i > 0:
            M[i, i - 1] = 1
        if i < N - 1:
            M[i, i + 1] = 1

    if neumann:
        M[0, 0] = -2
        M[0, 1] = 2
        M[-1, -2] = 2
        M[-1, -1] = -2

    return M / h**2

# Derivadas en x e y
Dx = laplace_dense(Nx, dx, neumann=True)
Dy = laplace_dense(Ny, dy, neumann=False)

# Producto de Kronecker (manual usando np.kron)
Ix = np.eye(Nx)
Iy = np.eye(Ny)

Ax = np.kron(Iy, Dx)
Ay = np.kron(Dy, Ix)

# Matriz A total
I = np.eye(Nx * Ny)
A = I - dt**2 * c2 * (Ax + Ay)

# Condiciones de frontera Dirichlet (superior e inferior)
def apply_dirichlet(T_vec):
    T_mat = T_vec.reshape((Ny, Nx))
    T_mat[0, :] = 50
    T_mat[-1, :] = 10
    return T_mat.flatten()

# Visualización
fig, ax = plt.subplots()
T_mat = Tn.reshape(Ny, Nx)
im = ax.imshow(T_mat, cmap='inferno', origin='lower', extent=[0, L, 0, L], vmin=-50, vmax=50)
cbar = plt.colorbar(im)
title = ax.set_title("Tiempo = 0.000 s")

# Función de actualización
def update(frame):
    global Tn, Tn1

    rhs = 2 * Tn - Tn1
    rhs = apply_dirichlet(rhs)

    # Resolver sistema A x = rhs (con np.linalg.solve, ya que A es densa)
    Tnp1 = np.linalg.solve(A, rhs)
    Tnp1 = apply_dirichlet(Tnp1)

    Tn1 = Tn
    Tn = Tnp1

    T_mat = Tnp1.reshape(Ny, Nx)
    im.set_array(T_mat)
    title.set_text(f"Tiempo = {frame * dt:.3f} s")
    return im, title

# Crear animación
anim = FuncAnimation(fig, update, frames=range(0, Nt, 5), interval=100)
plt.show()
