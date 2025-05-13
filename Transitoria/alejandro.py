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
Tmax = 1.0
Nt = int(Tmax / dt)
alpha = 0.01  # coeficiente de difusión

# Malla
x = np.linspace(0, L, Nx)
y = np.linspace(0, L, Ny)
X, Y = np.meshgrid(x, y)

# Condición inicial
T0 = 10 * np.exp(X + Y)
Tn = T0.flatten()

# Matrices de derivadas segunda orden
def laplace_dense(N, h, neumann_start=False, dirichlet_end=False):
    M = np.zeros((N, N))
    for i in range(N):
        M[i, i] = -2
        if i > 0:
            M[i, i - 1] = 1
        if i < N - 1:
            M[i, i + 1] = 1
    if neumann_start:
        M[0, 0] = -2
        M[0, 1] = 2
    if dirichlet_end:
        M[-1, -1] = 1  # Mantener T fijo
        M[-1, -2] = 0
    return M / h**2

# Laplacianos en x y y con condiciones mixtas
Dx = laplace_dense(Nx, dx, neumann_start=True, dirichlet_end=True)
Dy = laplace_dense(Ny, dy, neumann_start=True, dirichlet_end=True)

# Operadores 2D
Ix = np.eye(Nx)
Iy = np.eye(Ny)
Ax = np.kron(Iy, Dx)
Ay = np.kron(Dy, Ix)

# Matriz del sistema
Ibig = np.eye(Nx * Ny)
A = Ibig - dt * alpha * (Ax + Ay)

# Función para aplicar condiciones de frontera
def apply_boundary_conditions(T_vec):
    T_mat = T_vec.reshape((Ny, Nx))

    # Condición de Neumann en x=0: dT/dx = 0
    T_mat[:, 0] = T_mat[:, 1]
    # Dirichlet en x=1: T = -20
    T_mat[:, -1] = -20

    # Neumann en y=0: dT/dy = 0
    T_mat[0, :] = T_mat[1, :]
    # Dirichlet en y=1: T = 120
    T_mat[-1, :] = 120

    return T_mat.flatten()

# Gráfica
fig, ax = plt.subplots()
im = ax.imshow(T0, cmap='inferno', origin='lower',
               extent=[0, L, 0, L], vmin=-20, vmax=120)
cbar = plt.colorbar(im, ax=ax, label='Temperatura (°C)')
title = ax.set_title("Tiempo = 0.000 s")
ax.set_xlabel('x')
ax.set_ylabel('y')

# Animación
def update(frame):
    global Tn
    rhs = apply_boundary_conditions(Tn)
    Tn1 = np.linalg.solve(A, rhs)
    Tn1 = apply_boundary_conditions(Tn1)
    Tn = Tn1
    im.set_data(Tn.reshape((Ny, Nx)))
    title.set_text(f"Tiempo = {frame*dt:.3f} s")
    return im, title

anim = FuncAnimation(fig, update, frames=range(0, Nt, 30), interval=1)
plt.show()
