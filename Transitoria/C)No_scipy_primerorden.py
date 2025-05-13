import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Parámetros
L = 1.0
Nx = 20
Ny = 20
dx = L / (Nx - 1)
dy = L / (Ny - 1)
dt = 0.1
Tmax = 900
Nt = int(Tmax / dt)
alpha = 0.01  # coeficiente de difusión

# Malla
x = np.linspace(0, L, Nx)
y = np.linspace(0, L, Ny)
X, Y = np.meshgrid(x, y)

# Condición inicial
T0 = -50 * X * Y
Tn = T0.flatten()

# Construcción de las matrices de segunda derivada (Laplaciano) densas
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

# Kronecker para Laplaciano 2D
Ix = np.eye(Nx)
Iy = np.eye(Ny)
Ax = np.kron(Iy, Dx)
Ay = np.kron(Dy, Ix)

# Matriz implícita A = I - dt*alpha*(Ax+Ay)
Ibig = np.eye(Nx * Ny)
A = Ibig - dt * alpha * (Ax + Ay)

# Función para imponer contornos Dirichlet
def apply_dirichlet(T_vec):
    T_mat = T_vec.reshape((Ny, Nx))
    T_mat[0, :] = 50  # y=0
    T_mat[-1, :] = 10  # y=L
    return T_mat.flatten()

# Preparar figura
fig, ax = plt.subplots()
T_mat = Tn.reshape((Ny, Nx))
im = ax.imshow(T_mat, cmap='inferno', origin='lower',
               extent=[0, L, 0, L], vmin=10, vmax=50)
cbar = plt.colorbar(im, ax=ax, label='Temperatura (°C)')
title = ax.set_title("Tiempo = 0.000 s")
ax.set_xlabel('x')
ax.set_ylabel('y')

# Animación
def update(frame):
    global Tn
    # RHS: conservar interior y forzar Dirichlet en fronteras
    rhs = apply_dirichlet(Tn)
    # Resolver sistema implícito
    Tn1 = np.linalg.solve(A, rhs)
    Tn1 = apply_dirichlet(Tn1)
    Tn = Tn1
    # Actualizar imagen
    im.set_data(Tn.reshape((Ny, Nx)))
    title.set_text(f"Tiempo = {frame*dt:.3f} s")
    return im, title

anim = FuncAnimation(fig, update, frames=range(0, Nt, 30), interval=1)
plt.show()