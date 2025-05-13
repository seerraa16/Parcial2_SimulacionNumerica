import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Parámetros físicos y numéricos
alpha = 0.01
Nx = Ny = 40
L = 1.0
dt = 0.01
T_max = 10.0
Nt = int(T_max / dt)

dx = L / (Nx - 1)
x = np.linspace(0, L, Nx)
y = np.linspace(0, L, Ny)
X, Y = np.meshgrid(x, y, indexing='ij')

lambda_ = alpha * dt / dx**2

# Condición inicial
T = -50 * X * Y
frames = []

# Construcción de la matriz A (forma densa)
N = (Nx - 2) * (Ny - 2)
A = np.eye(N) * (1 + 4 * lambda_)

def idx(i, j):
    return (i - 1) * (Ny - 2) + (j - 1)

for i in range(1, Nx - 1):
    for j in range(1, Ny - 1):
        k = idx(i, j)
        if i > 1:
            A[k, idx(i - 1, j)] = -lambda_
        if i < Nx - 2:
            A[k, idx(i + 1, j)] = -lambda_
        if j > 1:
            A[k, idx(i, j - 1)] = -lambda_
        if j < Ny - 2:
            A[k, idx(i, j + 1)] = -lambda_

# Función para aplicar condiciones de contorno
def apply_boundary_conditions(T):
    T[:, 0] = 50                 # Dirichlet en y = 0
    T[:, -1] = 10                # Dirichlet en y = 1
    T[0, :] = T[1, :]            # Neumann en x = 0
    T[-1, :] = T[-2, :]          # Neumann en x = 1
    return T

# Simulación
save_every = int(0.1 / dt)
for n in range(Nt + 1):
    T = apply_boundary_conditions(T)
    b = np.zeros(N)

    for i in range(1, Nx - 1):
        for j in range(1, Ny - 1):
            k = idx(i, j)
            b[k] = T[i, j]
            if j == 1:
                b[k] += lambda_ * T[i, j - 1]
            if j == Ny - 2:
                b[k] += lambda_ * T[i, j + 1]
            if i == 1:
                b[k] += lambda_ * T[i - 1, j]
            if i == Nx - 2:
                b[k] += lambda_ * T[i + 1, j]

    T_new = np.linalg.solve(A, b).reshape((Nx - 2, Ny - 2))
    T[1:-1, 1:-1] = T_new

    if n % save_every == 0:
        frames.append(T.copy())

# Animación — redibujamos en cada frame
fig, ax = plt.subplots()
cbar = None

def animate(i):
    global cbar
    ax.clear()
    pcm = ax.pcolormesh(X, Y, frames[i], cmap='inferno', shading='auto', vmin=-50, vmax=50)
    ax.set_title(f"Tiempo t = {i * save_every * dt:.1f} s")
    if cbar is None:
        cbar = fig.colorbar(pcm, ax=ax)
    return pcm,

ani = FuncAnimation(fig, animate, frames=len(frames), interval=100)
plt.show()


from matplotlib.animation import PillowWriter

# Guardar la animación como GIF
ani.save("difusion_animacion.gif", writer=PillowWriter(fps=10))