import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy.sparse import diags, kron, identity

# Parámetros del problema
L = 1.0
Nx = 30
Ny = 30
dx = L / (Nx - 1)
dy = L / (Ny - 1)
dt = 0.001
Tmax = 2
Nt = int(Tmax / dt)
coef = 0.01  # Coeficiente de difusión

# Malla
x = np.linspace(0, L, Nx)
y = np.linspace(0, L, Ny)
X, Y = np.meshgrid(x, y)

# Condición inicial
TO = 100 * np.sin(np.pi * X) * np.sin(np.pi * Y)  # T(x,y,0)
Tn = TO.flatten()
Tn1 = Tn.copy()
V = np.zeros_like(TO)  # T_t(x,y,0) = 0
v = V.flatten()

# Coeficientes numéricos
rx = coef * dt / dx**2
ry = coef * dt / dy**2

# Matriz de derivadas en x (Neumann en x=0 y x=1)
Ax = diags([1, -2, 1], [-1, 0, 1], shape=(Nx, Nx)).tolil()
Ax[0, 1] = 2      # Neumann en x=0
Ax[-1, -2] = 2    # Neumann en x=1
Ax = Ax.tocsr()

# Matriz de derivadas en y (Dirichlet en y=0, y=1)
Ay = diags([1, -2, 1], [-1, 0, 1], shape=(Ny, Ny)).tolil()
Ay[0, 0] = -2     # Dirichlet en y=0
Ay[-1, -1] = -2   # Dirichlet en y=1
Ay = Ay.tocsr()

# Laplaciano modificado: T_xx - T_yy
Lx = kron(identity(Ny), Ax)
Ly = kron(Ay, identity(Nx))
Laplaciano_mod = rx * Lx - ry * Ly

# Gráfica
fig, ax = plt.subplots()
pcm = ax.pcolormesh(X, Y, TO, shading='auto', cmap='plasma', vmin=-20, vmax=120)
fig.colorbar(pcm, ax=ax)

def aplicar_condiciones_de_frontera(Tvec):
    """Imponer T(x,0) = -20, T(x,1) = 120"""
    Tmat = Tvec.reshape((Ny, Nx))
    Tmat[0, :] = -20     # y = 0
    Tmat[-1, :] = 120    # y = 1
    return Tmat.flatten()

def update(frame):
    global Tn, Tn1, v

    for _ in range(5):
        v += dt * (Laplaciano_mod @ Tn)
        Tn1 = Tn + dt * v
        Tn1 = aplicar_condiciones_de_frontera(Tn1)
        Tn = Tn1.copy()

    ax.clear()
    T_plot = Tn.reshape((Ny, Nx))
    pcm = ax.pcolormesh(X, Y, T_plot, shading='auto', cmap='plasma', vmin=-20, vmax=120)
    ax.set_title(f"Tiempo = {frame * 5 * dt:.3f} s")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    return pcm,

ani = FuncAnimation(fig, update, frames=Nt // 5, repeat=False)
plt.show()
