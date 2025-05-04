import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy.sparse import diags, kron, identity

# Parámetros
L = 1.0
Nx = 30
Ny = 30
dx = L / (Nx - 1)
dy = L / (Ny - 1)
dt = 0.001
Tmax = 2.0   # AUMENTAR TIEMPO TOTAL AQUÍ
Nt = int(Tmax / dt)
coef = 0.01  # Coeficiente del operador

# Malla
x = np.linspace(0, L, Nx)
y = np.linspace(0, L, Ny)
X, Y = np.meshgrid(x, y)

# Condición inicial
U = -50 * X * Y
V = np.zeros_like(U)  # T_t(x,y,0) = 0
u = U.flatten()
v = V.flatten()

# Coeficientes numéricos
rx = coef * dt / dx**2
ry = coef * dt / dy**2

# Matriz de derivadas en x (Neumann en x=0 y x=1)
Ax = diags([1, -2, 1], [-1, 0, 1], shape=(Nx, Nx)).tolil()
Ax[0, 1] = 2      # Neumann en x=0
Ax[-1, -2] = 2    # Neumann en x=1
Ax = Ax.tocsr()

# Matriz de derivadas en y (Dirichlet en y=0 y y=1)
Ay = diags([1, -2, 1], [-1, 0, 1], shape=(Ny, Ny)).tolil()
Ay[0, 0] = -2  # Dirichlet en y=0
Ay[-1, -1] = -2  # Dirichlet en y=1
Ay = Ay.tocsr()

# Laplaciano modificado: T_xx - T_yy
Lx = kron(identity(Ny), Ax)
Ly = kron(Ay, identity(Nx))
Laplaciano_mod = rx * Lx - ry * Ly

# Gráfica inicial
fig, ax = plt.subplots()
contour = ax.contourf(X, Y, U, levels=20, cmap='hot')
fig.colorbar(contour)

def aplicar_condiciones_de_frontera(u_vec):
    """Imponer T(x,0)=50, T(x,1)=10"""
    u_mat = u_vec.reshape((Ny, Nx))
    u_mat[0, :] = 50     # y = 0
    u_mat[-1, :] = 10    # y = 1
    return u_mat.flatten()

def update(frame):
    global u, v

    for _ in range(5):  # pasos por frame
        v += dt * (Laplaciano_mod @ u)
        u += dt * v
        u = aplicar_condiciones_de_frontera(u)

    ax.clear()
    T_plot = u.reshape((Ny, Nx))
    cont = ax.contourf(X, Y, T_plot, levels=20, cmap='hot')
    ax.set_title(f"Tiempo = {frame * 5 * dt:.3f} s")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    return cont.collections

ani = FuncAnimation(fig, update, frames=Nt // 5, repeat=False)
plt.show()
