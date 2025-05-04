import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy.sparse import diags, kron, identity, linalg

# Parámetros del dominio
Lx, Ly = 50.0, 100.0
Nx, Ny = 30, 60  # discretización en x y y
dx = Lx / (Nx + 1)
dy = Ly / (Ny + 1)
dt = 0.1
alpha = 0.00052
T_total = 300.0
steps = int(T_total / dt)
frames_interval = int(5 / dt)

# Malla
x = np.linspace(dx, Lx - dx, Nx)
y = np.linspace(dy, Ly - dy, Ny)
X, Y = np.meshgrid(x, y)

# Condición inicial: T(x,y,0) = 0
T = np.zeros((Ny, Nx)).flatten()

# Coeficientes para diferencias finitas
rx = alpha * dt / dx**2
ry = alpha * dt / dy**2

# Matriz en x (Dirichlet en x=0 y x=Lx)
Ax = diags([-rx, 1 + 2*rx, -rx], [-1, 0, 1], shape=(Nx, Nx)).tolil()
Ax[0, 0] = 1
Ax[0, 1] = 0
Ax[-1, -1] = 1
Ax[-1, -2] = 0
Ax = Ax.tocsr()

# Matriz en y (Neumann en y=0 y y=Ly)
Ay = diags([-ry, 1 + 2*ry, -ry], [-1, 0, 1], shape=(Ny, Ny)).tolil()
Ay[0, 1] = -2 * ry
Ay[-1, -2] = -2 * ry
Ay = Ay.tocsr()

# Matriz global A
Ix = identity(Nx)
Iy = identity(Ny)
A = kron(Iy, Ax) + kron(Ay, Ix)

# Gráfica
fig, ax = plt.subplots()
contour = ax.contourf(X, Y, T.reshape((Ny, Nx)), levels=20, cmap='hot')
fig.colorbar(contour)

def aplicar_condiciones_frontera(T_vec):
    """Aplica T=100 en x=0 y x=Lx (bordes izquierdo y derecho)"""
    T_mat = T_vec.reshape((Ny, Nx))
    T_mat[:, 0] = 100   # x = 0
    T_mat[:, -1] = 100  # x = Lx
    return T_mat.flatten()

def update(frame):
    global T
    for _ in range(frames_interval):  # avanza 5s por frame
        T = linalg.spsolve(A, T)
        T = aplicar_condiciones_frontera(T)
    ax.clear()
    T_plot = T.reshape((Ny, Nx))
    cont = ax.contourf(X, Y, T_plot, levels=20, cmap='hot')
    ax.set_title(f"t = {frame * 5:.1f} s")
    ax.set_xlabel('x [m]')
    ax.set_ylabel('y [m]')
    ax.set_xlim(0, 50)
    ax.set_ylim(0, 100)
    return cont.collections

ani = FuncAnimation(fig, update, frames=int(T_total / 5), repeat=False)
plt.show()
