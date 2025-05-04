import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy.sparse import diags, kron, identity, linalg

# Parámetros
L = 1.0
N = 30
dx = L / (N + 1)
dt = 0.001
T_total = 1.0
steps = int(T_total / dt)
coef = 0.01

# Malla
x = np.linspace(dx, L - dx, N)
y = np.linspace(dx, L - dx, N)
X, Y = np.meshgrid(x, y)

# Condición inicial
U = -50 * X * Y  # T(x,y,0)
V = np.zeros_like(U)  # T_t(x,y,0)

u = U.flatten()
v = V.flatten()

# Coeficiente para esquema implícito
r = coef * dt / dx**2

# Matrices en x (Neumann)
Ax = diags([1, -2, 1], [-1, 0, 1], shape=(N, N)).tolil()
Ax[0, 1] = 2
Ax[-1, -2] = 2
Ax = Ax.tocsr()

# Matrices en y (Dirichlet: impuestas en solución)
Ay = diags([1, -2, 1], [-1, 0, 1], shape=(N, N)).tolil()
Ay[0, 0] = -2  # Borde y=0 (se sobreescribirá con T=50)
Ay[-1, -1] = -2  # Borde y=1 (se sobreescribirá con T=10)
Ay = Ay.tocsr()

# Matriz del sistema para v_t = 0.01 (u_xx - u_yy)
Lx = kron(identity(N), Ax)
Ly = kron(Ay, identity(N))
Laplaciano_mod = Lx - Ly

# Figura
fig, ax = plt.subplots()
contour = ax.contourf(X, Y, U, levels=20, cmap='hot')
fig.colorbar(contour)

def aplicar_condiciones_de_frontera(u_vec):
    """Imponer T(x,0)=50 y T(x,1)=10"""
    u_mat = u_vec.reshape((N, N))
    u_mat[:, 0] = 50  # y=0
    u_mat[:, -1] = 10  # y=1
    return u_mat.flatten()

def update(frame):
    global u, v

    for _ in range(5):  # avanzar varios pasos por frame
        v = v + dt * (Laplaciano_mod @ u)
        u = u + dt * v
        u = aplicar_condiciones_de_frontera(u)

    ax.clear()
    U_plot = u.reshape((N, N))
    cont = ax.contourf(X, Y, U_plot, levels=20, cmap='hot')
    ax.set_title(f"Tiempo = {frame * 5 * dt:.3f} s")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    return cont.collections

ani = FuncAnimation(fig, update, frames=steps // 5, repeat=False)
plt.show()
