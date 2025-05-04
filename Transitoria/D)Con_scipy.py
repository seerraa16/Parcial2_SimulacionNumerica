import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy.sparse import diags, kron, eye, csc_matrix
from scipy.sparse.linalg import spsolve

# Parámetros
L = 1.0
Nx = 30
Ny = 30
dx = L / (Nx - 1)
dy = L / (Ny - 1)
dt = 0.001
Tmax = 0.5
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

# Función para construir matriz Laplaciana sparse
def laplace_sparse(N, h, neumann=False):
    main = -2 * np.ones(N)
    off = np.ones(N - 1)
    diagonals = [main, off, off]
    offsets = [0, -1, 1]
    M = diags(diagonals, offsets, format='lil')
    
    if neumann:
        M[0, 0] = -2
        M[0, 1] = 2
        M[-1, -2] = 2
        M[-1, -1] = -2

    return M.tocsc() / h**2

# Construcción sparse de derivadas
Dx = laplace_sparse(Nx, dx, neumann=True)
Dy = laplace_sparse(Ny, dy, neumann=False)

Ix = eye(Nx, format='csc')
Iy = eye(Ny, format='csc')

Ax = kron(Iy, Dx, format='csc')
Ay = kron(Dy, Ix, format='csc')

A = eye(Nx * Ny, format='csc') - dt**2 * c2 * (Ax + Ay)

# Aplicar condiciones de frontera Dirichlet
def apply_dirichlet(T_vec):
    T_mat = T_vec.reshape((Ny, Nx))
    T_mat[0, :] = 50
    T_mat[-1, :] = 10
    return T_mat.flatten()

# Inicializar figura
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
    Tnp1 = spsolve(A, rhs)
    Tnp1 = apply_dirichlet(Tnp1)

    Tn1 = Tn
    Tn = Tnp1

    T_mat = Tnp1.reshape(Ny, Nx)
    im.set_array(T_mat)
    title.set_text(f"Tiempo = {frame * dt:.3f} s")
    return im, title

# Crear la animación
anim = FuncAnimation(fig, update, frames=range(0, Nt, 5), interval=100)
plt.show()
