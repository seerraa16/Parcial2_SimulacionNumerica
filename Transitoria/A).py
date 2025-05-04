import numpy as np
import matplotlib.pyplot as plt

# Parámetros
L = 1.0
Nx = Ny = 20
dx = dy = L / Nx
alpha = 1.0
dt = 0.0005
r = alpha * dt / dx**2
Nt = 200  # número de pasos de tiempo

# Mallas
x = np.linspace(0, L, Nx+1)
y = np.linspace(0, L, Ny+1)
X, Y = np.meshgrid(x, y, indexing='ij')

# Condición inicial
T = np.sin(np.pi * X) * np.sin(np.pi * Y)

# Bucle en el tiempo
for n in range(Nt):
    T_new = T.copy()
    
    # Esquema explícito
    for i in range(1, Nx):
        for j in range(1, Ny):
            T_new[i, j] = T[i, j] + r * (
                T[i+1, j] - 2*T[i, j] + T[i-1, j] +
                T[i, j+1] - 2*T[i, j] + T[i, j-1]
            )

    # Condiciones de frontera:
    # Dirichlet: T = 0 en x = 0 y x = L
    T_new[0, :] = 0
    T_new[Nx, :] = 0

    # Neumann (aislante): derivada normal = 0 en y = 0 y y = L
    T_new[:, 0] = T_new[:, 1]      # y = 0
    T_new[:, Ny] = T_new[:, Ny-1]  # y = L

    T = T_new.copy()

# Visualización final
plt.figure(figsize=(6,5))
plt.contourf(X, Y, T, 20, cmap='hot')
plt.colorbar(label='Temperatura')
plt.title('Distribución de temperatura (t = {:.3f})'.format(Nt*dt))
plt.xlabel('x')
plt.ylabel('y')
plt.show()
