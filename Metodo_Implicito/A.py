import numpy as np
import matplotlib.pyplot as plt

# Crear matriz 5x5 con valores interiores arbitrarios (-20 por ejemplo)
Nx = Ny = 5
T = np.full((Nx, Ny), -20.0)

# Función de condiciones de contorno
def apply_boundary_conditions(T):
    T[:, 0] = 50                 # Dirichlet en y = 0
    T[:, -1] = 10                # Dirichlet en y = 1
    T[0, :] = T[1, :]            # Neumann en x = 0
    T[-1, :] = T[-2, :]          # Neumann en x = 1
    return T

# Aplicar condiciones
T = apply_boundary_conditions(T)

# Crear la cuadrícula de coordenadas para el plot
x = np.arange(Nx)
y = np.arange(Ny)
X, Y = np.meshgrid(y, x)  # ¡Ojo! y primero para que coincida con la orientación

# Graficar con puntos
fig, ax = plt.subplots()
scatter = ax.scatter(X, Y, c=T, cmap='coolwarm', s=500, edgecolors='k')

# Añadir valores numéricos sobre cada punto
for i in range(Nx):
    for j in range(Ny):
        ax.text(j, i, f"{T[i,j]:.0f}", ha='center', va='center', color='black', fontsize=10)

# Ajustar la visualización
ax.set_xticks(np.arange(Ny))
ax.set_yticks(np.arange(Nx))
ax.set_xlim(-0.5, Ny - 0.5)
ax.set_ylim(Nx - 0.5, -0.5)
ax.set_title("Temperatura en malla 5x5 con condiciones de contorno")
plt.colorbar(scatter, label="Temperatura")
plt.grid(True)
plt.gca().set_aspect('equal', adjustable='box')
plt.show()
