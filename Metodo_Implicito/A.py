import numpy as np
import matplotlib.pyplot as plt

# Tamaño de la malla incluyendo bordes
Nx, Ny = 6, 6  # puedes cambiar estos valores

# Inicializar la matriz de temperaturas
T = np.zeros((Nx, Ny))

# Aplicar condiciones de contorno
def apply_boundary_conditions(T):
    T[0, :] = 50                 # Dirichlet en y = 0 (parte inferior)
    T[-1, :] = 10                # Dirichlet en y = 1 (parte superior)
    T[:, 0] = T[:, 1]            # Neumann en x = 0 (lado izquierdo)
    T[:, -1] = T[:, -2]          # Neumann en x = 1 (lado derecho)
    return T

T = apply_boundary_conditions(T)

# Marcar los puntos interiores con NaN (no conocidos aún)
for i in range(1, Nx - 1):
    for j in range(1, Ny - 1):
        T[i, j] = np.nan

# Graficar la malla
fig, ax = plt.subplots()
im = ax.imshow(np.nan_to_num(T, nan=0), cmap='coolwarm', origin='lower')  # origin='lower' para que y=0 esté abajo

# Etiquetar cada celda
for i in range(Nx):
    for j in range(Ny):
        if np.isnan(T[i, j]):
            label = '?'
        else:
            label = str(int(T[i, j]))
        ax.text(j, i, label, ha='center', va='center', color='black', fontsize=12)

# Configurar ejes y estética
ax.set_xticks(np.arange(Ny))
ax.set_yticks(np.arange(Nx))
ax.set_xticklabels([])
ax.set_yticklabels([])
ax.set_title('Condiciones de Contorno: 50°C abajo, 10°C arriba')
plt.grid(True, which='both', color='gray', linewidth=0.5)
plt.colorbar(im, label='Temperatura')
plt.show()
