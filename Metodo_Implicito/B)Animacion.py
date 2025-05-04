import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Parámetros
L = 1.0            # Longitud del dominio en x y y
Nx = 50            # Número de puntos en x
Ny = 50            # Número de puntos en y
dx = L / (Nx - 1)  # Paso en x
dy = L / (Ny - 1)  # Paso en y
dt = 0.01          # Paso temporal

# Condiciones de frontera
T_bottom = 50      # Temperatura en y=0
T_top = 10         # Temperatura en y=1
T_left = 0         # Derivada de T en x=0 (condición de Neumann)
T_right = 0        # Derivada de T en x=1 (condición de Neumann)

# Condiciones iniciales
T_initial = lambda x, y: -50 * x * y  # Condición inicial T(x, y, 0)

# Malla de puntos (espacio)
x = np.linspace(0, L, Nx)
y = np.linspace(0, L, Ny)
T = np.zeros((Nx, Ny))  # Inicializamos la malla de temperaturas con ceros

# Aplicamos la condición inicial en la malla
for i in range(Nx):
    for j in range(Ny):
        T[i, j] = T_initial(x[i], y[j])

# Función para aplicar el método implícito (Gauss-Seidel)
def gauss_seidel(T, dx, dy, dt, tol=1e-6, max_iter=10000):
    error = np.inf
    iter_count = 0
    T_new = T.copy()  # Usamos T_new para evitar la sobrescritura en la misma iteración
    
    while error > tol and iter_count < max_iter:
        T_old = T_new.copy()  # Guardamos el estado anterior para calcular el error
        
        # Iteramos sobre el dominio interior (sin las fronteras)
        for i in range(1, Nx-1):
            for j in range(1, Ny-1):
                T_new[i, j] = (0.01 * (T_new[i+1, j] + T_new[i-1, j]) / dx**2 +
                               0.01 * (T_new[i, j+1] + T_new[i, j-1]) / dy**2) / \
                              (2 * 0.01 * (1/dx**2 + 1/dy**2))

        # Aplicamos las condiciones de frontera
        T_new[0, :] = T_left  # x = 0 (derivada 0)
        T_new[Nx-1, :] = T_right  # x = 1 (derivada 0)
        T_new[:, 0] = T_bottom  # y = 0
        T_new[:, Ny-1] = T_top  # y = 1
        
        # Calculamos el error (la diferencia con la iteración anterior)
        error = np.max(np.abs(T_new - T_old))
        iter_count += 1

    return T_new, iter_count

# Función para la animación
def animate(i):
    global T  # Usamos la variable T global para actualizar su valor
    T, _ = gauss_seidel(T, dx, dy, dt)  # Actualizamos la temperatura en cada paso
    cont.set_array(T.T)  # Actualizamos los datos de la imagen
    return cont,

# Crear la figura y el eje para la animación
fig, ax = plt.subplots()
cont = ax.contourf(x, y, T.T, 50, cmap='inferno')
fig.colorbar(cont, ax=ax, label="Temperatura (°C)")
ax.set_title("Distribución de temperatura en función del tiempo")
ax.set_xlabel("x")
ax.set_ylabel("y")

# Crear la animación
ani = FuncAnimation(fig, animate, frames=200, interval=50, blit=False)

# Mostrar la animación
plt.show()
