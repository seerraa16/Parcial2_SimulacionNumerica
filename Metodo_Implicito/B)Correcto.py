import numpy as np
import matplotlib.pyplot as plt

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

# Método implícito (resolver un sistema de ecuaciones en cada paso de tiempo)
def metodo_implicito(T, dx, dy, dt, tol=1e-6, max_iter=10000):
    error = np.inf
    iter_count = 0
    
    # Coeficientes para el sistema implícito
    alpha_x = dt / dx**2
    alpha_y = dt / dy**2
    
    # Construcción de la matriz de coeficientes A
    A = np.zeros((Nx * Ny, Nx * Ny))
    b = np.zeros(Nx * Ny)
    
    # Construcción de la matriz A como laplaciana
    for i in range(1, Nx-1):
        for j in range(1, Ny-1):
            idx = i * Ny + j  # Índice unidimensional
            
            # Coeficiente principal
            A[idx, idx] = -2 * (alpha_x + alpha_y)
            
            # Vecinos en el sistema implícito (en el paso de tiempo siguiente)
            if i > 0: A[idx, idx - Ny] = alpha_y  # Vecino superior
            if i < Nx-1: A[idx, idx + Ny] = alpha_y  # Vecino inferior
            if j > 0: A[idx, idx - 1] = alpha_x  # Vecino izquierdo
            if j < Ny-1: A[idx, idx + 1] = alpha_x  # Vecino derecho
            
            # Lado derecho de la ecuación (término conocido)
            b[idx] = -T[i, j]
    
    # Imponer las condiciones de frontera (actualizar el vector b con los valores de frontera)
    for i in range(Nx):
        for j in range(Ny):
            idx = i * Ny + j
            if i == 0:  # Condición en y=0
                b[idx] = T_bottom
            elif i == Nx-1:  # Condición en y=1
                b[idx] = T_top
            elif j == 0:  # Condición en x=0
                b[idx] = T_left
            elif j == Ny-1:  # Condición en x=1
                b[idx] = T_right
    
    # Resolver el sistema A * T_new = b
    T_new = np.linalg.solve(A, b)
    
    # Reshape la solución de nuevo a la matriz de temperatura
    T_new = T_new.reshape((Nx, Ny))
    
    # Calcular el error (diferencia entre la iteración anterior y la nueva)
    error = np.max(np.abs(T_new - T))
    iter_count += 1
    
    # Actualizamos T
    T = T_new.copy()
    
    return T, iter_count

# Resolvemos el sistema con el método implícito
T_solution, iterations = metodo_implicito(T, dx, dy, dt)

# Mostrar resultados
print(f"Solución encontrada en {iterations} iteraciones.")

# Graficamos la solución
plt.contourf(x, y, T_solution.T, 50, cmap='inferno')
plt.colorbar(label="Temperatura (°C)")
plt.title("Distribución de temperatura estacionaria (Método Implícito)")
plt.xlabel("x")
plt.ylabel("y")
plt.show()
