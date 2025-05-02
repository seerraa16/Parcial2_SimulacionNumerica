import numpy as np
import matplotlib.pyplot as plt
T = np.zeros((Nx, Ny))
T[:, :] = np.sin(np.pi * X) * np.sin(np.pi * Y)

tol = 1e-5
error = 1.0
while error > tol:
    Tn = T.copy()
    for i in range(1, Nx-1):
        for j in range(1, Ny-1):
            T[i, j] = 0.25 * (Tn[i+1, j] + Tn[i-1, j] + Tn[i, j+1] + Tn[i, j-1])
    T[0, :] = 0
    T[-1, :] = 0
    T[:, 0] = T[:, 1]
    T[:, -1] = T[:, -2]
    error = np.max(np.abs(T - Tn))

plt.contourf(X, Y, T, cmap='hot')
plt.title("Temperatura - Estacionaria Explícita (Jacobi)")
plt.colorbar()
plt.show()
