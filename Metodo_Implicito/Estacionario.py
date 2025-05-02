from scipy.sparse import lil_matrix
from scipy.sparse.linalg import spsolve

T = np.zeros((Nx, Ny))
x = np.linspace(0, L, Nx)
y = np.linspace(0, L, Ny)
X, Y = np.meshgrid(x, y)

# Ensamblaje matriz del sistema
A = lil_matrix((N, N))
b = np.zeros(N)

def idx(i, j): return i * Ny + j

for i in range(Nx):
    for j in range(Ny):
        k = idx(i, j)
        if i == 0 or i == Nx-1:
            A[k, k] = 1
            b[k] = 0
        elif j == 0:
            A[k, k] = 1
            A[k, idx(i, j+1)] = -1
        elif j == Ny-1:
            A[k, k] = 1
            A[k, idx(i, j-1)] = -1
        else:
            A[k, k] = -4
            A[k, idx(i+1, j)] = 1
            A[k, idx(i-1, j)] = 1
            A[k, idx(i, j+1)] = 1
            A[k, idx(i, j-1)] = 1

T = spsolve(A.tocsc(), b).reshape((Nx, Ny))

plt.contourf(X, Y, T, cmap='hot')
plt.title("Temperatura - Estacionaria Implícita (Laplaciano)")
plt.colorbar()
plt.show()
