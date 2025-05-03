import numpy as np

print("------ Matrix and Vector Operations ------")

# 1. Matrix A and Vector B
A = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
B = np.array([1, 2, 3])

# Matrix-vector multiplication
try:
    result = A @ B
    print("A x B =\n", result)
except ValueError as e:
    print("Matrix-vector multiplication error:", e)

# Trace of A
print("Trace of A:", np.trace(A))

# Eigenvalues and Eigenvectors
eigvals, eigvecs = np.linalg.eig(A)
print("Eigenvalues:\n", eigvals)
print("Eigenvectors:\n", eigvecs)

# 2. Replace last row
A[2] = [10, 11, 12]
print("\nUpdated Matrix A:\n", A)

# Determinant
det = np.linalg.det(A)
print("Determinant of updated A:", det)

# Check singularity
if det == 0:
    print("Matrix A is singular (non-invertible).")
else:
    print("Matrix A is non-singular (invertible).")

print("\n------ Invertibility of Matrices ------")

# 1. Inverse if possible
if det != 0:
    A_inv = np.linalg.inv(A)
    print("Inverse of A:\n", A_inv)
else:
    print("Cannot compute inverse of a singular matrix.")

# 2. Solve A x X = B
try:
    X = np.linalg.solve(A, B)
    print("Solution X to A x X = B:\n", X)
except np.linalg.LinAlgError as e:
    print("Cannot solve linear system:", e)

print("\n------ Practical Matrix Operations ------")

# 1. Random 4x4 matrix
C = np.random.randint(1, 21, size=(4, 4))
print("Matrix C:\n", C)

# Rank of C
print("Rank of C:", np.linalg.matrix_rank(C))

# Submatrix
submatrix = C[:2, -2:]
print("Submatrix (first 2 rows, last 2 cols):\n", submatrix)

# Frobenius norm
print("Frobenius norm of C:", np.linalg.norm(C, 'fro'))

# 2. Multiplication with updated A
try:
    C_trimmed = C[:3, :3]
    AC_product = A @ C_trimmed
    print("A x C_trimmed:\n", AC_product)
except ValueError as e:
    print("Matrix multiplication error:", e)

print("\n------ Data Science Context ------")

# 1. Dataset D
D = np.array([
    [3, 5, 7, 9, 11],
    [2, 4, 6, 8, 10],
    [1, 3, 5, 7, 9],
    [4, 6, 8, 10, 12],
    [5, 7, 9, 11, 13]
])
print("Dataset D:\n", D)

# Standardize D
D_mean = D.mean(axis=0)
D_std = D.std(axis=0)
D_standardized = (D - D_mean) / D_std
print("Standardized D:\n", D_standardized)

# Covariance matrix
cov_matrix = np.cov(D_standardized, rowvar=False)
print("Covariance Matrix:\n", cov_matrix)

# PCA - Eigen decomposition
eig_vals_D, eig_vecs_D = np.linalg.eig(cov_matrix)
print("Eigenvalues:\n", eig_vals_D)
print("Eigenvectors:\n", eig_vecs_D)

# Reduce to 2 principal components
indices = np.argsort(eig_vals_D)[::-1]
top_2_vectors = eig_vecs_D[:, indices[:2]]
D_reduced = D_standardized @ top_2_vectors
print("D reduced to 2 principal components:\n", D_reduced)
