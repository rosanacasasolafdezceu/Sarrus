import numpy as np

def sarrus_iterative(matrix):
    """
    Calcula el determinante de una matriz 3x3 usando la regla de Sarrus (iterativo).
    """
    if matrix.shape != (3, 3):
        raise ValueError("La matriz debe ser de 3x3 para la regla de Sarrus.")
    diag1 = matrix[0,0]*matrix[1,1]*matrix[2,2]
    diag2 = matrix[0,1]*matrix[1,2]*matrix[2,0]
    diag3 = matrix[0,2]*matrix[1,0]*matrix[2,1]
    diag4 = matrix[0,2]*matrix[1,1]*matrix[2,0]
    diag5 = matrix[0,0]*matrix[1,2]*matrix[2,1]
    diag6 = matrix[0,1]*matrix[1,0]*matrix[2,2]
    return (diag1 + diag2 + diag3) - (diag4 + diag5 + diag6)

def determinant_recursive(matrix):
    """
    Calcula el determinante de una matriz cuadrada de cualquier tamaño usando recursión (expansión de Laplace).
    """
    n = matrix.shape[0]
    if n != matrix.shape[1]:
        raise ValueError("La matriz debe ser cuadrada.")
    if n == 1:
        return matrix[0,0]
    if n == 2:
        return matrix[0,0]*matrix[1,1] - matrix[0,1]*matrix[1,0]
    det = 0
    for col in range(n):
        minor = np.delete(np.delete(matrix, 0, axis=0), col, axis=1)
        det += ((-1) ** col) * matrix[0, col] * determinant_recursive(minor)
    return det

if __name__ == "__main__":
    matriz = np.array([[1, 2, 3],
                       [4, 5, 6],
                       [7, 8, 9]])
    print("Determinante (Sarrus iterativo):", sarrus_iterative(matriz))
    print("Determinante (Recursivo):", determinant_recursive(matriz))