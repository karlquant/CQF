import numpy as np
import math


def MatrixIdentity(matrix_n):
    x = np.zeros((matrix_n, matrix_n))
    for i in range(0, matrix_n):
        x[i][i] = 1
    return x

def MatrixUpperTSumSq(Xmat):
    sum = 0
    matrix_n = len(Xmat[0])
    for i in range(1, matrix_n):
        for j in range(i+1, matrix_n):
            sum += pow(Xmat[i][j], 2)
    return sum

def Jacobi_Pivot_Finder(matrix_n, Xmat):
    maxvalue = -1
    s_i = -1
    s_j = -1
    temp = 0
    n = matrix_n
    for i in range(0, n):
        for j in range(i+1, n):
            temp = abs(Xmat[i][j])
            if temp > maxvalue:
                maxvalue = temp
                s_i = i 
                s_j = j
    if Xmat[s_i][s_i] == Xmat[j][j]:
       theta = 0.25 * math.pi * math.copysign(1.0, Xmat[s_i][s_j])
    else:
       theta = 0.5 * math.atan(2 * Xmat[s_i][s_j]/(Xmat[s_i][s_i] - Xmat[s_j][s_j])) 
    return [s_i, s_j, theta]
    
def Givens_Rotation_Matrix(matrix_n, j_pivot):
    g_matrix = MatrixIdentity(matrix_n)
    g_matrix[j_pivot[0]][j_pivot[0]] = np.cos(j_pivot[2])
    g_matrix[j_pivot[1]][j_pivot[0]] = np.sin(j_pivot[2])
    g_matrix[j_pivot[0]][j_pivot[1]] = -np.sin(j_pivot[2])
    g_matrix[j_pivot[1]][j_pivot[1]] = np.cos(j_pivot[2])

    return g_matrix

def Jacobi_Rotation_Va(matrix_n, s_Matrix):
    g_Matrix = Givens_Rotation_Matrix(
        matrix_n, Jacobi_Pivot_Finder(matrix_n, s_Matrix))
    s_Matrix = np.dot(np.transpose(g_Matrix), np.dot(s_Matrix, g_Matrix))
    return s_Matrix

def Jacobi_Rotation_Ve(matrix_n, s_Matrix, v_Matrix):
    g_Matrix = Givens_Rotation_Matrix(matrix_n, Jacobi_Pivot_Finder(matrix_n, s_Matrix))
    v_Matrix = np.dot(v_Matrix, g_Matrix)
    return v_Matrix

def Jacobi_Transformation(Xmat, tolerance):
    n = len(Xmat[0])
    s_Matrix = Xmat
    msumsq = MatrixUpperTSumSq(s_Matrix) 
    v_Matrix = MatrixIdentity(n)
    while msumsq > tolerance:
        s_next = Jacobi_Rotation_Va(n, s_Matrix)
        v_next = Jacobi_Rotation_Ve(n, s_Matrix, v_Matrix)
        msumsq = MatrixUpperTSumSq(s_next)
        s_Matrix = s_next
        v_Matrix = v_next
    return {'eigenValues':s_Matrix, 'eigenVectors':v_Matrix}



