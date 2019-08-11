# Ref. https://github.com/AdalbertoCq/Pathology-GAN/blob/133010ecad21a9940cd8d606fa9571299a5c5328/models/generative/evaluation.py

import tensorflow as tf
import numpy as np
from scipy import sparse
import scipy.sparse.linalg as sp_linalg


# Singular value desposition with Alrnoldi Iteration Method.
def matrix_singular_values(matrix, n_sing, mode='LM'):
    filter_shape = matrix.shape
    matrix_reshape = np.reshape(matrix, [-1, filter_shape[-1]])

    # Semi-positive definite matrix A*A.T, A.T*A
    dim1, dim2 = matrix_reshape.shape
    if dim1 > dim2:
        aa_t = np.matmul(matrix_reshape.T, matrix_reshape)
    else:
        aa_t = np.matmul(matrix_reshape, matrix_reshape.T)

    # RuntimeWarning
    # Trows warning to use eig instead if the say too is small.1
    # Eigs is an approximation, Eig calculated all eigenvalues, and eigenvectors of the matrix.

    try:
        eigenvalues, eigenvectors = sp_linalg.eigs(A=aa_t, k=n_sing, which=mode)
    except RuntimeWarning:
        pass
    except RuntimeError:
        eigenvalues = None
        pass

    if eigenvalues is None:
        return None

    if n_sing > 1:
        eigenvalues = np.sort(eigenvalues)
        if 'LM' in mode:
            eigenvalues = eigenvalues[::-1]

    sing_matrix = np.sqrt(eigenvalues)

    return np.real(sing_matrix)