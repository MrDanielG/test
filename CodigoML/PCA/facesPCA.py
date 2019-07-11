#importar modulos/librerias
import numpy as np
import pandas as pandas
import matplotlib.pyplot as plt
import seaborn as sb
from scipy.io import loadmat


def pca(X):
    #normalizar las caracteristicas
    X = (X - X.mean() / X.std())

    #Obtener las matriz de covarianza
    X = np.matrix(X)
    cov = (X.T * X) * X.shape[0]

    #Obtener SVD
    U, S, V = np.linalg.svd(conv)

    return U, S, V

def proyect_data(X, U, k):
    U_reduced = U[:,:k]
    return np.dot(X,U_reduced)

def recover_data(Z, U, k):
    U_reduced = U[:,:k]
    return np.dot(Z, U_reduced.T)

faces = loadmat('data/pca_faces.mat')
X = faces['X']
X.shape

face = np.reshape(X[70,:], (32,32))
plt.imshow(face)
plt.savefig('face_inicial.png')

