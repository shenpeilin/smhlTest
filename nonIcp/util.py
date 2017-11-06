import numpy as np
import scipy.sparse as sparseMatrix
import cv2

def Vec3DToNp(v):
    return np.array([v[0],v[1],v[2]])

def getSparseMatrixFromList(tripletList,matrixShape = None):
    tripletArray = np.array(tripletList)
    return sparseMatrix.coo_matrix((tripletArray[:,2]
        , (tripletArray[:,0],tripletArray[:,1]))
        , shape = matrixShape)
