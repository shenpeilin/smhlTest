import numpy as np
def rotation(mat):
    rotation = np.array([[np.cos(1.05), -np.sin(1.05), 0],[np.sin(1.05), np.cos(1.05), 0] , [0,0,1]])
    return rotation.dot(mat.T).T+np.array([1,2,3])