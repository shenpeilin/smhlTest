import numpy as np
import cv2

class KNNTree:
    def __init__(self, currentNonRigidMesh, currentScanMesh):
        self.nonRigidArray = np.zeros((currentNonRigidMesh.n_vertices(), 3), np.float32)
        self.currentScanArray = np.zeros((currentScanMesh.n_vertices(), 3), np.float32)
        self.mesh = currentScanMesh
        for i in range(currentNonRigidMesh.n_vertices()):
            vh = currentNonRigidMesh.vertex_handle(i)
            v = currentNonRigidMesh.point(vh)
            self.nonRigidArray[i] = [v[0], v[1], v[2]]

        for i in range(currentScanMesh.n_vertices()):
            vh = currentScanMesh.vertex_handle(i)
            v = currentScanMesh.point(vh)
            self.currentScanArray[i] = [v[0], v[1], v[2]]
        FLANN_INDEX_KDTREE = 1
        index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 8)
        search_params = dict(checks=50)   # or pass empty dictionary
        flann = cv2.FlannBasedMatcher(index_params,search_params)
        self.matches = flann.knnMatch(self.nonRigidArray, self.currentScanArray, 1)

    def nearest(self, pointIdx):
        matchIdx = self.matches[pointIdx][0].trainIdx
        return self.mesh.vertex_handle(matchIdx)