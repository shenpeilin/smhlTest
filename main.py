from serialization import loadObj
import numpy as np
from render import renderObj
from matUtil import rotation
import chumpy as ch
from chumpy import ch_ops
import numpy as np
import scipy.sparse as sp
import cv2

class Rodrigues(ch.Ch):
    dterms = 'rt'
    
    def compute_r(self):
        return cv2.Rodrigues(self.rt.r)[0]
    
    def compute_dr_wrt(self, wrt):
        if wrt is self.rt:
            return cv2.Rodrigues(self.rt.r)[1].T

class RotWithMedian(ch.Ch):
    dterms = 'protMat','matchMat'
    def compute_r(self):
        dx = ch.sort(self.matchMat[:,0])[self.matchMat.shape[0]/2]-ch.sort(self.protMat[:,0])[self.protMat.shape[0]/2]
        dy = ch.sort(self.matchMat[:,1])[self.matchMat.shape[0]/2]-ch.sort(self.protMat[:,1])[self.protMat.shape[0]/2]
        dz = ch.sort(self.matchMat[:,2])[self.matchMat.shape[0]/2]-ch.sort(self.protMat[:,2])[self.protMat.shape[0]/2]
        return self.matchMat.r - (self.protMat.r+np.array([dx[0],dy[0],dz[0]]).reshape(1,3))
    def compute_dr_wrt(self, wrt):
        if (wrt is self.protMat) == (wrt is self.matchMat):
            return None
        m = -1. if wrt is self.protMat else 1.
        return ch_ops._broadcast_matrix(self.matchMat,self.protMat,wrt,m)
def mapTwoMat(mat1,mat2):
    x = np.zeros((1,mat1.shape[1]))
    A = np.zeros((mat1.shape[0]*3,3))
    for i in range(0,10):
        point = mat1[0,:]
        A[0:3,:] = np.array([[0,point[2],-point[1]],[-point[2],0,point[0]],[point[1],-point[0],0]])
        for i in range(1,mat1.shape[0]):
            point = mat1[i,:]
            A[i*3:i*3+3,:] = np.array([[0,point[2],-point[1]],[-point[2],0,point[0]],[point[1],-point[0],0]])
        print 'aaaaaaaaaaaaaaaaaaaaaaaaaaaa'
        b=np.reshape(mat2-mat1, (1,-1)).T

        x = np.linalg.lstsq(A,b)
        print x
        R = np.array([[0,-x[0][2],x[0][1]], [x[0][2],0,-x[0][0]], [-x[0][1],x[0][0],0]])+np.eye(3)
        mat1 = R.dot(mat1.T).T
    return x
def func(mat1,mat2,R,t):
    return ch.linalg.norm(RotWithMedian(protMat=Rodrigues(R).dot(mat1.T).T,matchMat=mat2))

m=loadObj("CuiHao1.obj")
protMat = ch.array(m["v"])
m["v"] = ch.array(rotation(m["v"]))
R = ch.array(np.random.rand(1,3))
t = ch.array([1,2,3])
# renderObj(ch.array(m["v"]/np.max(m["v"])),m["f"])
# ans = ch.minimize(func(protMat,m["v"],R,t) , [R])
# print ans
mapTwoMat(protMat,m["v"])
