import chumpy as ch
import numpy as np
from chumpy import ch_ops
import cv2


class Rodrigues(ch.Ch):
    dterms = 'rt'
    
    def compute_r(self):
        return cv2.Rodrigues(self.rt.r)[0]
    
    def compute_dr_wrt(self, wrt):
        if wrt is self.rt:
            return cv2.Rodrigues(self.rt.r)[1].T

class MatchPoint(ch.Ch):
    dterms = 'protMat','matchMat'
    terms = 'dlist','mlist','indexArray', 'dpp', 'mpp'
    def compute_r(self):
        npa = np.array(self.protMat.r,np.float32)
        b = np.array(self.matchMat.r,np.float32)
        indexArray = np.zeros(npa.shape[0] , np.int32)
        for i in range(0,20):
            matches = getMatches(npa[self.dlist[i],:], b[self.mlist[i],:])
            for j in range(0,len(self.dlist[i])):
                indexArray[self.dlist[i][j]] = self.mlist[i][matches[j][0].trainIdx]
        indexArray[self.dpp] = self.mpp
        self.indexArray = indexArray
        return self.matchMat.r[indexArray,:] - self.protMat.r
    def compute_dr_wrt(self, wrt):
        if (wrt is self.protMat) == (wrt is self.matchMat):
            return None
        m = -1. if wrt is self.protMat else 1.
        return ch_ops._broadcast_matrix(self.matchMat[self.indexArray,:],self.protMat,wrt,m)


def lrotmin(p): 
    if isinstance(p, np.ndarray):
        p = p.ravel()[3:]
        return np.concatenate([(cv2.Rodrigues(np.array(pp))[0]-np.eye(3)).ravel() for pp in p.reshape((-1,3))]).ravel()        
    if p.ndim != 2 or p.shape[1] != 3:
        p = p.reshape((-1,3))
    p = p[1:]
    return ch.concatenate([(Rodrigues(pp)-ch.eye(3)).ravel() for pp in p]).ravel()

def posemap(s):
    if s == 'lrotmin':
        return lrotmin
    else:
        raise Exception('Unknown posemapping: %s' % (str(s),))

def getMatches(a,b,k):
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 8)
    search_params = dict(checks=50)   # or pass empty dictionary
    flann = cv2.FlannBasedMatcher(index_params,search_params)
    matches = flann.knnMatch(a,b,k)
    return matches