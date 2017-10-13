import chumpy as ch
import numpy as np
from chumpy import ch_ops
import config
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
        self.indexArray = np.zeros(len(self.dpp),np.int32)
        for i in range(0,len(self.dpp)):
            matches = getMatches(np.array(self.protMat[self.dpp[i]],np.float32),np.array(self.matchMat[self.mlist[i]],np.float32),1)
            self.indexArray[i] = self.mlist[i][matches[0][0].trainIdx]
        return self.matchMat.r[self.indexArray] - self.protMat.r[self.dpp]
    def compute_dr_wrt(self, wrt):
        return (self.matchMat[self.indexArray]-self.protMat[self.dpp]).dr_wrt(wrt)


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