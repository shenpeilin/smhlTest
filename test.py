import chumpy as ch

x, y, A = ch.array([10,20,30]), ch.array([5]), ch.eye(3)
def f(x1, A1):
    return x1.T.dot(A1).dot(x1)
f1 = f(x, A) + y ** 2
result = ch.minimize(f1, [x,y])
class RotWithMedian(ch.Ch):
    dterms = 'protMat'
    terms = 'matchMat'
    def compute_r(self):
        dx = ch.sort(self.matchMat[:,0])[self.matchMat.shape[0]/2]-ch.sort(self.protMat[:,0])[self.protMat.shape[0]/2]
        dy = ch.sort(self.matchMat[:,1])[self.matchMat.shape[0]/2]-ch.sort(self.protMat[:,1])[self.protMat.shape[0]/2]
        dz = ch.sort(self.matchMat[:,2])[self.matchMat.shape[0]/2]-ch.sort(self.protMat[:,2])[self.protMat.shape[0]/2]
        return self.matchMat.r - (self.protMat.r+ch.array([dx,dy,dz]).r)
    def compute_dr_srt(self, wrt):
        if wrt is self.protMat:
            print 'aaaaaaaaaaaaaaaaaaaaaaaaaaaaaa'
            return sp.eye(self.protMat.size, self.protMat.size)

print result[0]