from serialization import loadObj
from serialization import loadTemplate
import numpy as np
from render import renderObj
from matUtil import rotation
import chumpy as ch
import numpy as np
import scipy.sparse as sp
import cv2
from posemapper import Rodrigues
from posemapper import RotWithMedian
from verts import verts_core

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
def func(mat1,mat2,R):
    return ch.linalg.norm(RotWithMedian(protMat=Rodrigues(R).dot(mat1.T).T,matchMat=mat2))

def objFunc(dd,m):
    args = {
        'pose': dd['pose'],
        'v': dd['v'],
        'J': dd['J'],
        'weights': dd['weights'],
        'kintree_table': dd['kintree_table'],
        'xp': ch,
        'want_Jtr': False,
        'bs_style': 'lbs',
    }
    return ch.linalg.norm(RotWithMedian(protMat = verts_core(**args),matchMat=m['v']))+1000*ch.linalg.norm(dd['pose'])

dd=loadTemplate()
m = loadObj('./HandScan/ZhangYueYi3.obj')

# protMat = ch.array(m["v"])
# m["v"] = ch.array(rotation(m["v"]))
# t = ch.array([1,2,3])
# renderObj(ch.array(result/ch.max(result)),dd["f"])
affine = cv2.estimateAffine3D(dd['pp'],m['pp'])[1]
dd["v"]=ch.array(affine[:,0:3].dot(dd["v"].T).T+affine[:,3].T)
dd["J"]=ch.array(affine[:,0:3].dot(dd["J"].T).T+affine[:,3].T)
ch.minimize(objFunc(dd,m),[dd["pose"]])
args = {
    'pose': dd['pose'],
    'v': dd['v'],
    'J': dd['J'],
    'weights': dd['weights'],
    'kintree_table': dd['kintree_table'],
    'xp': ch,
    'want_Jtr': False,
    'bs_style': 'lbs',
}
result = verts_core(**args)
# dd['pose'] = ans[0]
print dd['pose']
renderObj(ch.array(result/ch.max(result)),dd["f"])

outmesh_path = './output.obj'
with open( outmesh_path, 'w') as fp:
    for v in result.r:
        fp.write( 'v %f %f %f\n' % ( v[0], v[1], v[2]) )

    for f in dd['f']+1: # Faces are 1-based, not 0-based in obj files
        fp.write( 'f %d %d %d\n' %  (f[0], f[1], f[2]) )

## Print message
print '..Output mesh saved to: ', outmesh_path 