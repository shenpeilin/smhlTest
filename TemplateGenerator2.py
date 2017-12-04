import cPickle as pickle
import numpy as np
import chumpy as ch
from posemapper import Rodrigues
from verts import verts_core
from render import renderObj
import scipy.optimize as scop
from openmesh import *
from sklearn.decomposition import PCA
from serialization import loadObj
import config

class MatchPoint(ch.Ch):
    dterms = 'protMat','matchMat'
    terms = 'mapArray'
    def compute_r(self):
        args = {
            'pose': self.dd['pose'],
            'v': self.dd['v'] + self.dd['beta'].dot(self.dd['shape']).reshape((-1,3)),
            'J': self.dd['v'].T.dot(self.dd['jRegressor']).T,
            'weights': self.dd['weights'],
            'kintree_table': self.dd['kintree_table'],
            'xp': ch,
            'want_Jtr': False,
            'bs_style': 'lbs',
        }
        self.protMat = verts_core(**args)
        self.matchMat = self.m['v']
        return self.matchMat.r[self.mapArray] - self.protMat.r
    def compute_dr_wrt(self, wrt):
        return (self.matchMat[self.mapArray]-self.protMat).dr_wrt(wrt)

class TemplateGenerator2:
    vList = []
    def alignFunc(self,mat1,mat2,R):
        return ch.linalg.norm(Rodrigues(R).dot(mat1.T).T-mat2)

    def poseMatchFunc(self,dd,m):
        return ch.linalg.norm(MatchPoint(dd=dd,m=m,mapArray = self.mapArray))

    def loadTemplate(self):
        with open('template.pkl', 'rb') as f:
            self.template = pickle.load(f)
        self.template['v'] = self.template['v'].reshape((-1,3))
        self.vertexNum = self.template['v'].shape[0]
        self.template['J'] = (self.template['v'].T).dot(self.template['jRegressor']).T
        self.template['shape'] = self.template['shape'][0:config.SHAPE_VEC_NUM]
        self.template["pose"] = ch.zeros(self.template["kintree_table"].shape[1]*3)
        self.template['beta'] = ch.zeros(config.SHAPE_VEC_NUM)

    def setScanModel(self, fileName):
        self.scanModel = loadObj(fileName)
        self.setMapArray(fileName)
    
    def setMapArray(self, objFileName):
        mapFilename = objFileName.replace(".obj", "Map.pkl")
        with open(mapFilename, 'rb') as f:
            indexMap = pickle.load(f)
        mapArray = np.zeros((self.vertexNum,) , np.int)
        for indexTur in indexMap:
            mapArray[indexTur[0]] = indexTur[1]
        
        self.mapArray = mapArray
    
    def alignTemplate(self):
        R = ch.zeros(3)
        t = self.scanModel['v'][self.mapArray[self.template['pl'][4]]]-self.template['pp'][4]
        ch.minimize(self.alignFunc(self.template['pp'][0:5,:]+t,self.scanModel['v'][self.mapArray[self.template['pl'][0:5]],:],R),[R])
        self.template['v'] = self.template['v']+t
        self.template['J'] = self.template['J']+t
        self.template['v'] = Rodrigues(R).dot(self.template['v'].T).T
        self.template['J'] = Rodrigues(R).dot(self.template['J'].T).T
        self.R = R
        self.t = t
    
    def poseMatch(self):
        ch.minimize(self.poseMatchFunc(dd=self.template , m=self.scanModel),[self.template["pose"]])
        ch.minimize(self.poseMatchFunc(dd=self.template , m=self.scanModel),[self.template["beta"]])
        ch.minimize(self.poseMatchFunc(dd=self.template , m=self.scanModel),[self.template["pose"]])
    
    def renderTransedTemplate(self):
        # self.template['pose'] = ans[0]
        print self.template['pose']
        renderObj(ch.array(self.result/ch.max(self.result)),self.template["f"])

    def transTemplate(self):
        args = {
            'pose': self.template['pose'],
            'v': self.template['v'] + self.template['beta'].dot(self.template['shape']).reshape((-1,3)),
            'J': self.template['v'].T.dot(self.template['jRegressor']).T,
            'weights': self.template['weights'],
            'kintree_table': self.template['kintree_table'],
            'xp': ch,
            'want_Jtr': False,
            'bs_style': 'lbs',
        }
        self.result = verts_core(**args)
        self.saveResult()
    
    def saveResult(self):
        outmesh_path = './output.obj'
        outmesh_path_1 = './outputB.obj'
        with open( outmesh_path, 'w') as fp: 
            for v in self.result.r:
                fp.write( 'v %f %f %f\n' % ( v[0], v[1], v[2]) )

            for f in self.template['f']+1: # Faces are 1-based, not 0-based in obj files
                fp.write( 'f %d %d %d\n' %  (f[0], f[1], f[2]) )

        with open( outmesh_path_1, 'w') as fp:
            for v in ch.array(self.template['v']).r:
                fp.write( 'v %f %f %f\n' % ( v[0], v[1], v[2]) )

            for f in self.template['f']+1: # Faces are 1-based, not 0-based in obj files
                fp.write( 'f %d %d %d\n' %  (f[0], f[1], f[2]) )

        ## Print message
        print '..Output mesh saved to: ', outmesh_path

    def saveGeneratedTemp(self):
        self.tmp = {}
        TemplateGenerator.vList.append(np.reshape(self.template['v'],(-1)))
        allShape = np.array(TemplateGenerator.vList)
        pca = PCA()
        self.tmp['shape'] = pca.fit_transform(allShape.T).T
        self.tmp['v'] = np.mean(allShape, axis=0)
        self.tmp['f'] = self.template['f']
        self.tmp['jRegressor'] = self.Jre
        self.tmp['weights'] = self.template['weights']
        self.result = ch.array(np.reshape(self.tmp['v'],(-1,3)))
        self.saveResult()
        tmpFileName = 'template.pkl'
        with open(tmpFileName, 'w') as f:
            pickle.dump(self.tmp, f)

