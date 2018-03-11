import cPickle as pickle
import numpy as np
import chumpy as ch
from posemapper import Rodrigues
from verts import verts_core
from render import renderObj
import scipy.optimize as scop
from openmesh import *
from sklearn.decomposition import PCA

class MatchPoint(ch.Ch):
    dterms = 'protMat','matchMat'
    terms = 'mapArray'
    def compute_r(self):
        args = {
            'pose': self.dd['pose'],
            'v': self.dd['v'],
            'J': self.dd['J'],
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

class MatchPointByJre(ch.Ch):
    dterms = 'protMat','matchMat'
    terms = 'mapArray','jreArray'
    def compute_r(self):
        args = {
            'pose': self.dd['pose'],
            'v': self.dd['v'],
            'J': self.dd['v'].T.dot(self.jreArray).T,
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

class TemplateGenerator:
    def alignFunc(self,mat1,mat2,R):
        return ch.linalg.norm(Rodrigues(R).dot(mat1.T).T-mat2)

    def poseMatchFunc(self,dd,m):
        return ch.linalg.norm(MatchPoint(dd=dd,m=m,mapArray = self.mapArray))

    def jrePointMatchFunc(self, dd, m):
        dis = ch.array(self.template['v'])
        return ch.linalg.norm(MatchPointByJre(dd=dd, m=m, mapArray = self.mapArray, jreArray = self.Jre)) + 0.5 * ch.linalg.norm(ch.amax(dd['v']-dis, axis=0))

    def setTemplate(self, template):
        self.template = template
        self.vertexNum = template['v'].shape[0]
        self.setTemplateMesh()
    
    def setScanModel(self, scanModel):
        self.scanModel = scanModel

    def setTemplateMesh(self):
        self.templateMesh = TriMesh()
        if not read_mesh(self.templateMesh, './template/template.obj'):
            print "load template mesh error"
            return
        self.edgeNum = self.templateMesh.n_edges()
        self.faceNum = self.templateMesh.n_faces()
        print "load template mesh successful"
        adjMatrix = ch.zeros((self.edgeNum*2, self.vertexNum))
        for h in self.templateMesh.halfedges():
            vh0 = self.templateMesh.from_vertex_handle(h)
            vh1 = self.templateMesh.to_vertex_handle(h)
            adjMatrix[h.idx(),vh0.idx()] = 1
            adjMatrix[h.idx(),vh1.idx()] = -1
        self.adjMatrix = adjMatrix.T
    
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
    
    def getJointRegressor(self):
        vMatrix = np.array(self.template['v'])
        factor = 1
        A = np.concatenate((vMatrix.T, factor * np.ones((1, self.vertexNum))), 0)
        jointNum = self.template['J'].shape[0]
        b = np.concatenate((self.template['J'].T, factor * np.ones((1, jointNum))), 0)
        x = scop.nnls(A, b[:, 0])[0]
        x = np.reshape(x, (self.vertexNum, 1))
        for i in range(1, jointNum):
            x1 = scop.nnls(A, b[:, i])[0]
            x1 = np.reshape(x1, (self.vertexNum, 1))
            x = np.concatenate((x, x1), 1)
        self.Jre = x
        return self.Jre
    
    def transTemplate(self):
        args = {
            'pose': self.template['pose'],
            'v': self.template['v'],
            'J': self.template['J'],
            'weights': self.template['weights'],
            'kintree_table': self.template['kintree_table'],
            'xp': ch,
            'want_Jtr': True,
            'bs_style': 'lbs',
        }
        self.result, self.Jtr = verts_core(**args)

    def renderTransedTemplate(self):
        # self.template['pose'] = ans[0]
        print self.template['pose']
        renderObj(ch.array(self.result/ch.max(self.result)),self.template["f"])
    
    def saveResult(self, result):
        outmesh_path = './output.obj'
        with open( outmesh_path, 'w') as fp: 
            for v in result.r:
                fp.write( 'v %f %f %f\n' % ( v[0], v[1], v[2]) )

            for f in self.template['f']+1: # Faces are 1-based, not 0-based in obj files
                fp.write( 'f %d %d %d\n' %  (f[0], f[1], f[2]) )

        ## Print message
        print '..Output mesh saved to: ', outmesh_path

    def normalizeScan(self):
        tjDic = {}
        tjDic['v'] = ch.array(self.template['v'])
        tjDic['pose'] = self.template['pose']
        tjDic['weights'] = self.template['weights']
        tjDic['kintree_table'] = self.template['kintree_table']
        ch.minimize(self.jrePointMatchFunc(dd=tjDic , m=self.scanModel),[tjDic['v']])
        tjDic['v'] = Rodrigues(-self.R).dot(tjDic['v'].T).T
        tjDic['v'] = tjDic['v']-self.t
        return np.reshape(tjDic['v'], (self.vertexNum * 3))

    def saveGeneratedTemp(self):
        self.tmp = {}
        allShape = np.array(TemplateGenerator.vList)
        pca = PCA()
        self.tmp['shape'] = pca.fit_transform(allShape.T).T
        self.tmp['v'] = np.mean(allShape, axis=0)
        self.tmp['f'] = self.template['f']
        self.tmp['jRegressor'] = self.Jre
        self.tmp['weights'] = self.template['weights']
        self.tmp['kintree_table'] = self.template['kintree_table']
        self.tmp['pp'] = self.template['pp']
        self.tmp['pl'] = self.template['pl']
        self.result = ch.array(np.reshape(self.tmp['v'],(-1,3)))
        self.saveResult()
        tmpFileName = 'template.pkl'
        shapeFileName = 'shapList.pkl'
        with open(tmpFileName, 'w') as f:
            pickle.dump(self.tmp, f)

        with open(shapeFileName, 'w') as f:
            pickle.dump(TemplateGenerator.vList, f)

