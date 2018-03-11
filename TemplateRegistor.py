from posemapper import Rodrigues,getMatches
import numpy as np
import chumpy as ch
from render import renderObj
from verts import verts_core
from openmesh import *
from nonIcp import nrIcp
import config
import cPickle as pickle
import sys, traceback
import logging
LOG_FILENAME = './runtimelog.out'
logging.basicConfig(filename=LOG_FILENAME, level=logging.DEBUG)

class MatchPointArea(ch.Ch):
    dterms = 'protMat','matchMat'
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
        return self.matchMat.r[self.m['pl']] - self.protMat.r[self.dd['pl']]
    def compute_dr_wrt(self, wrt):
        return (self.matchMat[self.m['pl']]-self.protMat[self.dd['pl']]).dr_wrt(wrt)

class MatchTotalArea(ch.Ch):
    dterms = 'protMat','matchMat'
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
        areaMatches = getMatches(np.array(self.matchMat,np.float32),np.array(self.protMat,np.float32),1)
        self.indexArray = np.zeros((self.matchMat.shape[0]),dtype=int)
        for i in range(0, self.matchMat.shape[0]):
            self.indexArray[i] = areaMatches[i][0].trainIdx 
        return self.matchMat.r - self.protMat.r[self.indexArray]
    def compute_dr_wrt(self, wrt):
        return (self.matchMat-self.protMat[self.indexArray]).dr_wrt(wrt)

class TemplateRigistor:
    def alignFunc(self,mat1,mat2,R):
        return ch.linalg.norm(Rodrigues(R).dot(mat1.T).T-mat2)
    def areaMatchFunc(self,dd,m):
        return ch.linalg.norm(MatchPointArea(dd=dd,m=m)) + 0.01 * ch.linalg.norm(MatchTotalArea(dd=dd, m=m))

    def setTemplate(self , template):
        self.template = template

    def setScanModel(self , scanModel):
        self.scanModel = scanModel

    def setTemplateMesh(self):
        self.templateMesh = TriMesh()
        if not read_mesh(self.templateMesh, self.outputPath):
            print "load template mesh error"
            return
        self.templateVertexNum = self.templateMesh.n_vertices()
        self.templateEdgeNum = self.templateMesh.n_edges()
        self.templateFaceNum = self.templateMesh.n_faces()
        print "load template mesh successful"

    def setScanMesh(self , fileName):
        self.scanMesh = TriMesh()
        self.modelFileName = fileName
        if not read_mesh(self.scanMesh, fileName):
            print "load scan mesh error"
        else:
            print "load scan mesh successful"

    def alignTemplate(self):
        R = ch.zeros(3)
        t = self.scanModel['pp'][4]-self.template['pp'][4]
        ch.minimize(self.alignFunc(self.template['pp'][0:5,:]+t,self.scanModel['pp'][0:5,:],R),[R])
        self.template['v'] = self.template['v']+t
        self.template['J'] = self.template['J']+t
        self.template['v'] = Rodrigues(R).dot(self.template['v'].T).T
        self.template['J'] = Rodrigues(R).dot(self.template['J'].T).T
    
    def areaMatch(self):
        ch.minimize(self.areaMatchFunc(dd=self.template , m=self.scanModel),[self.template["pose"]])

    def transTemplate(self):
        args = {
            'pose': self.template['pose'],
            'v': self.template['v'],
            'J': self.template['J'],
            'weights': self.template['weights'],
            'kintree_table': self.template['kintree_table'],
            'xp': ch,
            'want_Jtr': False,
            'bs_style': 'lbs',
        }
        self.result = verts_core(**args)

    def renderTransedTemplate(self):
        # self.template['pose'] = ans[0]
        print self.template['pose']
        renderObj(ch.array(self.result/ch.max(self.result)),self.template["f"])

    def saveResult(self):
        outmesh_path = './output/'
        idx = self.modelFileName.rfind('/')
        outmesh_path =outmesh_path + self.modelFileName[idx+1:].replace(".obj", "Out.obj")
        self.outputPath = outmesh_path
        with open( outmesh_path, 'w') as fp:
            for v in self.result.r:
                fp.write( 'v %f %f %f\n' % ( v[0], v[1], v[2]) )

            for f in self.template['f']+1: # Faces are 1-based, not 0-based in obj files
                fp.write( 'f %d %d %d\n' %  (f[0], f[1], f[2]) )


        ## Print message
        print '..Output mesh saved to: ', outmesh_path

    def runIcp(self):
        self.nonRigitIcp = nrIcp.NonRigidIcp(self.templateMesh, self.scanMesh)
        try:
            vertexPair = self.nonRigitIcp.runPipeLine()
            mapFilename = self.modelFileName.replace(".obj", "Map.pkl")
            with open(mapFilename, 'w') as f:
                pickle.dump(vertexPair, f)
            idx = self.modelFileName.rfind('/')
            nonIcpFileName = self.modelFileName[idx+1:].replace(".obj", "Icp.obj")
            write_mesh(self.templateMesh, './NonIcp/'+nonIcpFileName)
        except:
            print 'exception happen'
            traceback.print_exc(file=sys.stdout)
            logging.exception('Got exception in' + self.modelFileName)
