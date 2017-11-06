from posemapper import Rodrigues,MatchPointArea
import chumpy as ch
from render import renderObj
from verts import verts_core
from openmesh import *
from nonIcp import nrIcp
import config

class TemplateGenerator:
    def alignFunc(self,mat1,mat2,R):
        return ch.linalg.norm(Rodrigues(R).dot(mat1.T).T-mat2)
    def areaMatchFunc(self,dd,m):
        return ch.linalg.norm(MatchPointArea(dd=dd,m=m))

    def setTemplate(self , template):
        self.template = template

    def setScanModel(self , scanModel):
        self.scanModel = scanModel

    def setTemplateMesh(self , fileName):
        self.templateMesh = TriMesh()
        if not read_mesh(self.templateMesh, fileName):
            print "load template mesh error"
            return
        self.templateVertexNum = self.templateMesh.n_vertices()
        self.templateEdgeNum = self.templateMesh.n_edges()
        self.templateFaceNum = self.templateMesh.n_faces()
        print "load template mesh successful"

    def setScanMesh(self , fileName):
        self.scanMesh = TriMesh()
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

    def runIcp(self):
        self.nonRigitIcp = nrIcp.NonRigidIcp(self.templateMesh, self.scanMesh)
        self.nonRigitIcp.runPipeLine()
        write_mesh(self.templateMesh, 'nonIcp.obj')
