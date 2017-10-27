import numpy as np
from numpy import linalg as LA
import scipy.sparse as sparseMatrix
import scipy.sparse.linalg as sla
import util
import openmesh
class Geodesic:
    def __init__(self , mesh):
        self.vertexNum = mesh.n_vertices()
        self.faceNum = mesh.n_faces()
        self.edgeNum = mesh.n_edges()
        faceArea = np.zeros(self.faceNum)
        faceNorm = [None]*self.faceNum
        for h in mesh.halfedges():
            h0 = h
            if not mesh.is_boundary(h0):
                h1 = mesh.next_halfedge_handle(h0)
                h2 = mesh.next_halfedge_handle(h1)
                vh0 = mesh.to_vertex_handle(h2)
                vh1 = mesh.to_vertex_handle(h0)
                vh2 = mesh.to_vertex_handle(h1)

                v0 = mesh.point(vh0)
                v1 = mesh.point(vh1)
                v2 = mesh.point(vh2)
                e1 = v1 - v0
                e2 = v2 - v0
                fN = openmesh.cross(e1 , e2)
                area = fN.norm()/2.0
                
                fh = mesh.face_handle(h0)
                faceArea[fh.idx()] = area
                faceNorm[fh.idx()] = fN.normalize()
        
        maximumEdgeLen = 0.0

        for i in range(0 , self.edgeNum):
            eh = mesh.edge_handle(i)
            edgeLen = mesh.calc_edge_length(eh)
            if(edgeLen > maximumEdgeLen):
                maximumEdgeLen = edgeLen
        self.timeStep = 1.0 * maximumEdgeLen * maximumEdgeLen

        voronoiAreaTripletList = []
        cotangentWeightTripletList = []

        for i in range(0 , self.vertexNum):
            vh0 = mesh.vertex_handle(i)
            vertexVoronoiArea = 0.0
            vertexCotangentWeight = 0.0
            for vertexOHalfedge in mesh.voh(vh0):
                h0 = vertexOHalfedge
                if not mesh.is_boundary(h0):
                    h1 = mesh.next_halfedge_handle(h0)
                    t0 = mesh.opposite_halfedge_handle(h0)
                    t1 = mesh.next_halfedge_handle(t0)
                    fh = mesh.face_handle(h0)
                    vh1 = mesh.to_vertex_handle(h0)
                    vh2 = mesh.to_vertex_handle(h1)
                    vh3 = mesh.to_vertex_handle(t1)

                    v0 = mesh.point(vh0)
                    v1 = mesh.point(vh1)
                    v2 = mesh.point(vh2)
                    v3 = mesh.point(vh3)

                    e0 = util.Vec3DToNp(v1-v2)
                    e1 = util.Vec3DToNp(v0-v2)
                    e2 = util.Vec3DToNp(v1-v3)
                    e3 = util.Vec3DToNp(v0-v3)

                    fN = np.cross(e1,e0)
                    alpha = np.arccos(np.dot(e0 , e1) / (LA.norm(e0) * LA.norm(e1)))
                    beta = np.arccos(np.dot(e2 , e3) / (LA.norm(e2) * LA.norm(e3)))
                    fArea = LA.norm(fN)/2.0
                    cotangentWeight = 0.5 * (1.0/np.tan(alpha) + 1.0/np.tan(beta))

                    vertexCotangentWeight += -cotangentWeight
                    vertexVoronoiArea += fArea

                    cotangentWeightTripletList.append([vh0.idx(), vh1.idx(), cotangentWeight])
            cotangentWeightTripletList.append([vh0.idx(), vh0.idx(), vertexCotangentWeight])
            voronoiAreaTripletList.append([vh0.idx(), vh0.idx(), 1.0/3.0 * vertexVoronoiArea])

        self.voronoiAreaMatrix = util.getSparseMatrixFromList(voronoiAreaTripletList, (self.vertexNum, self.vertexNum))
        self.cotangentWeightMatrix = util.getSparseMatrixFromList(cotangentWeightTripletList, (self.vertexNum, self.vertexNum))

        self.A0 = self.voronoiAreaMatrix - self.timeStep * self.cotangentWeightMatrix
        self.A3 = self.cotangentWeightMatrix

        A1TripletList = []
        for i in range (0, self.faceNum):
            fh = mesh.face_handle(i)
            fvh = mesh.fv(fh)
            vh0 = fvh.next()
            vh1 = fvh.next()
            vh2 = fvh.next()

            v0 = mesh.point(vh0)
            v1 = mesh.point(vh1)
            v2 = mesh.point(vh2)

            e0 = v2 - v1
            e1 = v0 - v2
            e2 = v1 - v0

            idx0 = vh0.idx()
            idx1 = vh1.idx()
            idx2 = vh2.idx()
            fIdx = fh.idx()

            fN = faceNorm[fIdx]
            area = 2 * faceArea[fIdx]

            grad0 = openmesh.cross(fN, e0)
            grad1 = openmesh.cross(fN, e1)
            grad2 = openmesh.cross(fN, e2)

            A1TripletList.append([3*fIdx+0, idx0, grad0[0]/area])
            A1TripletList.append([3*fIdx+0, idx1, grad1[0]/area])
            A1TripletList.append([3*fIdx+0, idx2, grad2[0]/area])

            A1TripletList.append([3*fIdx+1, idx0, grad0[1]/area])
            A1TripletList.append([3*fIdx+1, idx1, grad1[1]/area])
            A1TripletList.append([3*fIdx+1, idx2, grad2[1]/area])

            A1TripletList.append([3*fIdx+2, idx0, grad0[2]/area])
            A1TripletList.append([3*fIdx+2, idx1, grad1[2]/area])
            A1TripletList.append([3*fIdx+2, idx2, grad2[2]/area])

        self.A1 = util.getSparseMatrixFromList(A1TripletList, (3* self.faceNum, self.vertexNum))
        
        A2TripletList = []
        for i in range(0, self.vertexNum):
            vh0 = mesh.vertex_handle(i)
            for vertexOHalfedge in mesh.voh(vh0):
                h0 = vertexOHalfedge
                if not mesh.is_boundary(h0):
                    h1 = mesh.next_halfedge_handle(h0)
                    vh1 = mesh.to_vertex_handle(h0)
                    vh2 = mesh.to_vertex_handle(h1)
                    v0 = mesh.point(vh0)
                    v1 = mesh.point(vh1)
                    v2 = mesh.point(vh2)

                    e1 = util.Vec3DToNp(v1 - v0)
                    e2 = util.Vec3DToNp(v2 - v0)
                    e0 = util.Vec3DToNp(v2 - v1)
                    theta1 = np.arccos(np.dot(e0,e2)/(LA.norm(e0) * LA.norm(e2)))
                    theta2 = np.arccos(np.dot(-e0,e1)/(LA.norm(e0) * LA.norm(e1)))

                    cotTheta1 = 0.5/np.tan(theta1)
                    cotTheta2 = 0.5/np.tan(theta2)

                    fh = mesh.face_handle(h0)
                    fIdx = fh.idx()

                    A2TripletList.append([i, 3*fIdx+0, cotTheta1*e1[0]+cotTheta2*e2[0]])
                    A2TripletList.append([i, 3*fIdx+1, cotTheta1*e1[1]+cotTheta2*e2[1]])
                    A2TripletList.append([i, 3*fIdx+2, cotTheta1*e1[2]+cotTheta2*e2[2]])
        self.A2 = util.getSparseMatrixFromList(A2TripletList, (self.vertexNum, 3*self.faceNum))
        self.A0lu = sla.splu(self.A0.tocsc())
        self.A3lu = sla.splu(self.A3.tocsc())

    def seed(self, seedIdx):
        heatKernel = np.zeros(self.vertexNum)
        distGrad = np.zeros(3*self.faceNum)

        heatDelta = np.zeros(self.vertexNum)
        heatDelta[seedIdx] = 1.0
        heatKernel = self.A0lu.solve(heatDelta)
        heatGrad = self.A1.dot(heatKernel)

        for j in range(0, self.faceNum):
            grad = np.array([heatGrad[3 * j + 0], heatGrad[3 * j + 1], heatGrad[3 * j + 2]])
            gradAbs = np.abs(grad)
            maxAbs = np.max(gradAbs)
            grad[0] = grad[0]/maxAbs
            grad[1] = grad[1]/maxAbs
            grad[2] = grad[2]/maxAbs
            grad = grad/LA.norm(grad)

            distGrad[3 * j + 0] = -grad[0]
            distGrad[3 * j + 1] = -grad[1]
            distGrad[3 * j + 2] = -grad[2]
        distDiv = self.A2.dot(distGrad)
        geoDist = self.A3lu.solve(distDiv)
        
        constShift = geoDist[seedIdx]
        for i in range(0, self.vertexNum):
            geoDist[i] = geoDist[i] - constShift
        return geoDist