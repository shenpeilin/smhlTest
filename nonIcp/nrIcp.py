import nodeSampler
import config
import numpy as np
import numpy.linalg as LA
from kdTree import KDTree
from knnTree import KNNTree
import util
import math
import openmesh
from gnParams import GnParams

class NonRigidIcp:
    def __init__(self, templateMesh, scanMesh):
        self.templateMesh = templateMesh
        self.scanMesh = scanMesh
        self.templateVertexNum = templateMesh.n_vertices()
        self.forwardControlParams = GnParams()
    
    def runPipeLine(self):
        forwardSampler = nodeSampler.NodeSampler()
        forwardSampler.sample(self.templateMesh , config.SAMPLERADIUS , 1)
        forwardSampler.constructGraph()
        # forwardSampler.drawNodes(self.templateMesh,'node.off')
        # forwardSampler.drawNodeGraph(self.templateMesh, 'nodeGraph.off')
        accumAffineVector = self.initAffineVector(forwardSampler.nodeSize())
        anchorSampler = forwardSampler
        anchorMesh = self.templateMesh
        currentNonRigidMesh = self.templateMesh
        currentScanMesh = self.scanMesh
        currentScanMesh.request_face_normals()
        currentScanMesh.request_vertex_normals()
        currentScanMesh.update_normals()
        icpEnergy = 0.0
        icpEnergyPrev = 0.0
        icpIterCnt = 0
        while True:
            currentNonRigidMesh.request_face_normals()
            currentNonRigidMesh.request_vertex_normals()
            currentNonRigidMesh.update_normals()
            vertexPair = self.findVertexPair(currentNonRigidMesh, currentScanMesh)
            affineVector = self.initAffineVector(forwardSampler.nodeSize())
            icpEnergy = self.gaussNewton(currentNonRigidMesh, currentScanMesh, vertexPair, forwardSampler, affineVector, self.forwardControlParams)
            self.accumulateAffine(affineVector, accumAffineVector, forwardSampler.nodeSize())
            self.updateNonRigidMesh(currentNonRigidMesh, affineVector, forwardSampler)
            icpEnergyChange = 0.0
            if icpIterCnt > 0:
                icpEnergyChange = abs(icpEnergy - icpEnergyPrev)/icpEnergy
                print 'icpEnergy = %f, icpEnergyChange = %f.' % (icpEnergy, icpEnergyChange)
            
            if icpEnergyChange < 0.005 and icpIterCnt > 0:
                self.forwardControlParams.relaxParams()
            
            if self.forwardControlParams.alphaRigid < 100 or icpIterCnt >= 100:
                return vertexPair

            icpEnergyPrev = icpEnergy
            icpIterCnt += 1
            
    def initAffineVector(self, nodeSize):
        affineVector = np.zeros((12 * nodeSize))
        for nodeIdx in range(nodeSize):
            affineVector[12*nodeIdx : 12*(nodeIdx+1)] = [1.0,0.0,0.0
            ,0.0,1.0,0.0
            ,0.0,0.0,1.0
            ,0.0,0.0,0.0]

        return affineVector

    def findVertexPair(self, currentNonRigidMesh, currentScanMesh):
        vertexPair = []
        knnTreeSolver = KNNTree(currentNonRigidMesh, currentScanMesh)
        for i in range(self.templateVertexNum):
            vh0 = currentNonRigidMesh.vertex_handle(i)
            v0 = currentNonRigidMesh.point(vh0)
            vh1 = knnTreeSolver.nearest(i)
            v1 = currentScanMesh.point(vh1)
            # isBoundary = currentScanMesh.is_boundary(vh1)
            # if(isBoundary):
            #     continue

            # n0 = util.Vec3DToNp(currentNonRigidMesh.normal(vh0))
            # n1 = util.Vec3DToNp(currentScanMesh.normal(vh1))
            # normalDot = np.dot(n0, n1)
            # if(normalDot < config.THETALIMIT/180.0 * np.pi):
            #     continue
            
            # dist = LA.norm(util.Vec3DToNp(v0 - v1))
            # if(dist > config.DISTLIMIT):
            #     continue

            vertexPair.append((vh0.idx(), vh1.idx()))
        
        return vertexPair

    def gaussNewton(self, currentNonRigidMesh, currentScanMesh, vertexPair, sampler, affineVector, controlParams):
        gaussNewtonEnergy = 0.0
        gaussNewtonEnergyPre = 0.0
        gaussNewtonEnergyIterCnt = 0
        while True:
            JrList = []
            AsList = []
            ApList = []
            AqList = []

            bs = []
            bp = []
            bq = []

            frVector = []

            for i in range(sampler.nodeSize()):
                rigid = [0] * 6
                rigid[0] += affineVector[12 * i + 0] * affineVector[12 * i + 3]
                rigid[0] += affineVector[12 * i + 1] * affineVector[12 * i + 4]
                rigid[0] += affineVector[12 * i + 2] * affineVector[12 * i + 5]

                rigid[1] += affineVector[12 * i + 3] * affineVector[12 * i + 6]
                rigid[1] += affineVector[12 * i + 4] * affineVector[12 * i + 7]
                rigid[1] += affineVector[12 * i + 5] * affineVector[12 * i + 8]

                rigid[2] += affineVector[12 * i + 0] * affineVector[12 * i + 6]
                rigid[2] += affineVector[12 * i + 1] * affineVector[12 * i + 7]
                rigid[2] += affineVector[12 * i + 2] * affineVector[12 * i + 8]

                rigid[3] += affineVector[12 * i + 0] * affineVector[12 * i + 0]
                rigid[3] += affineVector[12 * i + 1] * affineVector[12 * i + 1]
                rigid[3] += affineVector[12 * i + 2] * affineVector[12 * i + 2]
                rigid[3] -= 1

                rigid[4] += affineVector[12 * i + 3] * affineVector[12 * i + 3]
                rigid[4] += affineVector[12 * i + 4] * affineVector[12 * i + 4]
                rigid[4] += affineVector[12 * i + 5] * affineVector[12 * i + 5]
                rigid[4] -= 1

                rigid[5] += affineVector[12 * i + 6] * affineVector[12 * i + 6]
                rigid[5] += affineVector[12 * i + 7] * affineVector[12 * i + 7]
                rigid[5] += affineVector[12 * i + 8] * affineVector[12 * i + 8]
                rigid[5] -= 1

                frVector.append(rigid[0])
                frVector.append(rigid[1])
                frVector.append(rigid[2])
                frVector.append(rigid[3])
                frVector.append(rigid[4])
                frVector.append(rigid[5])
            fr = np.array(frVector)

            jOffset = 0
            for i in range(sampler.nodeSize()):
                x = [0] * 9
                x[0] = affineVector[12 * i + 0]
                x[1] = affineVector[12 * i + 1]
                x[2] = affineVector[12 * i + 2]
                x[3] = affineVector[12 * i + 3]
                x[4] = affineVector[12 * i + 4]
                x[5] = affineVector[12 * i + 5]
                x[6] = affineVector[12 * i + 6]
                x[7] = affineVector[12 * i + 7]
                x[8] = affineVector[12 * i + 8]

                JrList.append([jOffset, 12 * i + 0, x[3]])
                JrList.append([jOffset, 12 * i + 1, x[4]])
                JrList.append([jOffset, 12 * i + 2, x[5]])
                JrList.append([jOffset, 12 * i + 3, x[0]])
                JrList.append([jOffset, 12 * i + 4, x[1]])
                JrList.append([jOffset, 12 * i + 5, x[2]])
                jOffset += 1

                JrList.append([jOffset, 12 * i + 3, x[6]])
                JrList.append([jOffset, 12 * i + 4, x[7]])
                JrList.append([jOffset, 12 * i + 5, x[8]])
                JrList.append([jOffset, 12 * i + 6, x[3]])
                JrList.append([jOffset, 12 * i + 7, x[4]])
                JrList.append([jOffset, 12 * i + 8, x[5]])
                jOffset += 1

                JrList.append([jOffset, 12 * i + 0, x[6]])
                JrList.append([jOffset, 12 * i + 1, x[7]])
                JrList.append([jOffset, 12 * i + 2, x[8]])
                JrList.append([jOffset, 12 * i + 6, x[0]])
                JrList.append([jOffset, 12 * i + 7, x[1]])
                JrList.append([jOffset, 12 * i + 8, x[2]])
                jOffset += 1

                JrList.append([jOffset, 12 * i + 0, 2 * x[0]])
                JrList.append([jOffset, 12 * i + 1, 2 * x[1]])
                JrList.append([jOffset, 12 * i + 2, 2 * x[2]])
                jOffset += 1

                JrList.append([jOffset, 12 * i + 3, 2 * x[3]])
                JrList.append([jOffset, 12 * i + 4, 2 * x[4]])
                JrList.append([jOffset, 12 * i + 5, 2 * x[5]])
                jOffset += 1

                JrList.append([jOffset, 12 * i + 6, 2 * x[6]])
                JrList.append([jOffset, 12 * i + 7, 2 * x[7]])
                JrList.append([jOffset, 12 * i + 8, 2 * x[8]])
                jOffset += 1
            
            Jr = util.getSparseMatrixFromList(JrList, (jOffset, 12 * sampler.nodeSize()))
            JrTJr = Jr.T.dot(Jr)
            A0 = controlParams.alphaRigid * JrTJr
            b0 = (controlParams.alphaRigid * JrTJr).dot(affineVector)
            b0 -= (controlParams.alphaRigid * Jr.T).dot(fr)

            bsVector = []
            sOffset = 0

            for idx0 in range(sampler.nodeSize()):
                nodeIdx0 = sampler.getNodeVertexIdx(idx0)
                vh0 = currentNonRigidMesh.vertex_handle(nodeIdx0)
                v0 = currentNonRigidMesh.point(vh0)

                nDic = sampler.getNodeNodeDict(idx0)
                for key in nDic:
                    idx1 = key
                    weight = nDic[key]
                    weightRoot = math.sqrt(weight)
                    nodeIdx1 = sampler.getNodeVertexIdx(idx1)
                    vh1 = currentNonRigidMesh.vertex_handle(nodeIdx1)
                    v1 = currentNonRigidMesh.point(vh1)

                    vec = [0.0] * 3
                    vec[0] = weightRoot * (v1[0] - v0[0])
                    vec[1] = weightRoot * (v1[1] - v0[1])
                    vec[2] = weightRoot * (v1[2] - v0[2])

                    AsList.append([sOffset, 12 * idx0 + 0, vec[0]])
                    AsList.append([sOffset, 12 * idx0 + 3, vec[1]])
                    AsList.append([sOffset, 12 * idx0 + 6, vec[2]])
                    AsList.append([sOffset, 12 * idx0 + 9, weightRoot])
                    AsList.append([sOffset, 12 * idx1 + 9, -weightRoot])
                    sOffset += 1

                    AsList.append([sOffset, 12 * idx0 + 1, vec[0]])
                    AsList.append([sOffset, 12 * idx0 + 4, vec[1]])
                    AsList.append([sOffset, 12 * idx0 + 7, vec[2]])
                    AsList.append([sOffset, 12 * idx0 + 10, weightRoot])
                    AsList.append([sOffset, 12 * idx1 + 10, -weightRoot])
                    sOffset += 1

                    AsList.append([sOffset, 12 * idx0 + 2, vec[0]])
                    AsList.append([sOffset, 12 * idx0 + 5, vec[1]])
                    AsList.append([sOffset, 12 * idx0 + 8, vec[2]])
                    AsList.append([sOffset, 12 * idx0 + 11, weightRoot])
                    AsList.append([sOffset, 12 * idx1 + 11, -weightRoot])
                    sOffset += 1

                    bsVector.append(vec[0])
                    bsVector.append(vec[1])
                    bsVector.append(vec[2])
            
            As = util.getSparseMatrixFromList(AsList, (sOffset, 12 * sampler.nodeSize()))

            bs = np.array(bsVector)

            A1 = controlParams.alphaSmooth * As.T.dot(As)
            b1 = controlParams.alphaSmooth * As.T.dot(bs)
            
            bpVector = []
            pOffset = 0

            for i in range(len(vertexPair)):
                vIdx = vertexPair[i][0]
                cIdx = vertexPair[i][1]

                vh = currentNonRigidMesh.vertex_handle(vIdx)
                ch = currentScanMesh.vertex_handle(cIdx)
                v = currentNonRigidMesh.point(vh)
                c = currentScanMesh.point(ch)

                rhs = [0.0] * 3
                nDic = sampler.getVertexNodeDict(vIdx)
                for key in nDic:
                    nIdx = key
                    weight = nDic[key]
                    nodeIdx = sampler.getNodeVertexIdx(nIdx)
                    nh = currentNonRigidMesh.vertex_handle(nodeIdx)
                    n = currentNonRigidMesh.point(nh)

                    vec = [0.0] * 3
                    vec[0] = weight * (v[0] - n[0])
                    vec[1] = weight * (v[1] - n[1])
                    vec[2] = weight * (v[2] - n[2])

                    ApList.append([pOffset + 0, 12 * nIdx + 0, vec[0]])
                    ApList.append([pOffset + 0, 12 * nIdx + 3, vec[1]])
                    ApList.append([pOffset + 0, 12 * nIdx + 6, vec[2]])
                    ApList.append([pOffset + 0, 12 * nIdx + 9, weight])

                    ApList.append([pOffset + 1, 12 * nIdx + 1, vec[0]])
                    ApList.append([pOffset + 1, 12 * nIdx + 4, vec[1]])
                    ApList.append([pOffset + 1, 12 * nIdx + 7, vec[2]])
                    ApList.append([pOffset + 1, 12 * nIdx + 10, weight])


                    ApList.append([pOffset + 2, 12 * nIdx + 2, vec[0]])
                    ApList.append([pOffset + 2, 12 * nIdx + 5, vec[1]])
                    ApList.append([pOffset + 2, 12 * nIdx + 8, vec[2]])
                    ApList.append([pOffset + 2, 12 * nIdx + 1, weight])

                    rhs[0] -= weight * n[0]
                    rhs[1] -= weight * n[1]
                    rhs[2] -= weight * n[2]
                
                rhs[0] += c[0]
                rhs[1] += c[1]
                rhs[2] += c[2]
                bpVector.append(rhs[0])
                bpVector.append(rhs[1])
                bpVector.append(rhs[2])

                pOffset += 3

            Ap = util.getSparseMatrixFromList(ApList, (pOffset , 12 * sampler.nodeSize()))
            bp = np.array(bpVector)
            A2 = controlParams.alphaPoint * Ap.T.dot(Ap)
            b2 = (controlParams.alphaPoint * Ap.T).dot(bp)

            bqVector = []
            qOffset = 0

            for i in range(len(vertexPair)):
                vIdx = vertexPair[i][0]
                cIdx = vertexPair[i][1]
                vh = currentNonRigidMesh.vertex_handle(vIdx)
                ch = currentScanMesh.vertex_handle(cIdx)
                v = currentNonRigidMesh.point(vh)
                c = currentScanMesh.point(ch)
                cN = currentNonRigidMesh.normal(vh)

                rhs = 0.0

                nDic = sampler.getVertexNodeDict(vIdx)
                for key in nDic:
                    nIdx = key
                    weight = nDic[key]
                    nodeIdx = sampler.getNodeVertexIdx(nIdx)
                    nh = currentNonRigidMesh.vertex_handle(nodeIdx)
                    n = currentNonRigidMesh.point(nh)

                    vec = [0.0] * 3
                    vec[0] = weight * (v[0] - n[0])
                    vec[1] = weight * (v[1] - n[1])
                    vec[2] = weight * (v[2] - n[2])
                    AqList.append([qOffset, 12 * nIdx + 0, cN[0] * vec[0]])
                    AqList.append([qOffset, 12 * nIdx + 1, cN[1] * vec[0]])
                    AqList.append([qOffset, 12 * nIdx + 2, cN[2] * vec[0]])

                    AqList.append([qOffset, 12 * nIdx + 3, cN[0] * vec[1]])
                    AqList.append([qOffset, 12 * nIdx + 4, cN[1] * vec[1]])
                    AqList.append([qOffset, 12 * nIdx + 5, cN[2] * vec[1]])

                    AqList.append([qOffset, 12 * nIdx + 6, cN[0] * vec[2]])
                    AqList.append([qOffset, 12 * nIdx + 7, cN[1] * vec[2]])
                    AqList.append([qOffset, 12 * nIdx + 8, cN[2] * vec[2]])

                    AqList.append([qOffset, 12 * nIdx + 9, cN[0] * weight])
                    AqList.append([qOffset, 12 * nIdx + 10, cN[1] * weight])
                    AqList.append([qOffset, 12 * nIdx + 11, cN[2] * weight])

                    rhs -= cN[0] * weight * n[0]
                    rhs -= cN[1] * weight * n[1]
                    rhs -= cN[2] * weight * n[2]

                rhs += cN[0] * c[0]
                rhs += cN[1] * c[1]
                rhs += cN[2] * c[2]

                bqVector.append(rhs)
                qOffset+=1
            
            Aq = util.getSparseMatrixFromList(AqList, (qOffset, 12 * sampler.nodeSize()))
            bq = np.array(bqVector)

            A3 = controlParams.alphaPlane * Aq.T.dot(Aq)
            b3 = controlParams.alphaPlane * Aq.T.dot(bq)

            A = A0 + A1 + A2 + A3
            b = b0 + b1 + b2 + b3
            nextAffineVector = LA.solve(A.toarray(),b)
            
            rigidVector = fr + Jr.dot((nextAffineVector - affineVector))
            rigidEnergy = config.ALPHARIGID * rigidVector.T.dot(rigidVector)

            smoothVector = As.dot(nextAffineVector) - bs
            smoothEnergy = config.ALPHASMOOTH * smoothVector.T.dot(smoothVector)

            pointVector = Ap.dot(nextAffineVector) - bp
            pointEnergy = config.ALPHAPOINT * pointVector.T.dot(pointVector)

            planeVector = Aq.dot(nextAffineVector) - bq
            planeEnergy = config.ALPHAPLANE * planeVector.T.dot(planeVector)

            gaussNewtonEnergy = rigidEnergy + smoothEnergy + pointEnergy + planeEnergy
            gaussNewtonEnergyChange = abs(gaussNewtonEnergy - gaussNewtonEnergyPre)/(gaussNewtonEnergyPre + 1)

            affineVector[:] = nextAffineVector[:]   
            if gaussNewtonEnergyChange < 1e-6 or gaussNewtonEnergyIterCnt >= 50:
                break
            
            gaussNewtonEnergyPre = gaussNewtonEnergy

            gaussNewtonEnergyIterCnt += 1
        
        return gaussNewtonEnergy

    def accumulateAffine(self, affineVector, accumulatedAffineVector, nodeSize):
        for nodeIdx in range(nodeSize):
            rotMat = np.zeros((3,3))
            shiftVec = np.zeros((3))
            accumRotMat = np.zeros((3,3))
            accumShiftVec = np.zeros((3))
            rotMat[0,0] = affineVector[12 * nodeIdx + 0]
            rotMat[1,0] = affineVector[12 * nodeIdx + 1]
            rotMat[2,0] = affineVector[12 * nodeIdx + 2]
            rotMat[0,1] = affineVector[12 * nodeIdx + 3]
            rotMat[1,1] = affineVector[12 * nodeIdx + 4]
            rotMat[2,1] = affineVector[12 * nodeIdx + 5]
            rotMat[0,2] = affineVector[12 * nodeIdx + 6]
            rotMat[1,2] = affineVector[12 * nodeIdx + 7]
            rotMat[2,2] = affineVector[12 * nodeIdx + 8]

            shiftVec[0] = affineVector[12 * nodeIdx + 9]
            shiftVec[1] = affineVector[12 * nodeIdx + 10]
            shiftVec[2] = affineVector[12 * nodeIdx + 11]

            accumRotMat[0,0] = accumulatedAffineVector[12 * nodeIdx + 0]
            accumRotMat[1,0] = accumulatedAffineVector[12 * nodeIdx + 1]
            accumRotMat[2,0] = accumulatedAffineVector[12 * nodeIdx + 2]
            accumRotMat[0,1] = accumulatedAffineVector[12 * nodeIdx + 3]
            accumRotMat[1,1] = accumulatedAffineVector[12 * nodeIdx + 4]
            accumRotMat[2,1] = accumulatedAffineVector[12 * nodeIdx + 5]
            accumRotMat[0,2] = accumulatedAffineVector[12 * nodeIdx + 6]
            accumRotMat[1,2] = accumulatedAffineVector[12 * nodeIdx + 7]
            accumRotMat[2,2] = accumulatedAffineVector[12 * nodeIdx + 8]

            accumShiftVec[0] = accumulatedAffineVector[12 * nodeIdx + 9]
            accumShiftVec[1] = accumulatedAffineVector[12 * nodeIdx + 10]
            accumShiftVec[2] = accumulatedAffineVector[12 * nodeIdx + 11]

            accumRotMat = rotMat.dot(accumRotMat)
            accumShiftVec = shiftVec + accumShiftVec

            accumulatedAffineVector[12 * nodeIdx + 0] = accumRotMat[0,0]
            accumulatedAffineVector[12 * nodeIdx + 1] = accumRotMat[1,0]
            accumulatedAffineVector[12 * nodeIdx + 2] = accumRotMat[2,0]
            accumulatedAffineVector[12 * nodeIdx + 3] = accumRotMat[0,1]
            accumulatedAffineVector[12 * nodeIdx + 4] = accumRotMat[1,1]
            accumulatedAffineVector[12 * nodeIdx + 5] = accumRotMat[2,1]
            accumulatedAffineVector[12 * nodeIdx + 6] = accumRotMat[0,2]
            accumulatedAffineVector[12 * nodeIdx + 7] = accumRotMat[1,2]
            accumulatedAffineVector[12 * nodeIdx + 8] = accumRotMat[2,2]

            accumulatedAffineVector[12 * nodeIdx + 9] = accumShiftVec[0]
            accumulatedAffineVector[12 * nodeIdx + 10] = accumShiftVec[1]
            accumulatedAffineVector[12 * nodeIdx + 11] = accumShiftVec[2]
    
    def updateNonRigidMesh(self, currentNonRigidMesh, affineVector, sampler):
        nodeCoords = [openmesh.TriMesh.Point()] * sampler.nodeSize()
        for i in range(sampler.nodeSize()):
            nodeIdx = sampler.getNodeVertexIdx(i)
            nh = currentNonRigidMesh.vertex_handle(nodeIdx)
            vNode = currentNonRigidMesh.point(nh)
            nodeCoords[i] = vNode
        
        for i in range(self.templateVertexNum):
            vh = currentNonRigidMesh.vertex_handle(i)
            v = currentNonRigidMesh.point(vh)
            vUpdated = np.zeros((3))

            nDic = sampler.getVertexNodeDict(i)
            for key in nDic:
                idx = key
                weight = nDic[key]
                vNode = nodeCoords[idx]
                Aj = np.eye(3)
                bj = np.zeros((3))
                vi = np.zeros((3))
                uj = np.zeros((3))

                Aj[0,0] = affineVector[12 * idx + 0]
                Aj[1,0] = affineVector[12 * idx + 1]
                Aj[2,0] = affineVector[12 * idx + 2]
                Aj[0,1] = affineVector[12 * idx + 3]
                Aj[1,1] = affineVector[12 * idx + 4]
                Aj[2,1] = affineVector[12 * idx + 5]
                Aj[0,2] = affineVector[12 * idx + 6]
                Aj[1,2] = affineVector[12 * idx + 7]
                Aj[2,2] = affineVector[12 * idx + 8]

                bj[0] = affineVector[12 * idx + 9]
                bj[1] = affineVector[12 * idx + 10]
                bj[2] = affineVector[12 * idx + 11]

                vi[0] = v[0]
                vi[1] = v[1]
                vi[2] = v[2]

                uj[0] = vNode[0]
                uj[1] = vNode[1]
                uj[2] = vNode[2]

                vUpdated += weight * (Aj.dot(vi - uj) + uj + bj)

            currentNonRigidMesh.set_point(vh, openmesh.TriMesh.Point(vUpdated[0], vUpdated[1], vUpdated[2]))
 
