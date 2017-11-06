from geodesic import Geodesic
import numpy as np
from sets import Set
def axisValue(index , mesh , axis):
    vh = mesh.vertex_handle(index)
    v = mesh.point(vh)
    return v[axis]

class NodeSampler:
    def __init__(self):
        self.averageEdgeLen = 0
        self.geoDistContainer = []
        self.nodeContainer = []

    def sample(self , mesh , sampleRadius , axis):
        self.meshVertexNum = mesh.n_vertices()
        self.meshEdgeNum = mesh.n_edges()
        for i in range(0 , self.meshEdgeNum):
            eh = mesh.edge_handle(i)
            edgeLen = mesh.calc_edge_length(eh)
            self.averageEdgeLen += edgeLen
        self.averageEdgeLen /= self.meshEdgeNum
        self.sampleRadius = sampleRadius * self.averageEdgeLen
        self.vertexReorderedAlongAxis = range(0,self.meshVertexNum)
        self.vertexReorderedAlongAxis.sort(key = lambda index: axisValue(index , mesh , axis))
        firstVertexIdx = self.vertexReorderedAlongAxis[0]
        pGeodesic = Geodesic(mesh)
        geoDistVector = pGeodesic.seed(firstVertexIdx)
        self.geoDistContainer.append(geoDistVector)
        self.nodeContainer.append([0,firstVertexIdx])

        for vertexIdx in self.vertexReorderedAlongAxis:
            IsNode = True
            for distVector in self.geoDistContainer:
                dist = distVector[vertexIdx]
                if(dist < self.sampleRadius):
                    IsNode = False
                    break
            
            if IsNode:
                self.geoDistContainer.append(pGeodesic.seed(vertexIdx))
                self.nodeContainer.append([len(self.geoDistContainer) - 1 , vertexIdx])

    def nodeSize(self):
        return len(self.nodeContainer)

    def constructGraph(self):
        self.vertexGraph = [dict() for x in range(self.meshVertexNum)]
        for vertexIdx in range(0, self.meshVertexNum):
            totalWeight = 0.0
            for nodeIdx in range(len(self.geoDistContainer)):
                dist = self.geoDistContainer[nodeIdx][vertexIdx]
                weight = max(0.0, (1.0 - (dist/(2.0 * self.sampleRadius))**2)**3)

                if weight > 0:
                    self.vertexGraph[vertexIdx][nodeIdx] = weight
                    totalWeight += weight
            for key in self.vertexGraph[vertexIdx]:
                self.vertexGraph[vertexIdx][key] /= totalWeight

        nodeTopo = [Set() for x in range(len(self.nodeContainer))]
        for vertexIdx in range(0, self.meshVertexNum):
            for nodeIdx0 in self.vertexGraph[vertexIdx]:
                for nodeIdx1 in self.vertexGraph[vertexIdx]:
                    if nodeIdx0 != nodeIdx1:
                        nodeTopo[nodeIdx0].add(nodeIdx1)

        self.nodeGraph = [dict() for x in range(len(self.nodeContainer))]
        for nodeIdx in range(len(self.nodeContainer)):
            totalWeight = 0.0
            for eachNeighbor in nodeTopo[nodeIdx]:
                neighborNodeVertexIdx = self.nodeContainer[eachNeighbor][1]
                dist = self.geoDistContainer[nodeIdx][neighborNodeVertexIdx]
                weight = max(0.0, (1.0 - (dist/(2.0 * self.sampleRadius))**2)**3)
                if weight > 0:
                    self.nodeGraph[nodeIdx][eachNeighbor] = weight
                    totalWeight += weight
            for key in self.nodeGraph[nodeIdx]:
                self.nodeGraph[nodeIdx][key] /= totalWeight

    def drawNodes(self, mesh, nodeFile):
        with open( nodeFile, 'w') as fp:
            fp.write('COFF\n')
            fp.write('%d 0 0\n' % (len(self.nodeContainer)))
            for nodeIdx in range(0, len(self.nodeContainer)):
                vIdx = self.nodeContainer[nodeIdx][1]
                vh = mesh.vertex_handle(vIdx)
                v = mesh.point(vh)
                fp.write('%f %f %f 255 0 0 255\n' % (v[0], v[1], v[2]))

        print 'nodeMesh saved to', nodeFile

    def drawNodeGraph(self, mesh, nodeFile):
        eps = 1e-6

        faceNum = 0
        for node in self.nodeGraph:
            faceNum += len(node)
        with open( nodeFile, 'w') as fp:
            fp.write('COFF\n')
            fp.write('%d %d 0\n' % (3 * faceNum, faceNum))
            for nodeIdx0 in range(0, len(self.nodeContainer)):
                vIdx0 = self.nodeContainer[nodeIdx0][1]
                vh0 = mesh.vertex_handle(vIdx0)
                v0 = mesh.point(vh0)
                for nodeIdx1 in self.nodeGraph[nodeIdx0]:
                    vIdx1 = self.nodeContainer[nodeIdx1][1]
                    vh1 = mesh.vertex_handle(vIdx1)
                    v1 = mesh.point(vh1)
                    fp.write('%f %f %f\n' % (v0[0], v0[1], v0[2]))
                    fp.write('%f %f %f\n' % (v0[0]+eps, v0[1]+eps, v0[2]+eps))
                    fp.write('%f %f %f\n' % (v1[0], v1[1], v1[2]))

            for faceId in range(0, faceNum):
                fp.write('3 %d %d %d\n' % (3 * faceId, 3 * faceId + 1, 3 * faceId + 2))
        print 'nodeGraph saved to', nodeFile

    def getNodeVertexIdx(self, nodeIdx):
        return self.nodeContainer[nodeIdx][1]

    def getNodeNodeDict(self, nodeIdx):
        return self.nodeGraph[nodeIdx]

    def getVertexNodeDict(self, nodeIdx):
        return self.vertexGraph[nodeIdx]