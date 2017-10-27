from geodesic import Geodesic
import numpy as np
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