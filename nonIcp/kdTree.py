import openmesh
import sys
from boxNode import BoxNode
import numpy as np
from numpy import linalg as LA
import util
import math

class KDTree:
    def __init__(self, mesh):
        self.mesh = mesh
        self.vertexNum = mesh.n_vertices()

        M = 1
        tmp = self.vertexNum
        while tmp > 0:
            M <<= 1
            tmp >>=1

        self.boxNum  = 2 * self.vertexNum - (M >> 1)
        self.boxNum = min(self.boxNum , M)
        coords = [0] * 3 * self.vertexNum
        self.boxes = [None]*self.boxNum
        self.indices = np.zeros((self.vertexNum),np.int64)
        for i in range(mesh.n_vertices()):
            vh = mesh.vertex_handle(i)
            v = mesh.point(vh)
            self.indices[i] = vh.idx()

            coords[vh.idx() + self.vertexNum*0] = v[0]
            coords[vh.idx() + self.vertexNum*1] = v[1]
            coords[vh.idx() + self.vertexNum*2] = v[2]
        self.BIG = sys.float_info.max
        lowCorner = openmesh.TriMesh.Point(-self.BIG, -self.BIG, -self.BIG)
        highCorner = openmesh.TriMesh.Point(self.BIG, self.BIG, self.BIG)
        self.boxes[0] = BoxNode(lowCorner, highCorner, 0, 0, 0, 0, self.vertexNum-1)
        boxIdx = 0
        momBoxIdx = 0
        currentDim = 0
        taskBox = []
        taskDim = []
        taskBox.append(0)
        taskDim.append(0)

        while taskBox and  taskDim:
            momBoxIdx = taskBox.pop()
            currentDim = taskDim.pop()

            pLeftIdx = self.boxes[momBoxIdx].pLeftIdx
            pRightIdx = self.boxes[momBoxIdx].pRightIdx

            coordsArray = coords[currentDim * self.vertexNum:]
            indexArray = self.indices[pLeftIdx:]
            arrayLen = pRightIdx - pLeftIdx + 1
            partitionLoc = (arrayLen - 1)/2

            self._partition(partitionLoc, indexArray, coordsArray, arrayLen)

            lowCorner = self.boxes[momBoxIdx].lowCorner
            highCorner = self.boxes[momBoxIdx].highCorner

            lowCorner[currentDim] = coords[currentDim * self.vertexNum + indexArray[partitionLoc]]
            highCorner[currentDim] = lowCorner[currentDim]

            boxIdx += 1
            self.boxes[boxIdx] = BoxNode(self.boxes[momBoxIdx].lowCorner,
                                            highCorner,
                                            momBoxIdx,
                                            0, 0,
                                            pLeftIdx,
                                            pLeftIdx + partitionLoc)
            boxIdx += 1
            self.boxes[boxIdx] = BoxNode(lowCorner,
                                            self.boxes[momBoxIdx].highCorner,
                                            momBoxIdx,
                                            0, 0,
                                            pLeftIdx + partitionLoc + 1,
                                            pRightIdx)
            self.boxes[momBoxIdx].leftSonIdx = boxIdx - 1
            self.boxes[momBoxIdx].rightSonIdx = boxIdx

            if(partitionLoc > 1):
                taskBox.append(boxIdx - 1)
                taskDim.append((currentDim + 1) % 3)
            
            if(arrayLen - partitionLoc > 3):
                taskBox.append(boxIdx)
                taskDim.append((currentDim + 1) % 3)
    
    def _partition(self, k, indexArray, valueArray, N):
        assert k <= N and N > 0

        l = 0
        r = N - 1

        while True:
            if(r <= l + 1):
                if(r == l+1 and valueArray[indexArray[r]] < valueArray[indexArray[l]]):
                    indexArray[l], indexArray[r] = indexArray[r], indexArray[l]
                return indexArray[k]
            
            mid = (l + r) >> 1
            indexArray[l+1], indexArray[mid] = indexArray[mid], indexArray[l+1]
            if(valueArray[indexArray[l]] > valueArray[indexArray[r]]):
                indexArray[l], indexArray[r] = indexArray[r], indexArray[l]

            if(valueArray[indexArray[l + 1]] > valueArray[indexArray[r]]):
                indexArray[l + 1], indexArray[r] = indexArray[r], indexArray[l + 1]

            if(valueArray[indexArray[l]] > valueArray[indexArray[l+1]]):
                indexArray[l], indexArray[l+1] = indexArray[l+1], indexArray[l]

            pivot = valueArray[indexArray[l + 1]]
            i = l+2
            j = r-1

            while True:
                while valueArray[indexArray[i]] < pivot:
                    i += 1
                while valueArray[indexArray[j]] > pivot:
                    j -= 1
                if j < i:
                    break

                indexArray[i], indexArray[j] = indexArray[j], indexArray[i]
            
            indexArray[l+1], indexArray[j] = indexArray[j], indexArray[l+1]
            
            if j == k:
                return indexArray[k]

            if j > k:
                r = j-1
            
            if j < k:
                l = j+1 

    def _locate(self, p):
        boxIdx = 0
        currentDim = 0
        while self.boxes[boxIdx] and self.boxes[boxIdx].leftSonIdx:
            leftSonIdx = self.boxes[boxIdx].leftSonIdx
            rightSonIdx = self.boxes[boxIdx].rightSonIdx
            leftSon = self.boxes[leftSonIdx]
            cutPlane = leftSon.highCorner[currentDim]

            if(p[currentDim] <= cutPlane):
                boxIdx = leftSonIdx
            else:
                boxIdx = rightSonIdx

            currentDim = (currentDim + 1)%3

        return boxIdx

    def _dist2Point(self, pointIdx, p):
        vh = self.mesh.vertex_handle(pointIdx)
        v = self.mesh.point(vh)
        e = util.Vec3DToNp(v - p)
        return LA.norm(e)

    def _dist2Box(self, boxIdx, p):
        thisBox = self.boxes[boxIdx]
        dist = 0.0

        for i in range(3):
            if(p[i] < thisBox.lowCorner[i]):
                dist += (p[i] - thisBox.lowCorner[i])**2
            
            if(p[i] > thisBox.highCorner[i]):
                dist += (p[i] - thisBox.highCorner[i])**2
        
        return math.sqrt(dist)

    def nearest(self, queryPoint):
        hostBox = self._locate(queryPoint)
        pLeftIdx = self.boxes[hostBox].pLeftIdx
        pRightIdx = self.boxes[hostBox].pRightIdx

        nDist = self.BIG
        nIdx = -1

        for i in range(pLeftIdx, pRightIdx + 1):
            dist = self._dist2Point(self.indices[i], queryPoint)

            if(dist < nDist):
                nDist = dist
                nIdx = self.indices[i]

        taskBox = []
        taskBox.append(0)

        while taskBox:
            boxIdx = taskBox.pop()

            if self._dist2Box(boxIdx, queryPoint) < nDist:
                if self.boxes[boxIdx].leftSonIdx:
                    taskBox.append(self.boxes[boxIdx].leftSonIdx)
                    taskBox.append(self.boxes[boxIdx].rightSonIdx)

                else:
                    pLeftIdx = self.boxes[boxIdx].pLeftIdx
                    pRightIdx = self.boxes[boxIdx].pRightIdx

                    for i in range (pLeftIdx, pRightIdx+1):
                        dist = self._dist2Point(self.indices[i], queryPoint)

                        if dist < nDist:
                            nDist = dist
                            nIdx = self.indices[i]
                
        if nIdx == -1:
            return self.mesh.InvalidVertexHandle

        else:
            return self.mesh.vertex_handle(nIdx)


