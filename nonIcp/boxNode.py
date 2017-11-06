class BoxNode:
    def __init__(self, lowCorner, highCorner, momIdx, leftSonIdx, rightSonIdx, pLeftIdx, pRightIdx):
        self._lowCorner = lowCorner
        self._highCorner = highCorner
        self._momIdx = momIdx
        self._leftSonIdx = leftSonIdx
        self._rightSonIdx = rightSonIdx
        self._pLeftIdx = pLeftIdx
        self._pRightIdx = pRightIdx
    @property
    def lowCorner(self):
        return self._lowCorner
    
    @property
    def highCorner(self):
        return self._highCorner

    @property
    def momIdx(self):
        return self._momIdx

    @property
    def leftSonIdx(self):
        return self._leftSonIdx

    @property
    def rightSonIdx(self):
        return self._rightSonIdx

    @property
    def pLeftIdx(self):
        return self._pLeftIdx

    @property
    def pRightIdx(self):
        return self._pRightIdx