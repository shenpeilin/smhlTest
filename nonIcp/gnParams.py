import config
class GnParams:
    def __init__(self):
        self.alphaRigid = config.ALPHARIGID
        self.alphaSmooth = config.ALPHASMOOTH
        self.alphaPoint = config.ALPHAPOINT
        self.alphaPlane = config.ALPHAPLANE
    
    def relaxParams(self):
        self.alphaRigid /= 2.0
        self.alphaSmooth /= 2.0

    @property
    def alphaRigid(self):
        return self.alphaRigid

    @property
    def alphaSmooth(self):
        return self.alphaSmooth

    @property
    def alphaPlane(self):
        return self.alphaPlane

    @property
    def alphaPoint(self):
        return self.alphaPoint