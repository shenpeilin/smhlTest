from serialization import getObjList, loadTemplate, loadObj
from TemplateGenerator import TemplateGenerator as tempGen

objList = getObjList()
for objFileName in objList:
    temp = tempGen()
    temp.setTemplate(loadTemplate())
    temp.setMapArray(objFileName)
    temp.setScanModel(loadObj(objFileName))
    temp.setTemplateMesh()
    temp.alignTemplate()
    temp.getJointRegressor()
    temp.poseMatch()
    temp.normalizeScan()
    # temp.renderTransedTemplate()
    # temp.saveResult()
temp = tempGen()
print len(temp.vList)
temp.setTemplate(loadTemplate())
temp.getJointRegressor()
temp.saveGeneratedTemp()