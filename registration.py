from serialization import loadObj
from serialization import loadTemplate, getObjList
from TemplateRegistor import TemplateRigistor as temReg

objList = getObjList()
for objFileName in objList:
    temp = temReg()
    temp.setTemplate(loadTemplate())
    temp.setScanModel(loadObj(objFileName))
    temp.alignTemplate()
    temp.areaMatch()
    temp.transTemplate()
    # temp.renderTransedTemplate()
    temp.saveResult()
    temp.setTemplateMesh('./output.obj')
    temp.setScanMesh(objFileName)
    temp.runIcp()