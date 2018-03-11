from serialization import loadObj
from serialization import loadTemplate, getObjList
from TemplateRegistor import TemplateRigistor as temReg
from multiprocessing import Pool

def register(objFileName):
    temp = temReg()
    temp.setTemplate(loadTemplate())
    temp.setScanModel(loadObj(objFileName))
    temp.alignTemplate()
    temp.areaMatch()
    temp.transTemplate()
    # temp.renderTransedTemplate()
    temp.setScanMesh(objFileName)
    temp.saveResult()
    temp.setTemplateMesh()
    temp.runIcp()

objList = ['./HandScan2/FuXiaoMing_Day2_2.obj', './HandScan2/FuXiaoMing_Day2_1.obj']
p = Pool()
p.map(register, objList)

# for objFileName in objList:
#     temp = temReg()
#     temp.setTemplate(loadTemplate())
#     temp.setScanModel(loadObj(objFileName))
#     temp.alignTemplate()
#     temp.areaMatch()
#     temp.transTemplate()
#     temp.renderTransedTemplate()
#     temp.saveResult()
#     temp.setTemplateMesh('./output.obj')
#     temp.setScanMesh(objFileName)
#     temp.runIcp()