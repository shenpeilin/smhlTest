from TemplateGenerator2 import TemplateGenerator2 as tempGen2
from serialization import getObjList
import cPickle as pickle

tempGen = tempGen2()
tempGen.loadTemplate()
fileName = './HandScan/ZhangYueYi3.obj'
print fileName
tempGen.setScanModel(fileName)
tempGen.alignTemplate()
tempGen.poseMatch()
tempGen.transTemplate()