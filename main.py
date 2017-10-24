from serialization import loadObj
from serialization import loadTemplate
from TemplateGenerator import TemplateGenerator as temGen 

temp = temGen()
temp.setTemplate(loadTemplate())
temp.setScanModel(loadObj('./HandScan/ZhangYueYi3.obj'))
temp.alignTemplate()
temp.areaMatch()
temp.transTemplate()
temp.renderTransedTemplate()
temp.saveResult()