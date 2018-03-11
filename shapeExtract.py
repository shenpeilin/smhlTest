from serialization import getObjList, loadTemplate, loadObj
from TemplateGenerator import TemplateGenerator as tempGen
from multiprocessing import Pool, Value
from sklearn.decomposition import PCA
import numpy as np
import chumpy as ch
import cPickle as pickle

num = Value('d', 0)
def shapeExtract(objFileName):
    num.value = num.value+1
    print num.value
    temp = tempGen()
    temp.setTemplate(loadTemplate())
    temp.setMapArray(objFileName)
    temp.setScanModel(loadObj(objFileName))
    temp.alignTemplate()
    temp.getJointRegressor()
    temp.poseMatch()
    return temp.normalizeScan()
    # temp.renderTransedTemplate()
    # temp.saveResult()

objList = getObjList()
p = Pool(2)
shapeList = p.map(shapeExtract, objList)
temp = tempGen()
t = loadTemplate()
shapeList.append(np.reshape(t['v'],(t['v'].shape[0]*3)))
temp.setTemplate(t)
Jre = temp.getJointRegressor()
tmp = {}
allShape = np.array(shapeList)
pca = PCA()
tmp['shape'] = pca.fit_transform(allShape.T).T
tmp['v'] = np.mean(allShape, axis=0)
tmp['f'] = t['f']
tmp['jRegressor'] = Jre
tmp['weights'] = t['weights']
tmp['kintree_table'] = t['kintree_table']
tmp['pp'] = t['pp']
tmp['pl'] = t['pl']
result = ch.array(np.reshape(tmp['v'],(-1,3)))
temp.saveResult(result)
tmpFileName = 'template.pkl'
shapeFileName = 'shapList.pkl'
with open(tmpFileName, 'w') as f:
    pickle.dump(tmp, f)

with open(shapeFileName, 'w') as f:
    pickle.dump(shapeList, f)
