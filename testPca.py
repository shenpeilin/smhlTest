from sklearn.decomposition import PCA
import numpy as np
import chumpy as ch
import cPickle as pickle
from TemplateGenerator import TemplateGenerator as tempGen
from serialization import getObjList, loadTemplate, loadObj
with open('shapList.pkl', 'rb') as f:
    shapeList = pickle.load(f)
tmp = np.mean(shapeList,axis=0)
allShape = np.array(shapeList)
pca = PCA()
shape = pca.fit_transform(allShape.T).T
tmp = tmp + 0.1*shape[0]
tmp = tmp.reshape((-1,3))
temp = tempGen()
temp.setTemplate(loadTemplate())
temp.saveResult(ch.array(tmp))