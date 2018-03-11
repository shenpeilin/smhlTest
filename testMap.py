import cPickle as pickle
import numpy as np
import chumpy as ch
from serialization import loadTemplate,loadObj,getObjList

fileList = getObjList()
tmp = loadTemplate()
vertexNum = tmp['v'].shape[0]
for fileName in fileList:
    model = loadObj(fileName)
    mapFilename = fileName.replace(".obj", "Map.pkl")
    with open(mapFilename, 'rb') as f:
        indexMap = pickle.load(f)
    mapArray = np.zeros((vertexNum,) , np.int)
    for indexTur in indexMap:
        mapArray[indexTur[0]] = indexTur[1]
    result = ch.array(model['v'][mapArray])
    idx = fileName.rfind('/')
    nonIcpFileName = fileName[idx+1:].replace(".obj", "map.obj")
    outmesh_path = './mapobj/'+nonIcpFileName
    with open( outmesh_path, 'w') as fp: 
        for v in result.r:
            fp.write( 'v %f %f %f\n' % ( v[0], v[1], v[2]) )

        for f in tmp['f']+1: # Faces are 1-based, not 0-based in obj files
            fp.write( 'f %d %d %d\n' %  (f[0], f[1], f[2]) )

    ## Print message
    print '..Output mesh saved to: ', outmesh_path