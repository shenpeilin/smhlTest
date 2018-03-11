import cPickle as pickle
from serialization import loadTemplate,getObjList
import chumpy as ch

tmp = loadTemplate()
fileList = getObjList()
with open('shapList.pkl', 'rb') as f:
    shapeList = pickle.load(f)

for i in range(len(shapeList)):
    vArray = ch.array(shapeList[i])
    vArray = vArray.reshape((-1,3))
    idx = fileList[i].rfind('/')
    nonIcpFileName = fileList[i][idx+1:].replace(".obj", "Normal.obj")
    outmesh_path = './normalize/'+nonIcpFileName
    with open( outmesh_path, 'w') as fp: 
        for v in vArray.r:
            fp.write( 'v %f %f %f\n' % ( v[0], v[1], v[2]) )

        for f in tmp['f']+1: # Faces are 1-based, not 0-based in obj files
            fp.write( 'f %d %d %d\n' %  (f[0], f[1], f[2]) )

    ## Print message
    print '..Output mesh saved to: ', outmesh_path