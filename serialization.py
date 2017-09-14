import cPickle as pickle
import numpy as np

def loadObj(fname):
    dd = {}
    vlist = []
    flist = []
    with open(fname) as objFile:
        for line in objFile:
            strArray = line.split(" ")
            if(strArray[0]=="v"):
                point=[float(strArray[1]), float(strArray[2]), float(strArray[3])]
                vlist.append(point)
            
            elif(strArray[0]=="f"):
                face=[int(strArray[1].split('''/''')[0])-1, int(strArray[2].split('''/''')[0])-1, int(strArray[3].split('''/''')[0])-1]
                flist.append(face)
    dd["v"]=np.array(vlist)
    dd["v"]=dd["v"]
    dd["f"]=np.array(flist)
    return dd
