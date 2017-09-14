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
    dd["f"]=np.array(flist)
    return dd

def loadTemplate():
    dd = {}
    weights = []
    kintree_table = []
    J = []
    dd = loadObj('./template/template.obj')

    with open('./template/newskel.skel') as skelObj:
        for line in skelObj:
            strArray = line.split(" ")
            tree = [int(strArray[4]), int(strArray[0])]
            kintree_table.append(tree)
            point = [float(strArray[1]), float(strArray[2]), float(strArray[3])]
            J.append(point)
    dd["kintree_table"] = np.array(kintree_table).T
    dd["J"] = np.array(J)
    
    with open('./template/attachment.out') as weightFile:
        for line in weightFile:
            strArray = line.split(" ")
            w = []
            for i in range(0,15):
                w.append(float(strArray[i]))
            w.insert(0,1-sum(w))
            weights.append(w)
    dd["weights"] = np.array(weights)
    return dd