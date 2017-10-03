import cPickle as pickle
import numpy as np
import chumpy as ch
import xml.etree.ElementTree as ET

def readyArgument(dd):
    for s in ['v','J','pose','weights']:
        if (s in dd) and not hasattr(dd[s] , 'dterms'):
            dd[s] = ch.array(dd[s])
    return dd

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

    ppFname = fname.replace("obj","pp")
    plist = []
    tree = ET.parse(ppFname)
    root = tree.getroot()
    for i in range(1,45):
        point = [float(root[i].get('x')),float(root[i].get('y')),float(root[i].get('z'))]
        plist.append(point)
    dd["pp"] = np.array(plist)
    return readyArgument(dd)

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
    dd["pose"] = np.zeros(dd["kintree_table"].shape[1]*3)
    return readyArgument(dd)