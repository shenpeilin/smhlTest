import cPickle as pickle
import numpy as np
import chumpy as ch
import xml.etree.ElementTree as ET
from posemapper import getMatches
import config

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
    for i in range(1,config.NUM_OF_POINTS+1):
        point = [float(root[i].get('x')),float(root[i].get('y')),float(root[i].get('z'))]
        plist.append(point)
    dd["pp"] = np.array(plist)
    # matches = getMatches(np.array(dd['v'],np.float32),np.array(dd['pp'],np.float32))
    # dd['pv'] = []
    # for i in range(0,config.NUM_OF_POINTS):
    #     dd['pv'].append([])
    # for i in range(0,dd['v'].shape[0]):
    #     dd['pv'][matches[i][0].trainIdx].append(i)
    areaPointList = []
    areaMatches = getMatches(np.array(dd['v'],np.float32),np.array(dd['pp'],np.float32),1)
    for i in range(0,config.NUM_OF_POINTS):
        areaPointList.append([])

    for i in range(0,dd['v'].shape[0]):
        areaPointList[areaMatches[i][0].trainIdx].append(i)
    matches = getMatches(np.array(dd['pp'],np.float32),np.array(dd['v'],np.float32),1)
    dd['pl'] = []
    for i in range(0,config.NUM_OF_POINTS):
        dd['pl'].append(matches[i][0].trainIdx)
    dd['pv'] = []
    for i in range(0,config.NUM_OF_POINTS):
        pointMatches = getMatches(np.array(dd['pp'][i],np.float32)
        ,np.array(dd['v'][areaPointList[i]],np.float32)
        ,config.POINT_IN_AREA)
        dd['pv'].append([])
        for j in range(0,config.POINT_IN_AREA):
            dd['pv'][i].append(areaPointList[i][pointMatches[0][j].trainIdx])
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