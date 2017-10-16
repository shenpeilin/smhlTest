'''
Copyright 2015 Matthew Loper, Naureen Mahmood and the Max Planck Gesellschaft.  All rights reserved.
This software is provided for research purposes only.
By using this software you agree to the terms of the SMPL Model license here http://smpl.is.tue.mpg.de/license

More information about SMPL is available here http://smpl.is.tue.mpg.
For comments or questions, please email us at: smpl@tuebingen.mpg.de


About this file:
================
This file defines the basic skinning modules for the SMPL loader which 
defines the effect of bones and blendshapes on the vertices of the template mesh.

Modules included:
- verts_decorated: 
  creates an instance of the SMPL model which inherits model attributes from another 
  SMPL model.
- verts_core: [overloaded function inherited by lbs.verts_core]
  computes the blending of joint-influences for each vertex based on type of skinning

'''

import chumpy
import lbs
import scipy.sparse as sp
from chumpy.ch import MatVecMult

def ischumpy(x): return hasattr(x, 'dterms')
def verts_core(pose, v, J, weights, kintree_table, bs_style, want_Jtr=False, xp=chumpy):
    
    if xp == chumpy:
        assert(hasattr(pose, 'dterms'))
        assert(hasattr(v, 'dterms'))
        assert(hasattr(J, 'dterms'))
        assert(hasattr(weights, 'dterms'))
     
    assert(bs_style=='lbs')
    result = lbs.verts_core(pose, v, J, weights, kintree_table, want_Jtr, xp)

    return result
