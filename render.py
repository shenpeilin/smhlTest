import numpy as np
from opendr.renderer import ColoredRenderer
from opendr.lighting import LambertianPointLight
from opendr.camera import ProjectPoints
import chumpy as ch
import numpy as np
def renderObj(v,f):
    rn = ColoredRenderer()

    ## Assign attributes to renderer
    w, h = (640, 480)

    rn.camera = ProjectPoints(v=v, rt=np.zeros(3), t=np.array([0, 0, 2.]), f=np.array([w,w])/2., c=np.array([w,h])/2., k=np.zeros(5))
    rn.frustum = {'near': 1., 'far': 10., 'width': w, 'height': h}
    rn.set(v=v, f=f, bgcolor=np.zeros(3))

    ## Construct point light source
    rn.vc = LambertianPointLight(
        f=f,
        v=v,
        num_verts=len(v),
        light_pos=np.array([-1000,-1000,-2000]),
        vc=np.ones_like(v)*.9,
        light_color=np.array([1., 1., 1.]))

    ## Show it using OpenCV
    import cv2
    cv2.imshow('render_SMPL', rn.r)
    print ('..Print any key while on the display window')
    cv2.waitKey(0)
    cv2.destroyAllWindows()