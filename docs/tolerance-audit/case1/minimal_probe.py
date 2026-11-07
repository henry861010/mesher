import numpy as np
from mesher.mesh2d.generators.holed_rectilinear import generate_rectilinear_mesh_optimal
from mesher.mesh2d.adapter.layer._planning.front import _outline_o_directions

def box(x0,y0,x1,y1):
    return np.array([[x0,y0],[x1,y0],[x1,y1],[x0,y1]],float)

result=generate_rectilinear_mesh_optimal(1,box(0,0,5,5),
    holes=[box(1,1,np.nextafter(2.,np.inf),2),box(1,3,2,4)],
    coordinate_tolerance=.001)
for index,ids in enumerate(result.hole_outlines):
    points=result.mesh.nodes[ids,:2]
    lengths=np.linalg.norm(np.roll(points,-1,axis=0)-points,axis=1)
    print('hole',index,'minimum edge',float(lengths.min()))
    try:
        _outline_o_directions(points,128*np.finfo(float).eps)
        print('accepted by layer target-direction predicate')
    except Exception as error:
        print(type(error).__name__,str(error),getattr(error,'context',{}))
