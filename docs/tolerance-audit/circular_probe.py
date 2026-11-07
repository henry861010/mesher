import numpy as np
from mesher import Mesh2D
from mesher.mesh2d.circular.extend.mesher import extend_circular_mesh
from mesher.mesh2d.circular.imprint_v2.mesher import imprint_circle
from mesher.mesh2d.circular.imprint_v2.utils import _to_circle
from mesher.mesh2d.circular.utils.topology import _search_circle
from mesher.mesh2d.quality import MeshQualityChecker, GeometryTolerance

def fan(r):
 a=np.arange(8)*np.pi/4; xy=np.vstack(([0,0],r*np.column_stack((np.cos(a),np.sin(a)))))
 ids=np.arange(1,9); e=np.array([[0,i,j,j] for i,j in zip(ids,np.roll(ids,-1))])
 return Mesh2D(xy,e)
for r in (1.0,0.9995,1.0005):
 m=fan(r)
 try:
  extend_circular_mesh(m,element_size=.1,center_x=0,center_y=0,inner_radius=1,outer_radius=1.2)
  print('extend radius',r,'OK',len(m.elements))
 except Exception as e: print('extend radius',r,type(e).__name__,str(e))
m=Mesh2D(np.array([[.5,.8655],[0,0],[1,0]]),np.array([[0,1,2,2]]))
try:
 imprint_circle(m,center=(0,0),radius=1,projection_tolerance=.001,guide_tolerance=.001,merge_tolerance=.001,minimum_area=1e-6,guide_segments=[[[.5,-2],[.5000001,2]]])
 print('guide nearly vertical OK')
except Exception as e: print('guide nearly vertical',type(e).__name__,str(e))
for guides in ([[[.5,-2],[.5,2]]],[[[.5,-2],[.5,2]],[[.5001,-2],[.5001,2]]]):
 m=Mesh2D(np.array([[.50005,.8655],[0,0],[1,0]]),np.array([[0,1,2,2]]))
 _to_circle(m,(0,0),1,.001,.001,guides)
 print('guide count',len(guides),'projected node',m.nodes[0].tolist(),'radius',np.hypot(*m.nodes[0,:2]))
for tol in (None,GeometryTolerance(absolute=.001,relative=0)):
 m=Mesh2D(np.array([[0,0],[1,0],[1,.0005],[0,.0005]]),np.array([[0,1,2,3]]))
 q=MeshQualityChecker(m,tolerance=tol).calculate_jacobian()
 print('quality thin quad',tol,'reason_flags',list(q.reason_flags),'jacobian',list(q.values))
 m=Mesh2D(np.array([[0,0],[1000,0],[500,.0005]]),np.array([[0,1,2,2]]))
 q=MeshQualityChecker(m,tolerance=tol).calculate_jacobian()
 print('quality needle triangle',tol,'reason_flags',list(q.reason_flags),'jacobian',list(q.values))
m=Mesh2D(np.array([[1.005,0],[1.006,0],[1.005,.001]]),np.array([[0,1,2,2]]))
print('search circle default',_search_circle(m,0,0,1,type='ALL').tolist(),'explicit .001',_search_circle(m,0,0,1,type='ALL',tolerance=.001).tolist())
for h in (0.0,1e-14,1e-12,1e-9,0.0005):
 m=Mesh2D(np.array([[0,0],[1,0],[2,h],[0,1]]),np.array([[0,1,2,3]]))
 try:
  imprint_circle(m,center=(0,0),radius=.5,projection_tolerance=.001,guide_tolerance=.001,merge_tolerance=.001,minimum_area=1e-6)
  print('near-flat quad height',h,'OK',len(m.elements))
 except Exception as e: print('near-flat quad height',h,type(e).__name__,str(e))
m=Mesh2D(np.array([[0,0],[1,0],[1,1],[0,1]]),np.array([[0,1,2,3]]))
imprint_circle(m,center=(0,0),radius=.01,projection_tolerance=.001,guide_tolerance=.001,merge_tolerance=.001,minimum_area=.001)
xy=m.nodes[m.elements,:2]; q=np.roll(xy,-1,axis=1); areas=.5*np.sum(xy[:,:,0]*q[:,:,1]-xy[:,:,1]*q[:,:,0],axis=1)
print('minimum_area .001 output areas',areas.tolist())
