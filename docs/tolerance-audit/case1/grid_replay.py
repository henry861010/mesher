"""Observe which grid cuts introduce near-duplicate boundary anchors."""
import json
from pathlib import Path
import numpy as np
from unittest.mock import patch
from mesher.mesh2d.generators import holed_rectilinear as hr
z=np.load(Path(__file__).resolve().parent / 'grid_input.npz')
original=hr._Builder.add
reports=[]
def add(self,axis,coordinate,interval,kind):
 cut=hr._Cut(axis,float(coordinate),interval,kind)
 if cut.key not in self.cuts:
  for owner,point in zip((interval.start_owner,interval.end_owner),cut.points):
   for other in self.boundary_points[owner]:
    distance=float(np.linalg.norm(np.asarray(point)-other))
    if 0<distance<.001:
     reports.append(dict(owner=owner,hole_index=owner-1,point=point,prior=other,distance=distance,kind=kind,axis=axis,coordinate=float(coordinate),interval=[interval.start,interval.end]))
 return original(self,axis,coordinate,interval,kind)
with patch.object(hr._Builder,'add',add):
 result=hr.generate_rectilinear_mesh_optimal(1000,z['outer'],holes=[z['hole_'+str(i)] for i in range(8)],pattern_lines=z['patterns'],coordinate_tolerance=.001)
print(json.dumps({'strategy':result.diagnostics.get('strategy'),'near_anchor_additions':reports}))
