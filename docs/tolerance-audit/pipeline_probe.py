import copy,json
import numpy as np
from mesher.mesh25d import build_mesh
from mesher.mesh25d.geometry.patterns import extract_pattern
from mesher.mesh25d.geometry.primitives import _geometry_to_z
from mesher.mesh25d.circle_planning import _validate_circle_domain_topology,_CirclePattern,_collect_pattern_segments
from mesher.mesh25d.domain import ModelDomain,SymmetryMode,polygon_intersection_area,polygon_area_tolerance,face_intersects_domain
from mesher.mesh25d.plan3d import plan_layers
from mesher.mesh25d.region_adapter import _orthogonal_loop
from mesher.mesh3d.extrusion import Dragger
from mesher.meshcontrol.methods import MeshControl_ZsectionTop
from mesher.process_flow.exporters.cdb import _format_float

def out(name,data): print(json.dumps({'probe':name, 'result':data}, default=str))
def attempt(fn):
    try: return fn()
    except Exception as e: return f'{type(e).__name__}: {e}'
def geometry(z=0,thk=1): return {'type':'BoxGeometry','bottom_left':[0,0,z],'top_right':[1,1,z],'thk':thk}
def root(): return {'bodies':[{'geometry':geometry(),'material':'Si'}], 'vias':[], 'bumps':[], 'circuits':[], 'children':[]}
def control(z): return {'method':'Z_POINT','coordinateMode':'absolute','label':'probe','z':z}
mc={'schemaVersion':'1.0.0','unitSystem':'um','mesher':'process_flow_2_5d','globalElementSize':1,'symmetry':'full','controls':[]}

for cs in [[],[control(.5)],[control(.5),control(.5004)],[control(.3),control(.1+.2)]]:
    mesh=build_mesh({'root':root()},dict(mc,controls=cs))
    out('Z_POINT thin slab',{'cuts': [c['z'] for c in cs], 'z_levels':np.unique(mesh.nodes[:,2]).tolist(),'elements':mesh.element_count})
out('geometry rounding discontinuity',[(z,_geometry_to_z(geometry(z))) for z in [1.000499,1.000501]])
r=root(); r['bodies'][0]['geometry']['thk']=.0004
out('sub-tolerance geometry silently disappears',attempt(lambda: {'elements':build_mesh({'root':r},mc).element_count}))
for koz in [None,.05,.0001]:
    r=root()
    if koz is not None: r['vias']=[{'geometry':{'type':'CylinderGeometry','center':[.5,.5,0],'bottom_radius':.1,'thk':1},'koz':koz}]
    p=extract_pattern(r)
    out('KOZ lowers global tolerance',{'koz':koz,'coordinate':p.coordinate_tolerance,'clearance':p.clearance_tolerance})
r=root(); r['bodies'].append({'geometry':geometry(),'material':'Cu'}); r['bodies'][1]['geometry']['bottom_left'][0]=.0004
p=extract_pattern(r,tolerance=.001)
out('explicit pattern tolerance does not define all policy',{'faces':len(p.all_faces),'coordinate':p.coordinate_tolerance,'clearance':p.clearance_tolerance})
for shift in [0,.0004]:
    out('circle symmetry center',{'offset':shift,'status':attempt(lambda:_validate_circle_domain_topology([_CirclePattern(shift,0,1)],ModelDomain(SymmetryMode.RIGHT_HALF,center_x=0),band_width=.1))})
for shift in [0,.0004]:
    loop=[[0,0],[1,shift],[1,1],[0,1]]
    out('region orthogonality',{'jitter':shift,'status':attempt(lambda:_orthogonal_loop(loop,sources=['probe'])),'guides':len(_collect_pattern_segments([{'type':'POLYGON','dim':[loop]}]))})
# Entire element lies beyond right edge by more than 0.001.
d=Dragger();d.set_2D(np.array([[1.002,.1],[1.004,.1],[1.004,.9],[1.002,.9]]),np.array([[0,3,2,1]]))
out('default material tolerance accepts outside cells',{'default':d._search_faces({'type':'BOX','dim':[0,0,1,1]}).tolist(),'at_0.001':d._search_faces({'type':'BOX','dim':[0,0,1,1]},tolerance=.001).tolist()})
# Path radius is half geometric displacement, unlike box and circle branches.
d=Dragger(); d.set_2D(np.array([[.9,.1],[1.00075,.1],[1.00075,.9],[.9,.9]]),np.array([[0,3,2,1]]))
out('polygon vs box tolerance',{'box':d._search_faces({'type':'BOX','dim':[0,0,1,1]},tolerance=.001).tolist(),'polygon':d._search_faces({'type':'POLYGON','dim':[[[0,0],[1,0],[1,1],[0,1]]]},tolerance=.001).tolist()})
for end in [1.005,1.0005,1.9995]:
    c=MeshControl_ZsectionTop({'method':'Z_SECTION_TOP','coordinateMode':'absolute','startZ':0,'endZ':end,'elementSize':1})
    out('Z_SECTION threshold',{'end':end,'default':c.get_25D_z_controls(),'at_0.001':c.get_25D_z_controls(min_dis=.001)})
for shift in [0,1e7,1e8]:
    face={'type':'POLYGON','dim':[[[shift,shift],[shift+1,shift],[shift+1,shift+1],[shift,shift+1]]]}
    domain=ModelDomain(SymmetryMode.RIGHT_HALF,center_x=shift+.5)
    out('translated symmetry polygon',{'offset':shift,'area':polygon_intersection_area(face,domain),'threshold':polygon_area_tolerance(face),'retained':face_intersects_domain(face,domain)})
for x in [1e6+.004,1e9+.004,1e10+.004]:
    out('CDB float precision',{'x':x,'serialized':_format_float(x),'error':float(_format_float(x))-x})
from mesher.mesh25d.plan2d import _section_avg_lines
for shift in [0,1e7,1e8]:
    d=Dragger(); d.set_2D(np.array([[shift,shift],[shift+1,shift],[shift+1,shift+1],[shift,shift+1],[shift+2,shift],[shift+2,shift+1]]),np.array([[0,3,2,1],[1,2,5,4]]))
    out('translated extrusion areas/density',{'offset':shift,'areas':d.element_2D_volume.tolist(),'25pct_selection':d._assign_metal(np.array([0,1]),25,float(d.element_2D_volume.sum()),0).tolist()})
for thickness in [.3,.300000001]:
    r=root();r['bodies'][0]['geometry']['thk']=thickness
    mesh=build_mesh({'root':r},dict(mc,globalElementSize=.1))
    out('integer step division drift',{'thickness':thickness,'z_levels':np.unique(mesh.nodes[:,2]).tolist(),'xy_section_cuts':_section_avg_lines([0,thickness],.1,(-1,1))})
# Reference z anchor derives raw .5004 while corresponding material event rounds to .500.
r=root();r['bodies'][0]['geometry']['thk']=.5004
ref={'method':'Z_POINT','coordinateMode':'reference','label':'same upper material boundary','reference':{'kind':'root'},'z':{'anchor':'z_max','offset':0}}
from mesher.mesh25d.controls import resolve_controls
from mesher.mesh25d.geometry.assignments import extract_assignments
cc=resolve_controls(r,r,[ref],global_element_size=1)
a=extract_assignments(r)
out('reference anchor vs geometry event',{'control':cc.z,'event_z':[e['z'] for e in a]})
