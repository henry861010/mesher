import json, numpy as np
from mesher.mesh2d.model import Mesh2D
from mesher.mesh2d.adapter.layer import plan_quad_sections, generate_layer_adaptive_mesh
from mesher.mesh2d.adapter.layer._planning.source import _prepare
from mesher.mesh2d.generators.rectilinear import _sort, generate_rectilinear_mesh
from mesher.mesh2d.generators.holed_rectilinear import generate_rectilinear_mesh_optimal

def outcome(name, fn):
    try:
        result=fn()
        print(name, 'OK', result)
    except Exception as e:
        print(name, type(e).__name__, str(e), getattr(e,'code',None), getattr(e,'context',None))

def mesh(L=1,d=0):
    return Mesh2D(np.array([[0,0,0],[L,0,0],[L,L+d,0],[0,L,0]],float), np.array([[0,1,2,3]]))

def prep(L,d):
    p=_prepare(mesh(L,d), [0,3,2,1], L, (), 1, .001)
    return {'length':L,'deviation':d,'normalized_tol':p['tol'],'physical_tol':p['tol']*p['scale']}
for L,d in [(1,0),(.01,.0005),(1000,.5)]: outcome('prepare',lambda L=L,d=d:prep(L,d))
for d in [0,1e-10,.0005]:
    outcome('layer_pattern_tilt '+str(d),lambda d=d:len(plan_quad_sections(mesh(),[0,3,2,1],1,pattern_lines=[[[-2,-1],[2,-1+d]]],repair_slots=False).sections))
outer=np.array([[0,0],[2,0],[2,2],[0,2]],float)
for d in [0,1e-10,.0005]:
    o=outer.copy();o[2,1]+=d
    outcome('holed_outline_tilt '+str(d),lambda o=o:generate_rectilinear_mesh_optimal(1,o,coordinate_tolerance=.001).mesh.element_count)
for d in [0,1e-10,.0005]:
    outcome('holed_pattern_tilt '+str(d),lambda d=d:generate_rectilinear_mesh_optimal(1,outer,pattern_lines=[[[0,1],[2,1+d]]],coordinate_tolerance=.001).mesh.element_count)
print('sort_chain',_sort([0,.00075,.0015,1], .001))
for vals in [[0,.00075,.0015],[0,.00075,.0015,1]]:
    outcome('rect_chain '+str(vals),lambda vals=vals:generate_rectilinear_mesh(1,vals,[0,1],coordinate_tolerance=.001).nodes[:,0].tolist())
for d in [0,.0005,1e-10]:
    def close_patterns(d=d):
        r=generate_rectilinear_mesh_optimal(1,outer,pattern_lines=[[[0,1],[2,1]],[[0,1+d],[2,1+d]]],coordinate_tolerance=.001)
        edges=r.mesh.nodes[r.mesh.elements]; length=np.linalg.norm(np.roll(edges,-1,axis=1)-edges,axis=2)
        return {'elements':r.mesh.element_count,'min_edge':float(length.min())}
    outcome('holed_close_patterns '+str(d),close_patterns)
for end in [.3,.1+.2]:
    outcome('rect_float_count '+repr(end),lambda end=end: generate_rectilinear_mesh(.1,[0,end],[0,.1]).element_count)
for x in [0,1e12]:
    def numeric_densify(x=x):
        r=generate_rectilinear_mesh(1e-5,[x,x+.002],[0,.002],coordinate_tolerance=.001)
        points=r.nodes[r.elements]
        edges=np.linalg.norm(np.roll(points,-1,axis=1)-points,axis=2)
        return {'nodes':r.node_count,'unique_coordinates':len(np.unique(r.nodes,axis=0)),'min_edge':float(edges.min()),'zero_edges':int(np.count_nonzero(edges==0))}
    outcome('rect_numeric_densify '+str(x),numeric_densify)
from mesher.mesh2d.editor import EditorMesh2D
for x in [0,1,10]:
    def dense_probe(x=x):
        r=generate_rectilinear_mesh(1,np.array([0,.3,.6,.9])+x,[0,1])
        outline=EditorMesh2D.outlines(r)[0]
        p=plan_quad_sections(r,outline,1,repair_slots=False,small_element_size=.3)
        return {'lengths':np.diff(np.unique(r.nodes[:,0])).tolist(),'dense_sections':len(p.diagnostics['dense_sections']),'dense_seams':len(p.diagnostics['dense_seams']),'mappings':p.diagnostics['dense_mappings']}
    outcome('dense_translate '+str(x),dense_probe)
