"""Diagnostic observer; expected errors are printed, not a passing regression test."""
import argparse, json, tempfile, time, traceback
from pathlib import Path
from unittest.mock import patch
import numpy as np
from mesher.mesh25d.controls import resolve_controls
from mesher.mesh25d.domain import normalize_symmetry
from mesher.mesh25d.geometry.context import prepare_geometry
from mesher.mesh25d.plan2d import build_2d_mesh, plan_2d
from mesher.mesh2d.adapter.layer._planning import front
from mesher.mesh2d.adapter import _full

parser = argparse.ArgumentParser(description="Observe case1's 2D failure without changing input/source files.")
parser.add_argument("--case-dir", type=Path, default=Path(__file__).resolve().parent)
parser.add_argument("--output-dir", type=Path)
options = parser.parse_args()
source = options.case_dir
output = options.output_dir or Path(tempfile.mkdtemp(prefix="mesher-tolerance-case1-"))
output.mkdir(parents=True, exist_ok=True)
print(json.dumps({"event": "diagnostic_output", "path": str(output)}), flush=True)
structure=json.loads((source/'model.geometry.json').read_text())
settings=json.loads((source/'model.mesh-control.json').read_text())

def emit(event,**data): print(json.dumps({'event':event,**data}),flush=True)
start=time.perf_counter()
geometry=prepare_geometry(structure['root'],normalize_symmetry(settings['symmetry']))
controls=resolve_controls(structure['root'],geometry.container,settings['controls'],global_element_size=settings['globalElementSize'])
plan=plan_2d(geometry.pattern,controls,element_size=settings['globalElementSize'],domain=geometry.domain,region_meshing=settings['regionMeshing'])
emit('plan',tol=geometry.pattern.coordinate_tolerance,regions=[dict(source=r.source,bounds=r.geometry.bounds,seed=r.seed_size,targets=r.transition_sizes) for r in plan.adaptive.regions])
from mesher.mesh2d.adapter.layer import _mesher
original_prepare=_mesher._prepare
active_mesh=None
def prepare(mesh,*args,**kwargs):
    global active_mesh
    result=original_prepare(mesh,*args,**kwargs)
    active_mesh=mesh
    return result
original=front._normalize_outline_o
original_grid=_full.generate_rectilinear_mesh_optimal
count=0

def grid(*args,**kwargs):
    global count
    emit('grid_start',call=count,tol=kwargs.get('coordinate_tolerance'))
    np.savez_compressed(output / ('grid_input_'+str(count)+'.npz'),outer=np.asarray(args[1]),patterns=np.asarray(kwargs.get('pattern_lines',())),**{'hole_'+str(i):h for i,h in enumerate(kwargs.get('holes',()))})
    result=original_grid(*args,**kwargs)
    np.savez_compressed(output / ('grid_'+str(count)+'.npz'),nodes=result.mesh.nodes,elements=result.mesh.elements,outer=result.outer_outline,**{'hole_'+str(i):h for i,h in enumerate(result.hole_outlines)})
    emit('grid_done',call=count,nodes=result.mesh.node_count,elements=result.mesh.element_count)
    count+=1
    return result

def normalize(outline_o,data):
    try:return original(outline_o,data)
    except Exception as e:
        ids=np.asarray(outline_o)
        points=data['xy'][ids]
        vectors=np.roll(points,-1,axis=0)-points
        axis=np.argmax(abs(vectors),axis=1); rows=np.arange(len(points))
        bad=(abs(vectors[rows,axis])<=data['tol'])|(abs(vectors[rows,1-axis])>data['tol'])
        emit('failing_target',error=str(e),context=getattr(e,'context',{}),tol=float(data['tol']),scale=float(data['scale']),physical_tol=float(data['tol']*data['scale']),origin=data['origin'].tolist(),bad=[dict(edge=int(i),ids=[int(ids[i]),int(ids[(i+1)%len(ids)])],first=points[i].tolist(),second=points[(i+1)%len(ids)].tolist(),world_first=active_mesh.nodes[ids[i]].tolist(),world_second=active_mesh.nodes[ids[(i+1)%len(ids)]].tolist(),delta=vectors[i].tolist(),world_delta=(active_mesh.nodes[ids[(i+1)%len(ids)]]-active_mesh.nodes[ids[i]]).tolist(),length=float(np.linalg.norm(active_mesh.nodes[ids[(i+1)%len(ids)]]-active_mesh.nodes[ids[i]])),normalized_delta_world_estimate=(vectors[i]*data['scale']).tolist()) for i in np.flatnonzero(bad)])
        np.savez_compressed(output / 'failing_target.npz',nodes=active_mesh.nodes,elements=active_mesh.elements,ids=ids,points=points,tol=data['tol'],origin=data['origin'],scale=data['scale'])
        raise
try:
    with patch.object(front,'_normalize_outline_o',normalize), patch.object(_full,'generate_rectilinear_mesh_optimal',grid):
        # _mesher imported this function directly; patch its binding as well.
        from mesher.mesh2d.adapter.layer import _mesher
        with patch.object(_mesher,'_normalize_outline_o',normalize), patch.object(_mesher,'_prepare',prepare):
            mesh=build_2d_mesh(plan)
    emit('success',nodes=mesh.node_count,elements=mesh.element_count,seconds=time.perf_counter()-start)
except Exception as e:
    emit('failure',error=str(e),context=getattr(e,'context',{}),seconds=time.perf_counter()-start)
    traceback.print_exc()
