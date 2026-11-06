"""Kaggle CUDA extraction measurements; analytic sheets, not asset QA."""
import argparse
import gc
import json
from pathlib import Path
import statistics
import time
import torch
from t3d.extraction_torch import voxels_to_mesh


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source-sha256', required=True)
    parser.add_argument('--output', default='/kaggle/working/extraction-benchmark.json')
    parser.add_argument('--resolutions', type=int, nargs='+', default=[512,1024,1536])
    parser.add_argument('--repeats', type=int, default=3)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError('This benchmark requires CUDA')
    if args.repeats < 1 or any(r < 2 for r in args.resolutions):
        raise ValueError('positive repeats and resolutions >=2 required')
    report = dict(source_sha256=args.source_sha256, torch=torch.__version__,
                  cuda=torch.version.cuda, device=torch.cuda.get_device_name(0),
                  fixture='single axis-aligned sheet; surface support scales as resolution squared',
                  includes='validation, key sort, neighbor queries, vertex placement and triangulation',
                  excludes='input transfers, VAE decode, orientation, cleanup, UVs and GLB export',
                  results=[])
    destination = Path(args.output)
    with torch.inference_mode():
        for resolution in args.resolutions:
            xy = torch.cartesian_prod(torch.arange(resolution, device='cuda'),
                                      torch.arange(resolution, device='cuda'))
            coords = torch.cat([xy, torch.full((len(xy),1),resolution//2,device='cuda')],-1)
            del xy
            dual = torch.full(coords.shape,.5,device='cuda')
            flags = torch.zeros(coords.shape,device='cuda',dtype=torch.bool)
            flags[:,2] = True
            for weighted in (True, False):
                weights = torch.ones((len(coords),1),device='cuda') if weighted else None
                call = lambda: voxels_to_mesh(coords,dual,flags,(resolution,)*3,
                                               [[-.5]*3,[.5]*3],weights)
                v,f = call() # warm kernels and allocator before recording
                torch.cuda.synchronize()
                expected_faces=2*(resolution-1)**2
                assert len(f)==expected_faces and len(v)==resolution**2
                assert int(f.min())>=0 and int(f.max())<len(v)
                area = torch.linalg.vector_norm(torch.linalg.cross(
                    v[f[:,1]]-v[f[:,0]],v[f[:,2]]-v[f[:,0]],dim=-1),dim=-1).sum()/2
                assert abs(area.item()-((resolution-1)/resolution)**2)<1e-5
                del v,f,area
                gc.collect()
                torch.cuda.synchronize()
                baseline = torch.cuda.memory_allocated()
                torch.cuda.reset_peak_memory_stats()
                wall_times=[]
                gpu_times=[]
                for _ in range(args.repeats):
                    start,end=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
                    torch.cuda.synchronize()
                    wall=time.perf_counter()
                    start.record()
                    v,f=call()
                    end.record()
                    torch.cuda.synchronize()
                    wall_times.append((time.perf_counter()-wall)*1000)
                    gpu_times.append(start.elapsed_time(end))
                    del v,f
                row=dict(resolution=resolution,active_voxels=len(coords),faces=expected_faces,
                         split='weights' if weighted else 'minimum_angle',wall_ms=wall_times,
                         gpu_event_ms=gpu_times,median_wall_ms=statistics.median(wall_times),
                         baseline_allocated_bytes=baseline,
                         peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                         peak_increment_bytes=torch.cuda.max_memory_allocated()-baseline,
                         count_bounds_area_checks_passed=True)
                report['results'].append(row)
                destination.write_text(json.dumps(report,indent=2)+'\n')
                print(json.dumps(row),flush=True)
            del coords,dual,flags,weights
            gc.collect()
            torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
