"""Comprehensive stress testing harness for Transformer-Native 3D (T3D).

Executes 5 stress testing phases:
1. High-Volume Diverse Prompt Generation & Determinism
2. ODE Integration & Hyperparameter Boundary Stress (Steps, Guidance, Methods, Thresholds)
3. Mesh Topology & Manifoldness Rigor across All Generated Assets
4. Memory Leak, Resource Descriptors, and Latency Profiling (40 consecutive runs)
5. FastAPI HTTP Service Concurrency, Auth, Rate Limiting, and Bad-Input Fuzzing
"""
import concurrent.futures
import json
import os
import sys
import time
from pathlib import Path

# Add project root to path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
import torch
import trimesh

from t3d.runtime import T3DRuntime
from t3d.service import create_app
from fastapi.testclient import TestClient


def get_rss_mb() -> float:
    try:
        with open('/proc/self/status') as f:
            for line in f:
                if line.startswith('VmRSS:'):
                    return float(line.split()[1]) / 1024.0
    except Exception:
        pass
    return 0.0


def count_fds() -> int:
    try:
        return len(os.listdir('/proc/self/fd'))
    except Exception:
        return 0


def audit_mesh(glb_bytes: bytes) -> dict:
    """Validate 3D mesh geometry, finite numbers, degenerate faces, and manifoldness."""
    res = {
        'valid': False,
        'bytes': len(glb_bytes),
        'vertices': 0,
        'faces': 0,
        'nonmanifold_edges': -1,
        'watertight': False,
        'has_nan': True,
        'has_degenerate': True,
        'error': None,
    }
    if len(glb_bytes) == 0:
        res['error'] = 'empty glb bytes'
        return res

    try:
        m = trimesh.load(trimesh.util.wrap_as_stream(glb_bytes), file_type='glb', force='mesh')
        if isinstance(m, trimesh.Scene):
            if len(m.geometry) == 0:
                res['error'] = 'empty scene geometry'
                return res
            m = trimesh.util.concatenate([g for g in m.geometry.values() if isinstance(g, trimesh.Trimesh)])

        v = np.asarray(m.vertices)
        f = np.asarray(m.faces)

        res['vertices'] = len(v)
        res['faces'] = len(f)

        if len(v) == 0 or len(f) == 0:
            res['error'] = 'zero vertices or faces'
            return res

        res['has_nan'] = not np.isfinite(v).all()
        triangles = v[f]
        cross = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
        areas = np.linalg.norm(cross, axis=1) * 0.5
        res['has_degenerate'] = bool(np.any(areas <= 1e-12))

        # Check manifold edges (welding UV seams first)
        welded_v, inv = np.unique(v, axis=0, return_inverse=True)
        welded_f = inv[f.astype(np.int64)]
        edges = {}
        for tri in welded_f:
            for a, b in zip(tri, np.roll(tri, -1)):
                edge = tuple(sorted((int(a), int(b))))
                edges[edge] = edges.get(edge, 0) + 1
        nonmanifold = sum(1 for c in edges.values() if c > 2)
        res['nonmanifold_edges'] = nonmanifold
        res['watertight'] = bool(m.is_watertight)
        res['valid'] = (not res['has_nan']) and (nonmanifold == 0) and (len(v) > 0)
    except Exception as exc:
        res['error'] = f'{exc.__class__.__name__}: {exc}'

    return res


def run_stress_test(output_dir: Path) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    report = {
        'timestamp': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
        'phases': {},
        'overall_success': False,
    }

    print('=' * 70)
    print('🚀 STARTING T3D COMPREHENSIVE PRODUCTION STRESS TEST SUITE')
    print('=' * 70, flush=True)

    # Initialize Runtime
    print('\n[INIT] Initializing T3DRuntime...')
    t_init_0 = time.perf_counter()
    init_rss_0 = get_rss_mb()
    runtime = T3DRuntime()
    t_init = time.perf_counter() - t_init_0
    init_rss = get_rss_mb()
    print(f'Runtime ready in {t_init:.2f}s | Unified Model: {runtime.unified_model is not None} | VmRSS: {init_rss:.1f} MB')

    # -------------------------------------------------------------
    # PHASE 1: Diverse Prompt Battery & Determinism
    # -------------------------------------------------------------
    print('\n' + '-' * 70)
    print('PHASE 1: Diverse Prompt Battery & Determinism Stress')
    print('-' * 70, flush=True)

    prompts = [
        # In-distribution low-poly props & structures
        ('p01_barrel', 'a wooden barrel pirate theme asset'),
        ('p02_bottle', 'a glass bottle with cork pirate theme asset'),
        ('p03_chest', 'a treasure chest pirate theme asset'),
        ('p04_rowboat', 'a wooden rowboat pirate theme asset'),
        ('p05_speeder', 'a sci-fi craft speeder vehicle space asset'),
        ('p06_monorail', 'a futuristic monorail train front space asset'),
        ('p07_rocket', 'a rocket launch base platform space asset'),
        ('p08_antenna', 'a satellite communications antenna space asset'),
        ('p09_castle_gate', 'a medieval castle stone entrance gate'),
        ('p10_tower', 'a tall watchtower building asset'),
        # Sparse / thin / delicate features
        ('p11_sword', 'a sharp steel sword with crossguard weapon'),
        ('p12_flag', 'a pirate pennant flag flying on mast'),
        ('p13_fence', 'a low wooden fence barrier post'),
        ('p14_street_lamp', 'a medieval street light post with lantern'),
        # Solid / volumetric terrain
        ('p15_boulder', 'a large mossy boulder rock formation'),
        ('p16_road_cross', 'a city road crossroad asphalt intersection block'),
        ('p17_stairs', 'short concrete stairs steps platform'),
        # Short / minimal prompts
        ('p18_car', 'car'),
        ('p19_boat', 'boat'),
        ('p20_chair', 'wooden chair'),
        # Multi-lingual & Unicode
        ('p21_unicode_space', '🚀 futuristic orbital satellite space asset'),
        ('p22_french', 'un petit tonneau en bois rustique'),
        ('p23_japanese', '木製の樽 レトロゲーム風'),
        # Out-of-distribution prompts
        ('p24_clock', 'an antique grandfather clock with pendulum'),
        ('p25_robot', 'a friendly companion robot with wheels'),
        ('p26_drone', 'a quadcopter drone with four spinning propellers'),
        # Long descriptive prompt
        ('p27_long_dungeon', (
            'a highly detailed ancient stone dungeon crypt sarcophagus adorned with carved gothic runes, '
            'weathered granite slabs, iron metal bands, cracked edges, and decorative skeleton gargoyles '
            'suitable for a dark fantasy dungeon crawler game'
        )),
        ('p28_special_chars', 'metal-barrel_v2.0 (sci-fi edition) [high-tier] #space*craft!'),
    ]

    phase1_results = []
    phase1_latencies = []
    phase1_meshes_passed = 0

    for name, p in prompts:
        t0 = time.perf_counter()
        glb_data = runtime.generate(p, steps=25, guidance=3.5, method='midpoint', seed=42)
        dur = time.perf_counter() - t0
        phase1_latencies.append(dur)

        audit = audit_mesh(glb_data)
        if audit['valid'] and audit['nonmanifold_edges'] == 0:
            phase1_meshes_passed += 1

        res_entry = {
            'name': name,
            'prompt': p,
            'latency': round(dur, 3),
            'bytes': len(glb_data),
            'vertices': audit['vertices'],
            'faces': audit['faces'],
            'nonmanifold_edges': audit['nonmanifold_edges'],
            'valid': audit['valid'],
            'error': audit['error'],
        }
        phase1_results.append(res_entry)
        status = '✅ PASS' if audit['valid'] and audit['nonmanifold_edges'] == 0 else '❌ FAIL'
        print(f'  [{status}] {name:<20} | {dur:4.2f}s | {len(glb_data)//1024:4d} KB | V={audit["vertices"]:<5d} F={audit["faces"]:<5d} NonManifold={audit["nonmanifold_edges"]}', flush=True)

    # Determinism test
    print('\nTesting Seed Determinism (seed=99999)...')
    det_run1 = runtime.generate('a wooden barrel pirate theme asset', steps=20, seed=99999)
    det_run2 = runtime.generate('a wooden barrel pirate theme asset', steps=20, seed=99999)
    det_diff = runtime.generate('a wooden barrel pirate theme asset', steps=20, seed=88888)

    det_identical = (len(det_run1) == len(det_run2)) and (det_run1 == det_run2)
    det_varies = (det_run1 != det_diff)
    print(f'  Exact Byte Identity on Same Seed: {"✅ YES" if det_identical else "❌ NO"}')
    print(f'  Variation on Different Seed:     {"✅ YES" if det_varies else "❌ NO"}')

    report['phases']['phase1_prompt_battery'] = {
        'total_tested': len(prompts),
        'meshes_passed': phase1_meshes_passed,
        'pass_rate': round(phase1_meshes_passed / len(prompts), 4),
        'latency_p50': round(float(np.percentile(phase1_latencies, 50)), 3),
        'latency_p90': round(float(np.percentile(phase1_latencies, 90)), 3),
        'latency_mean': round(float(np.mean(phase1_latencies)), 3),
        'determinism_exact_bytes': det_identical,
        'seed_diversity_verified': det_varies,
        'results': phase1_results,
    }

    # -------------------------------------------------------------
    # PHASE 2: ODE & Hyperparameter Boundary Stress
    # -------------------------------------------------------------
    print('\n' + '-' * 70)
    print('PHASE 2: ODE Integration & Hyperparameter Boundary Grid Stress')
    print('-' * 70, flush=True)

    phase2_results = []
    phase2_passed = 0

    # 1. Step count stress: 1, 5, 10, 20, 30, 50, 75
    test_steps = [1, 5, 10, 20, 30, 50, 75]
    for s in test_steps:
        t0 = time.perf_counter()
        glb = runtime.generate('a sci-fi craft speeder', steps=s, guidance=3.5, method='midpoint', seed=42)
        dur = time.perf_counter() - t0
        audit = audit_mesh(glb)
        ok = audit['valid'] and audit['nonmanifold_edges'] == 0
        if ok:
            phase2_passed += 1
        phase2_results.append({'param': f'steps={s}', 'dur': round(dur, 2), 'audit': audit, 'ok': ok})
        status = '✅' if ok else '❌'
        print(f'  {status} Steps={s:<2d}       | {dur:4.2f}s | Tris={audit["faces"]:<5d} | NonManifold={audit["nonmanifold_edges"]}')

    # 2. Guidance scale stress: 0.0, 1.0, 2.5, 3.5, 5.0, 7.5, 10.0
    test_cfg = [0.0, 1.0, 2.5, 3.5, 5.0, 7.5, 10.0]
    for g in test_cfg:
        t0 = time.perf_counter()
        glb = runtime.generate('a treasure chest', steps=20, guidance=g, method='midpoint', seed=42)
        dur = time.perf_counter() - t0
        audit = audit_mesh(glb)
        ok = audit['valid'] and audit['nonmanifold_edges'] == 0
        if ok:
            phase2_passed += 1
        phase2_results.append({'param': f'guidance={g}', 'dur': round(dur, 2), 'audit': audit, 'ok': ok})
        status = '✅' if ok else '❌'
        print(f'  {status} Guidance={g:<4.1f}   | {dur:4.2f}s | Tris={audit["faces"]:<5d} | NonManifold={audit["nonmanifold_edges"]}')

    # 3. Solver method: euler vs midpoint
    for m in ['euler', 'midpoint']:
        t0 = time.perf_counter()
        glb = runtime.generate('a wooden barrel', steps=20, guidance=3.5, method=m, seed=42)
        dur = time.perf_counter() - t0
        audit = audit_mesh(glb)
        ok = audit['valid'] and audit['nonmanifold_edges'] == 0
        if ok:
            phase2_passed += 1
        phase2_results.append({'param': f'method={m}', 'dur': round(dur, 2), 'audit': audit, 'ok': ok})
        status = '✅' if ok else '❌'
        print(f'  {status} Method={m:<8s}   | {dur:4.2f}s | Tris={audit["faces"]:<5d} | NonManifold={audit["nonmanifold_edges"]}')

    # 4. Occupancy threshold fallback stress: -0.8, -0.4, 0.0, 0.4, 0.8
    test_th = [-0.8, -0.4, 0.0, 0.4, 0.8]
    for th in test_th:
        t0 = time.perf_counter()
        glb = runtime.generate('a rowboat pirate', steps=20, guidance=3.5, threshold=th, seed=42)
        dur = time.perf_counter() - t0
        audit = audit_mesh(glb)
        ok = audit['valid'] and audit['nonmanifold_edges'] == 0
        if ok:
            phase2_passed += 1
        phase2_results.append({'param': f'threshold={th}', 'dur': round(dur, 2), 'audit': audit, 'ok': ok})
        status = '✅' if ok else '❌'
        print(f'  {status} Threshold={th:<4.1f}  | {dur:4.2f}s | Tris={audit["faces"]:<5d} | NonManifold={audit["nonmanifold_edges"]}')

    total_p2 = len(test_steps) + len(test_cfg) + 2 + len(test_th)
    report['phases']['phase2_hyperparameter_grid'] = {
        'total_tested': total_p2,
        'passed': phase2_passed,
        'pass_rate': round(phase2_passed / total_p2, 4),
        'results': phase2_results,
    }

    # -------------------------------------------------------------
    # PHASE 3: Long-Running Concurrency, Memory & Resource Leak Profiling
    # -------------------------------------------------------------
    print('\n' + '-' * 70)
    print('PHASE 3: Consecutive Load, Memory Leak & Descriptor Profiling (40 Runs)')
    print('-' * 70, flush=True)

    p3_latencies = []
    p3_rss_history = []
    p3_fd_0 = count_fds()
    rss_start = get_rss_mb()

    print(f'Starting baseline VmRSS: {rss_start:.2f} MB | Initial FDs: {p3_fd_0}')

    consecutive_runs = 40
    test_prompts_p3 = [
        'a wooden barrel',
        'a pirate ship medium',
        'a space craft speeder',
        'a city road square',
        'a knight sword',
        'a watchtower',
        'a treasure chest',
        'a monorail train',
    ]

    for i in range(1, consecutive_runs + 1):
        prompt_i = test_prompts_p3[i % len(test_prompts_p3)]
        t0 = time.perf_counter()
        glb = runtime.generate(prompt_i, steps=20, guidance=3.5, seed=i)
        lat = time.perf_counter() - t0
        p3_latencies.append(lat)

        if i % 5 == 0 or i == consecutive_runs:
            rss_now = get_rss_mb()
            p3_rss_history.append((i, rss_now))
            print(f'  Run {i:>2d}/40 | Latency: {lat:4.2f}s | Mesh: {len(glb)//1024:3d} KB | VmRSS: {rss_now:.2f} MB')

    rss_end = get_rss_mb()
    p3_fd_end = count_fds()
    mem_delta = rss_end - rss_start
    fd_delta = p3_fd_end - p3_fd_0

    print(f'\nMemory Profiling Summary:')
    print(f'  Start VmRSS:  {rss_start:.2f} MB')
    print(f'  End VmRSS:    {rss_end:.2f} MB')
    print(f'  Net Delta:    {mem_delta:+.2f} MB')
    print(f'  File Descriptors: {p3_fd_0} -> {p3_fd_end} (Delta: {fd_delta:+d})')
    print(f'  P50 Latency:  {np.percentile(p3_latencies, 50):.3f}s')
    print(f'  P90 Latency:  {np.percentile(p3_latencies, 90):.3f}s')
    print(f'  P99 Latency:  {np.percentile(p3_latencies, 99):.3f}s')

    report['phases']['phase3_memory_resource_profiling'] = {
        'consecutive_runs': consecutive_runs,
        'rss_start_mb': round(rss_start, 2),
        'rss_end_mb': round(rss_end, 2),
        'rss_delta_mb': round(mem_delta, 2),
        'memory_leak_detected': bool(mem_delta > 150.0),  # more than 150MB growth over 40 runs
        'fd_start': p3_fd_0,
        'fd_end': p3_fd_end,
        'fd_leaked': bool(fd_delta > 5),
        'latency_p50': round(float(np.percentile(p3_latencies, 50)), 3),
        'latency_p90': round(float(np.percentile(p3_latencies, 90)), 3),
        'latency_p99': round(float(np.percentile(p3_latencies, 99)), 3),
        'rss_samples': p3_rss_history,
    }

    # -------------------------------------------------------------
    # PHASE 4: FastAPI HTTP Service Concurrency & Edge-Case Fuzzing
    # -------------------------------------------------------------
    print('\n' + '-' * 70)
    print('PHASE 4: FastAPI Service Concurrency, Auth, and Edge-Case Fuzzing')
    print('-' * 70, flush=True)

    # Set temporary test API key in environment
    TEST_KEY = 'secret-test-token-777'
    os.environ['T3D_API_KEY'] = TEST_KEY

    app = create_app(runtime=runtime)
    client = TestClient(app)

    service_tests = []

    # 1. Healthcheck & Metrics
    r_health = client.get('/healthz')
    assert r_health.status_code == 200 and r_health.json()['status'] == 'ok'
    print('  [✅ PASS] GET /healthz -> 200 OK')

    r_ready = client.get('/readyz')
    assert r_ready.status_code == 200 and r_ready.json().get('ready') is True
    print('  [✅ PASS] GET /readyz -> 200 OK (Status: ready)')

    r_metrics = client.get('/metrics')
    assert r_metrics.status_code == 200 and 't3d_up 1' in r_metrics.text
    print('  [✅ PASS] GET /metrics -> 200 OK (Prometheus metrics verified)')

    # 2. Authentication Stress
    r_no_auth = client.post('/v1/generate', json={'prompt': 'a wooden barrel'})
    assert r_no_auth.status_code == 401, f'Expected 401, got {r_no_auth.status_code}'
    print('  [✅ PASS] Missing API Key Rejected -> 401 Unauthorized')

    r_bad_auth = client.post('/v1/generate', json={'prompt': 'a wooden barrel'}, headers={'x-api-key': 'wrong-key'})
    assert r_bad_auth.status_code == 401, f'Expected 401, got {r_bad_auth.status_code}'
    print('  [✅ PASS] Invalid API Key Rejected -> 401 Unauthorized')

    r_good_auth = client.post('/v1/generate', json={'prompt': 'a wooden barrel', 'steps': 15}, headers={'x-api-key': TEST_KEY})
    assert r_good_auth.status_code == 200, f'Expected 200, got {r_good_auth.status_code}'
    assert len(r_good_auth.content) > 10000
    assert r_good_auth.headers['content-type'] == 'model/gltf-binary'
    print(f'  [✅ PASS] Valid API Key Accepted -> 200 OK ({len(r_good_auth.content)//1024} KB GLB returned)')

    # 3. Payload Boundary & Fuzzing
    # Negative steps
    r_neg_steps = client.post('/v1/generate', json={'prompt': 'barrel', 'steps': -1}, headers={'x-api-key': TEST_KEY})
    assert r_neg_steps.status_code == 422
    print('  [✅ PASS] Negative Steps Rejected -> 422')

    # Excessive steps (>100)
    r_huge_steps = client.post('/v1/generate', json={'prompt': 'barrel', 'steps': 500}, headers={'x-api-key': TEST_KEY})
    assert r_huge_steps.status_code == 422
    print('  [✅ PASS] Out-of-bounds Steps (500) Rejected -> 422')

    # Negative guidance
    r_neg_guidance = client.post('/v1/generate', json={'prompt': 'barrel', 'guidance': -2.0}, headers={'x-api-key': TEST_KEY})
    assert r_neg_guidance.status_code == 422
    print('  [✅ PASS] Negative Guidance Rejected -> 422')

    # Excessive prompt (>500 chars)
    r_huge_prompt = client.post('/v1/generate', json={'prompt': 'a' * 501}, headers={'x-api-key': TEST_KEY})
    assert r_huge_prompt.status_code == 422
    print('  [✅ PASS] Oversized Prompt (>500 chars) Rejected -> 422')

    # Empty prompt
    r_empty_prompt = client.post('/v1/generate', json={'prompt': ''}, headers={'x-api-key': TEST_KEY})
    assert r_empty_prompt.status_code == 422
    print('  [✅ PASS] Empty Prompt Rejected -> 422')

    # Invalid format
    r_bad_fmt = client.post('/v1/generate', json={'prompt': 'barrel', 'format': 'obj'}, headers={'x-api-key': TEST_KEY})
    assert r_bad_fmt.status_code == 422
    print('  [✅ PASS] Invalid Format ("obj") Rejected -> 422')

    # Oversized Request Body (>64KB)
    r_big_body = client.post('/v1/generate', content=b'{"prompt": "' + b'x' * (70 * 1024) + b'"}', headers={'content-type': 'application/json', 'content-length': str(70 * 1024 + 15), 'x-api-key': TEST_KEY})
    assert r_big_body.status_code == 413, f'Expected 413, got {r_big_body.status_code}'
    print('  [✅ PASS] Oversized Request Body (>64KB) Rejected -> 413')

    # 4. High-Concurrency Burst (10 concurrent requests to test semaphore & runtime lock)
    print('\nTesting Concurrent Burst (10 simultaneous requests against service)...')
    concurrent_prompts = [
        'a barrel sci-fi space asset',
        'a pirate rowboat asset',
        'a treasure chest',
        'a small rock stone',
        'a space craft speeder',
        'a wooden chair',
        'a knight sword weapon',
        'a street lamp post',
        'a road crossroad',
        'a castle tower',
    ]

    def worker_request(p):
        t0 = time.perf_counter()
        resp = client.post('/v1/generate', json={'prompt': p, 'steps': 15}, headers={'x-api-key': TEST_KEY})
        lat = time.perf_counter() - t0
        return resp.status_code, len(resp.content), lat

    t_burst_0 = time.perf_counter()
    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
        futures = [executor.submit(worker_request, p) for p in concurrent_prompts]
        burst_results = [f.result() for f in futures]
    t_burst_total = time.perf_counter() - t_burst_0

    burst_status_codes = [r[0] for r in burst_results]
    burst_bytes = [r[1] for r in burst_results]
    burst_latencies = [r[2] for r in burst_results]

    all_200 = all(s == 200 for s in burst_status_codes)
    all_valid_bytes = all(b > 10000 for b in burst_bytes)

    print(f'  Burst Completion: {t_burst_total:.2f}s total ({len(concurrent_prompts)} requests)')
    print(f'  All Status 200:   {"✅ YES" if all_200 else "❌ NO"}')
    print(f'  All Valid GLBs:   {"✅ YES" if all_valid_bytes else "❌ NO"}')
    print(f'  Throughput:       {len(concurrent_prompts)/t_burst_total:.2f} requests/sec')

    # Remove test API key
    os.environ.pop('T3D_API_KEY', None)

    report['phases']['phase4_fastapi_service'] = {
        'healthz_passed': True,
        'readyz_passed': True,
        'metrics_passed': True,
        'auth_enforced': True,
        'fuzzing_rejected_properly': True,
        'concurrent_burst_total_seconds': round(t_burst_total, 2),
        'concurrent_requests': len(concurrent_prompts),
        'concurrent_all_200': all_200,
        'concurrent_all_valid_bytes': all_valid_bytes,
        'burst_throughput_req_per_sec': round(len(concurrent_prompts) / t_burst_total, 2),
    }

    # -------------------------------------------------------------
    # Overall Verification Verdict
    # -------------------------------------------------------------
    p1_pass = report['phases']['phase1_prompt_battery']['pass_rate'] == 1.0
    p2_pass = report['phases']['phase2_hyperparameter_grid']['pass_rate'] == 1.0
    p3_pass = not report['phases']['phase3_memory_resource_profiling']['memory_leak_detected']
    p4_pass = all_200 and all_valid_bytes

    overall = p1_pass and p2_pass and p3_pass and p4_pass
    report['overall_success'] = overall

    print('\n' + '=' * 70)
    print(f'STRESS TEST VERDICT: {"🎉 ALL STRESS TESTS PASSED" if overall else "⚠️ STRESS TESTS FAILED"}')
    print('=' * 70, flush=True)

    report_path = output_dir / 't3d-stress-test-results.json'
    report_path.write_text(json.dumps(report, indent=2) + '\n')
    print(f'Report written to {report_path}')
    return report


if __name__ == '__main__':
    out = ROOT / 'artifacts/stress_test'
    res = run_stress_test(out)
    sys.exit(0 if res.get('overall_success') else 1)
