"""HTTP service for game-ready 3D asset generation and processing.

Endpoints
---------
GET  /healthz          process liveness
GET  /readyz           model/checkpoint readiness
GET  /metrics          Prometheus-style counters
GET  /v1/checkpoints   resolved checkpoint inventory
POST /v1/generate      generate GLB or T3D bytes

The runtime is loaded lazily in a worker thread so a slow model load does not
block health checks and container orchestration can distinguish liveness from
readiness.
"""
from __future__ import annotations

import asyncio
import hmac
import logging
import uuid
from contextlib import asynccontextmanager
import os
import re
import time
from pathlib import Path
from typing import Literal

from fastapi import FastAPI, Header, HTTPException, Request, Response
from fastapi.responses import JSONResponse, PlainTextResponse
from pydantic import BaseModel, Field

from .runtime import T3DRuntime

LOGGER = logging.getLogger('t3d.service')
MAX_BODY_BYTES = 64 * 1024
_SAFE_NAME = re.compile(r'[^a-zA-Z0-9._-]+')


class GenerateRequest(BaseModel):
    prompt: str = Field(min_length=1, max_length=500)
    format: Literal['glb', 't3d'] = 'glb'
    steps: int = Field(default=20, ge=1, le=100)
    guidance: float = Field(default=2.0, ge=0.0, le=10.0)
    occupancy_guidance: float = Field(default=1.0, ge=0.0, le=10.0)
    seed: int = Field(default=42, ge=0, le=2**32 - 1)
    threshold: float = Field(default=0.5, gt=0.0, lt=1.0)
    decode_threshold: float | None = Field(default=None, gt=0.0, lt=1.0)
    method: Literal['euler', 'midpoint'] = 'midpoint'


class ServiceState:
    def __init__(self, runtime: T3DRuntime | None = None):
        self.runtime = runtime
        self.error: str | None = None
        self.started_at = time.time()
        self.generation_seconds_total = 0.0
        self.generation_count = 0
        self.generation_errors = 0
        self._locks: dict = {}
        self._semaphores: dict = {}

    @property
    def lock(self) -> asyncio.Lock:
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.Lock()
        if loop not in self._locks:
            self._locks[loop] = asyncio.Lock()
        return self._locks[loop]

    @property
    def semaphore(self) -> asyncio.Semaphore:
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.Semaphore(max(1, int(os.environ.get('T3D_MAX_CONCURRENCY', '1'))))
        if loop not in self._semaphores:
            self._semaphores[loop] = asyncio.Semaphore(max(1, int(os.environ.get('T3D_MAX_CONCURRENCY', '1'))))
        return self._semaphores[loop]

    async def ensure_runtime(self) -> T3DRuntime:
        if self.runtime is not None:
            return self.runtime
        if self.error is not None:
            raise RuntimeError(self.error)
        async with self.lock:
            if self.runtime is None and self.error is None:
                try:
                    self.runtime = await asyncio.to_thread(
                        T3DRuntime,
                        material=_flag('T3D_ENABLE_MATERIAL'),
                        device=os.environ.get('T3D_DEVICE') or None,
                        artifacts=os.environ.get('T3D_ARTIFACTS') or None,
                    )
                    LOGGER.info('runtime ready on %s', self.runtime.device)
                except Exception as exc:
                    self.error = f'{exc.__class__.__name__}: {exc}'
                    LOGGER.exception('runtime initialization failed')
        if self.runtime is None:
            raise RuntimeError(self.error or 'runtime unavailable')
        return self.runtime


def _flag(name: str) -> bool:
    return os.environ.get(name, '').lower() in ('1', 'true', 'yes', 'on')


def _authorized(api_key: str | None) -> bool:
    expected = os.environ.get('T3D_API_KEY')
    if not expected:
        return True
    return bool(api_key) and hmac.compare_digest(api_key, expected)


def _filename(prompt: str, suffix: str) -> str:
    stem = _SAFE_NAME.sub('-', prompt.strip().lower()).strip('-')[:48] or 'asset'
    return f'{stem}.{suffix}'


def create_app(runtime: T3DRuntime | None = None) -> FastAPI:
    state = ServiceState(runtime)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        if _flag('T3D_EAGER_LOAD'):
            try:
                await state.ensure_runtime()
            except Exception:
                pass
        yield

    app = FastAPI(title='T3D Generation API', version='1.0.0', lifespan=lifespan)

    @app.middleware('http')
    async def request_context(request: Request, call_next):
        request_id = request.headers.get('x-request-id') or uuid.uuid4().hex
        response = await _limit_body_and_call(request, call_next)
        response.headers['X-Request-ID'] = request_id
        return response

    async def _limit_body_and_call(request: Request, call_next):
        length = request.headers.get('content-length')
        if length is not None:
            try:
                if int(length) > MAX_BODY_BYTES:
                    return JSONResponse({'detail': 'request body too large'}, status_code=413)
            except ValueError:
                return JSONResponse({'detail': 'invalid content-length'}, status_code=400)
        return await call_next(request)

    @app.get('/healthz')
    async def healthz():
        return {'status': 'ok', 'uptime_seconds': round(time.time() - state.started_at, 3)}

    @app.get('/readyz')
    async def readyz():
        if state.runtime is None:
            await state.ensure_runtime()
        body = state.runtime.readiness()
        body['uptime_seconds'] = round(time.time() - state.started_at, 3)
        return body

    @app.get('/metrics', response_class=PlainTextResponse)
    async def metrics():
        lines = [
            '# HELP t3d_up Whether the runtime is loaded.',
            '# TYPE t3d_up gauge',
            f't3d_up {1 if state.runtime is not None else 0}',
            '# HELP t3d_generation_requests_total Total generation requests.',
            '# TYPE t3d_generation_requests_total counter',
            f't3d_generation_requests_total {state.generation_count}',
            '# HELP t3d_generation_errors_total Failed generation requests.',
            '# TYPE t3d_generation_errors_total counter',
            f't3d_generation_errors_total {state.generation_errors}',
            '# HELP t3d_generation_seconds_total Total generation latency.',
            '# TYPE t3d_generation_seconds_total counter',
            f't3d_generation_seconds_total {state.generation_seconds_total:.6f}',
        ]
        return '\n'.join(lines) + '\n'

    @app.get('/v1/checkpoints')
    async def checkpoints(x_api_key: str | None = Header(default=None)):
        if not _authorized(x_api_key):
            raise HTTPException(status_code=401, detail='invalid API key')
        runtime_model = await state.ensure_runtime()
        return runtime_model.checkpoints

    @app.post('/v1/generate')
    async def generate(payload: GenerateRequest, x_api_key: str | None = Header(default=None)):
        if not _authorized(x_api_key):
            raise HTTPException(status_code=401, detail='invalid API key')
        runtime = await state.ensure_runtime()
        start = time.perf_counter()
        state.generation_count += 1
        try:
            async with state.semaphore:
                data = await asyncio.to_thread(
                    runtime.generate,
                    payload.prompt,
                    format=payload.format,
                    steps=payload.steps,
                    guidance=payload.guidance,
                    occupancy_guidance=payload.occupancy_guidance,
                    seed=payload.seed,
                    threshold=payload.threshold,
                    decode_threshold=payload.decode_threshold,
                    method=payload.method,
                )
        except ValueError as exc:
            state.generation_errors += 1
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        except Exception:
            state.generation_errors += 1
            runtime.error_count += 1
            LOGGER.exception('generation failed for prompt=%r', payload.prompt)
            raise HTTPException(status_code=500, detail='generation failed') from None
        finally:
            state.generation_seconds_total += time.perf_counter() - start
        media = 'model/gltf-binary' if payload.format == 'glb' else 'application/octet-stream'
        return Response(
            content=data,
            media_type=media,
            headers={
                'Content-Disposition': f'attachment; filename="{_filename(payload.prompt, payload.format)}"',
                'X-T3D-Seed': str(payload.seed),
                'Cache-Control': 'no-store',
            },
        )

    @app.exception_handler(RuntimeError)
    async def runtime_error(request: Request, exc: RuntimeError):
        return JSONResponse({'detail': f'runtime not ready: {exc}'}, status_code=503)

    return app


app = create_app()
