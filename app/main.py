"""HealthGraph production API (FastAPI) with the Gradio UI mounted at /ui.

    uvicorn app.main:app --host 0.0.0.0 --port 7860
"""
import logging
import os
import time
import uuid
from contextlib import asynccontextmanager

import gradio as gr
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import RedirectResponse

from healthgraph.inference import HealthGraphPredictor

from .schemas import BatchRequest, Patient, Prediction
from .ui import build_ui

BUNDLE = os.environ.get("HEALTHGRAPH_BUNDLE", os.path.join("artifacts", "healthgat_uci.pt"))
log = logging.getLogger("healthgraph.api")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
state = {}


def get_predictor() -> HealthGraphPredictor:
    if "predictor" not in state:
        raise HTTPException(503, "model not loaded")
    return state["predictor"]


@asynccontextmanager
async def lifespan(app):
    t0 = time.perf_counter()
    state["predictor"] = HealthGraphPredictor(BUNDLE)
    log.info("loaded %s in %.1fs", BUNDLE, time.perf_counter() - t0)
    yield
    state.clear()


app = FastAPI(
    title="HealthGraph API",
    version="1.0.0",
    description="30-day readmission risk for diabetic inpatients with a relation-aware Graph Attention "
                "Network. Each request is attached inductively to a reference patient graph. "
                "Research prototype: not for clinical use.",
    lifespan=lifespan,
)


@app.middleware("http")
async def timing(request: Request, call_next):
    rid = request.headers.get("x-request-id", uuid.uuid4().hex[:12])
    t0 = time.perf_counter()
    response = await call_next(request)
    ms = 1000 * (time.perf_counter() - t0)
    response.headers["x-request-id"] = rid
    response.headers["x-process-time-ms"] = f"{ms:.1f}"
    if request.url.path.startswith(("/predict", "/model", "/health")):
        log.info("%s %s %s %d %.1fms", rid, request.method, request.url.path, response.status_code, ms)
    return response


@app.get("/", include_in_schema=False)
def root():
    return RedirectResponse("/ui")


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": "predictor" in state}


@app.get("/model/info")
def model_info():
    return get_predictor().info()


@app.post("/predict", response_model=Prediction)
def predict(patient: Patient, explain: bool = True):
    """Score one encounter. Invalid or out-of-range fields are rejected with HTTP 422."""
    return get_predictor().predict(patient.model_dump(), explain=explain)


@app.post("/predict/batch", response_model=list[Prediction])
def predict_batch(req: BatchRequest):
    p = get_predictor()
    return [p.predict(x.model_dump(), explain=req.explain) for x in req.patients]


app = gr.mount_gradio_app(app, build_ui(lambda: state["predictor"]), path="/ui")
