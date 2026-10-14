"""API contract tests: valid requests succeed, invalid ones are rejected with 422."""
import os

import pytest

from conftest import BUNDLE

pytestmark = pytest.mark.skipif(not os.path.exists(BUNDLE), reason="deployment bundle not exported")


@pytest.fixture(scope="module")
def client():
    from fastapi.testclient import TestClient
    from app.main import app
    with TestClient(app) as c:
        yield c


@pytest.fixture
def patient():
    from app.schemas import EXAMPLE
    return dict(EXAMPLE)


def test_health(client):
    assert client.get("/health").json() == {"status": "ok", "model_loaded": True}


def test_model_info(client):
    info = client.get("/model/info").json()
    assert info["reference_patients"] > 0 and len(info["relations"]) == 3


def test_predict_valid(client, patient):
    r = client.post("/predict", json=patient)
    assert r.status_code == 200, r.text
    body = r.json()
    assert 0 <= body["readmission_risk"] <= 1 and "x-process-time-ms" in r.headers


@pytest.mark.parametrize("field,value", [
    ("age", "[70-85)"), ("time_in_hospital", 0), ("time_in_hospital", 30), ("number_inpatient", -1),
    ("diag_1", "ABC"), ("diag_1", "4280.123"), ("gender", "X"), ("discharge_disposition_id", 11),
    ("A1Cresult", ">9"),
])
def test_predict_invalid(client, patient, field, value):
    patient[field] = value
    assert client.post("/predict", json=patient).status_code == 422


def test_unknown_field_rejected(client, patient):
    patient["weight"] = 80
    assert client.post("/predict", json=patient).status_code == 422


def test_missing_required_rejected(client, patient):
    del patient["diag_1"]
    assert client.post("/predict", json=patient).status_code == 422


def test_batch(client, patient):
    r = client.post("/predict/batch", json={"patients": [patient, patient]})
    assert r.status_code == 200 and len(r.json()) == 2
    assert client.post("/predict/batch", json={"patients": [patient] * 101}).status_code == 422


def test_root_redirects_to_ui(client):
    assert client.get("/", follow_redirects=False).status_code in (302, 307)
