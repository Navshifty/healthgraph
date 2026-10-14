---
title: HealthGraph Readmission Risk
emoji: 🩺
colorFrom: blue
colorTo: red
sdk: docker
app_port: 7860
pinned: false
license: mit
short_description: 30-day readmission risk with Graph Attention Networks
---

# HealthGraph: patient risk prediction with Graph Attention Networks

Relation-aware GATv2 predicting 30-day readmission for diabetic inpatients
(UCI Diabetes 130-US Hospitals). Each new patient is attached to a reference patient graph
(clinical similarity, shared ICD-9 diagnoses, shared admitting specialty) and scored on its
L-hop computational subgraph (L = number of GAT layers).

* Web UI: `/ui` · REST API: `POST /predict`, `POST /predict/batch`, `GET /model/info` · OpenAPI docs: `/docs`
* Research prototype on de-identified 1999–2008 data. **Not for clinical use.**
