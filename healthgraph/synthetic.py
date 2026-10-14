"""Synthetic EHR generator with a known causal structure.

Each patient belongs to a latent clinical phenotype and is treated by a provider
with a latent quality level. Both drive the outcome, but neither is observed
directly: they only leave noisy traces in labs, vitals, diagnosis codes and care
processes. A single patient's record is therefore an unreliable estimate of
them, while patients who share codes or a provider pool their evidence. That is
exactly the situation in which message passing over a patient graph should beat
a per-patient tabular model, and the ground-truth risk tells us how much
headroom is left.
"""
import numpy as np
import pandas as pd

from .data import PatientData, standardize

CODES = [
    "I10 Hypertension", "E11 Type 2 diabetes", "E78 Hyperlipidemia", "E66 Obesity",
    "N18 Chronic kidney disease", "D63 Anemia in CKD", "E87 Electrolyte disorder",
    "I50 Heart failure", "I48 Atrial fibrillation", "I25 Coronary artery disease",
    "J44 COPD", "J96 Respiratory failure", "J18 Pneumonia", "F17 Nicotine dependence",
    "C34 Lung cancer", "C50 Breast cancer", "C18 Colon cancer", "D70 Neutropenia",
    "R64 Cachexia", "F03 Dementia", "W19 Fall", "M81 Osteoporosis", "R26 Gait abnormality",
    "F32 Depression", "F10 Alcohol use disorder", "K70 Alcoholic liver disease",
    "Z59 Housing instability", "Z91.1 Medication nonadherence", "J45 Asthma", "K21 GERD",
]
_C = {c.split()[0]: i for i, c in enumerate(CODES)}

# name, prevalence, baseline log-odds, feature shifts, signature codes
PHENOTYPES = [
    ("low-acuity", 0.22, -1.2, {"age": -14, "prior_adm": -0.4}, ["J45", "K21"]),
    ("cardiometabolic", 0.20, -0.3, {"bmi": 6, "hba1c": 1.6, "sbp": 14}, ["I10", "E11", "E78", "E66"]),
    ("renal", 0.10, 0.5, {"creat": 1.3, "hgb": -1.6, "na": -2}, ["N18", "D63", "E87", "I10"]),
    ("heart-failure", 0.10, 0.8, {"los": 2, "prior_adm": 1.0, "alb": -0.3, "hr": 8, "na": -3}, ["I50", "I48", "I25"]),
    ("pulmonary", 0.10, 0.4, {"age": 4, "ed": 0.8, "wbc": 2}, ["J44", "J96", "J18", "F17"]),
    ("oncology", 0.08, 0.9, {"hgb": -2, "alb": -0.7, "wbc": -1.5}, ["C34", "C50", "C18", "D70", "R64"]),
    ("frail-elderly", 0.10, 0.7, {"age": 16, "bmi": -3, "alb": -0.4, "alone": 0.3}, ["F03", "W19", "M81", "R26"]),
    ("psychosocial", 0.10, 0.6, {"age": -8, "ed": 1.5, "medicaid": 0.3}, ["F32", "F10", "F17", "K70", "Z59", "Z91.1"]),
]


def _sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def generate(n_patients=6000, n_providers=120, prevalence=0.15, seed=42) -> PatientData:
    rng = np.random.default_rng(seed)
    n, K = n_patients, len(PHENOTYPES)
    probs = np.array([p[1] for p in PHENOTYPES])
    z = rng.choice(K, size=n, p=probs / probs.sum())
    shift = lambda key: np.array([PHENOTYPES[k][3].get(key, 0.0) for k in z])

    # Providers: each has a specialty phenotype and a hidden quality (positive = worse care).
    prov_spec = np.arange(n_providers) % K
    prov_q = rng.normal(0, 0.8, n_providers)
    provider = np.empty(n, dtype=int)
    for i in range(n):
        pool = np.flatnonzero(prov_spec == z[i]) if rng.random() < 0.65 else np.arange(n_providers)
        provider[i] = rng.choice(pool)
    q = prov_q[provider]

    # Diagnosis codes: phenotype signatures + background noise.
    p_code = np.full((n, len(CODES)), 0.03)
    for k, ph in enumerate(PHENOTYPES):
        for c in ph[4]:
            p_code[z == k, _C[c]] = 0.55
    codes = (rng.random(p_code.shape) < p_code).astype(np.float32)

    # Demographics, vitals, labs, utilisation.
    age = np.clip(rng.normal(62, 13, n) + shift("age"), 18, 100)
    female = rng.random(n) < 0.5
    bmi = np.clip(rng.normal(28, 5, n) + shift("bmi"), 14, 60)
    sbp = rng.normal(130, 16, n) + shift("sbp")
    hr = rng.normal(80, 12, n) + shift("hr")
    hba1c = np.clip(rng.normal(5.8, 0.7, n) + shift("hba1c") * rng.lognormal(0, 0.3, n), 4, 14)
    creat = 0.9 * np.exp(rng.normal(0, 0.25, n)) - 0.15 * female + shift("creat") * rng.lognormal(0, 0.4, n)
    kappa, alpha = np.where(female, 0.7, 0.9), np.where(female, -0.241, -0.302)
    egfr = 142 * np.minimum(creat / kappa, 1) ** alpha * np.maximum(creat / kappa, 1) ** -1.2 \
        * 0.9938 ** age * np.where(female, 1.012, 1.0)
    hgb = rng.normal(13.8, 1.3, n) - 1.2 * female + shift("hgb")
    alb = np.clip(rng.normal(3.9, 0.4, n) + shift("alb"), 1.5, 5.5)
    na = rng.normal(139, 3, n) + shift("na")
    wbc = np.clip(rng.lognormal(np.log(7.5), 0.25, n) + shift("wbc"), 0.5, 40)
    prior_adm = rng.poisson(np.clip(0.5 + shift("prior_adm"), 0.05, None))
    ed = rng.poisson(0.6 + shift("ed"))
    alone = rng.random(n) < 0.22 + shift("alone")
    medicaid = rng.random(n) < 0.18 + shift("medicaid")

    # Care-process traces of provider quality (noisy per patient, informative in aggregate).
    n_meds = rng.poisson(3 + 1.2 * codes.sum(1) + np.clip(1.5 * q, 0, None))
    los = np.clip(1 + rng.gamma(2, 1.8, n) + shift("los") - 0.6 * q, 1, 45)
    followup = rng.random(n) < _sigmoid(1.0 - 1.2 * q)
    med_rec = rng.random(n) < _sigmoid(0.5 - 1.0 * q)

    # Outcome: 30-day adverse event / readmission.
    logit = (np.array([PHENOTYPES[k][2] for k in z])
             + 0.035 * (age - 62) + 0.35 * np.minimum(prior_adm, 5) + 0.2 * np.minimum(ed, 6)
             + 0.9 * np.maximum(3.3 - alb, 0) + 0.6 * np.maximum(creat - 1.4, 0)
             + 0.5 * ((n_meds >= 12) & alone) + 0.4 * ((hba1c > 9) & (egfr < 45))
             + 0.3 * (hgb < 10) + 0.3 * ~followup + 1.0 * q
             + rng.normal(0, 0.6, n))
    lo, hi = -15.0, 15.0                              # solve intercept for target prevalence
    for _ in range(60):
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if _sigmoid(logit + mid).mean() < prevalence else (lo, mid)
    true_risk = _sigmoid(logit + (lo + hi) / 2)
    y = (rng.random(n) < true_risk).astype(np.int64)

    # Realistic missingness: HbA1c mostly drawn for diabetics, albumin often not drawn.
    hba1c_obs = rng.random(n) < np.where(codes[:, _C["E11"]] > 0, 0.9, 0.45)
    alb_obs = rng.random(n) < 0.7

    clinical = pd.DataFrame({
        "age": age, "female": female, "bmi": bmi, "sbp": sbp, "heart_rate": hr,
        "hba1c": np.where(hba1c_obs, hba1c, np.nan), "creatinine": creat, "egfr": egfr,
        "hemoglobin": hgb, "albumin": np.where(alb_obs, alb, np.nan), "sodium": na, "wbc": wbc,
        "los_days": los, "prior_admissions_1y": prior_adm, "ed_visits_1y": ed,
        "n_medications": n_meds, "lives_alone": alone, "medicaid": medicaid,
        "followup_scheduled": followup, "med_reconciliation": med_rec,
    })
    meta = clinical.copy()
    meta.insert(0, "provider", provider)
    meta["phenotype"] = [PHENOTYPES[k][0] for k in z]
    meta["provider_quality"] = q
    meta["true_risk"] = true_risk

    feats = clinical.astype(float)
    for col in ["hba1c", "albumin"]:
        feats[f"{col}_missing"] = feats[col].isna().astype(float)
        feats[col] = feats[col].fillna(feats[col].median())
    x_clin = standardize(feats)
    x = np.hstack([x_clin, codes])

    return PatientData(
        x=x, y=y, feature_names=list(feats.columns) + CODES, sim_view=x_clin,
        codes=codes, code_names=CODES, group=provider, meta=meta,
        true_risk=true_risk, name="synthetic",
    )
