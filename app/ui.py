"""Gradio web UI for HealthGraph (served at /ui by the FastAPI app and on Hugging Face Spaces)."""
import gradio as gr
from pydantic import ValidationError

from .schemas import EXAMPLE, Patient

AGES = ["[0-10)", "[10-20)", "[20-30)", "[30-40)", "[40-50)", "[50-60)", "[60-70)", "[70-80)", "[80-90)",
        "[90-100)"]
ADM_TYPE = {"1 Emergency": 1, "2 Urgent": 2, "3 Elective": 3, "4 Newborn": 4, "5 Not available": 5,
            "6 NULL": 6, "7 Trauma center": 7, "8 Not mapped": 8}
ADM_SOURCE = {"7 Emergency room": 7, "1 Physician referral": 1, "2 Clinic referral": 2,
              "4 Transfer from hospital": 4, "6 Transfer from other facility": 6, "17 NULL": 17}
DISCHARGE = {"1 Home": 1, "6 Home with health service": 6, "3 Skilled nursing facility": 3,
             "2 Short-term hospital": 2, "5 Other inpatient care": 5, "22 Rehab facility": 22,
             "7 Left against medical advice": 7}
SPECIALTIES = ["(unknown)", "InternalMedicine", "Emergency/Trauma", "Family/GeneralPractice", "Cardiology",
               "Surgery-General", "Nephrology", "Orthopedics", "Pulmonology", "Psychiatry", "Oncology"]

# "shared-provider" is the care-team relation; on UCI data it is the admitting specialty.
RELATION_NAMES = {"clinical-similarity": "similar profile", "shared-diagnoses": "shared diagnoses",
                  "shared-provider": "same specialty", "self": "self"}

DISCLAIMER = ("Research prototype trained on the de-identified UCI Diabetes 130-US Hospitals data (1999-2008). "
              "Not a medical device and not for clinical decision-making.")


def _first(d, value):
    return next(k for k, v in d.items() if v == value)


def build_ui(get_predictor):
    def predict(age, gender, race, adm_type, adm_source, discharge, los, labs, procs, meds, outp, emerg, inp,
                ndiag, d1, d2, d3, spec, glu, a1c, insulin, change, dmed):
        payload = {
            "age": age, "gender": gender, "race": None if race == "(unknown)" else race,
            "admission_type_id": ADM_TYPE[adm_type], "admission_source_id": ADM_SOURCE[adm_source],
            "discharge_disposition_id": DISCHARGE[discharge], "time_in_hospital": int(los),
            "num_lab_procedures": int(labs), "num_procedures": int(procs), "num_medications": int(meds),
            "number_outpatient": int(outp), "number_emergency": int(emerg), "number_inpatient": int(inp),
            "number_diagnoses": int(ndiag), "diag_1": d1.strip(), "diag_2": d2.strip() or None,
            "diag_3": d3.strip() or None, "medical_specialty": None if spec == "(unknown)" else spec,
            "max_glu_serum": glu, "A1Cresult": a1c, "insulin": insulin, "change": change, "diabetesMed": dmed,
        }
        try:
            patient = Patient(**payload)
        except ValidationError as e:
            msg = "\n".join(f"- **{'.'.join(map(str, err['loc']))}**: {err['msg']}" for err in e.errors())
            return {}, f"### Invalid input\n{msg}", None
        r = get_predictor().predict(patient.model_dump())
        band = {"low": "🟢 LOW", "elevated": "🟠 ELEVATED", "high": "🔴 HIGH"}[r["risk_band"]]
        lines = [f"## {band} risk: {r['readmission_risk']:.1%} chance of readmission within 30 days",
                 f"Cohort baseline {r['cohort_baseline_risk']:.1%} · follow-up threshold "
                 f"{r['decision_threshold']:.1%} · **{'flag for follow-up' if r['flag_for_follow_up'] else 'no flag'}**"
                 f" · scored in {r['latency_ms']:.0f} ms on a {r['subgraph_nodes']}-patient subgraph", "",
                 "### Why: the patient's own risk drivers (gradient × input)",
                 "| feature | contribution |", "|---|---|"]
        lines += [f"| {d['feature']} | {d['contribution']:+.3f} |" for d in r["top_risk_drivers"]]
        lines += ["", "### Similar past patients the model attended to most",
                  "| linked by | attention | patient | primary dx | prior stays | readmitted |",
                  "|---|---|---|---|---|---|"]
        for n in r["most_attended_similar_patients"]:
            out = {1: "yes", 0: "no", -1: "held out"}[n["readmitted_30d"]]
            sex = {"Female": "F", "Male": "M"}.get(n["gender"], "?")
            lines.append(f"| {RELATION_NAMES.get(n['relation'], n['relation'])} | {n['attention']:.3f} | "
                         f"{n['age']} {sex}, {n['specialty']} | {n['primary_dx']} ({n['primary_dx_group']}) | "
                         f"{n['prior_inpatient']} | {out} |")
        label = {"30-day readmission risk": r["readmission_risk"]}
        return label, "\n".join(lines), r

    e = EXAMPLE
    with gr.Blocks(title="HealthGraph: readmission risk") as demo:
        gr.Markdown("# HealthGraph · 30-day readmission risk with Graph Attention Networks\n"
                    "Enter a diabetic inpatient encounter at discharge. HealthGraph links the patient to "
                    "similar past patients (clinical similarity, shared diagnoses, same specialty) and a "
                    f"relation-aware GATv2 network predicts calibrated readmission risk.\n\n*{DISCLAIMER}*")
        with gr.Row():
            with gr.Column(scale=3):
                with gr.Row():
                    age = gr.Dropdown(AGES, value=e["age"], label="Age band")
                    gender = gr.Dropdown(["Female", "Male", "Unknown/Invalid"], value=e["gender"], label="Gender")
                    race = gr.Dropdown(["(unknown)", "Caucasian", "AfricanAmerican", "Hispanic", "Asian", "Other"],
                                       value=e["race"], label="Race")
                with gr.Row():
                    adm_type = gr.Dropdown(list(ADM_TYPE), value=_first(ADM_TYPE, e["admission_type_id"]),
                                           label="Admission type")
                    adm_source = gr.Dropdown(list(ADM_SOURCE), value=_first(ADM_SOURCE, e["admission_source_id"]),
                                             label="Admission source")
                    discharge = gr.Dropdown(list(DISCHARGE), value=_first(DISCHARGE, e["discharge_disposition_id"]),
                                            label="Discharge disposition")
                with gr.Row():
                    los = gr.Slider(1, 14, value=e["time_in_hospital"], step=1, label="Length of stay (days)")
                    labs = gr.Slider(0, 132, value=e["num_lab_procedures"], step=1, label="Lab procedures")
                    procs = gr.Slider(0, 6, value=e["num_procedures"], step=1, label="Procedures")
                    meds = gr.Slider(1, 81, value=e["num_medications"], step=1, label="Medications")
                with gr.Row():
                    outp = gr.Number(e["number_outpatient"], label="Outpatient visits (prior yr)", precision=0)
                    emerg = gr.Number(e["number_emergency"], label="ED visits (prior yr)", precision=0)
                    inp = gr.Number(e["number_inpatient"], label="Inpatient stays (prior yr)", precision=0)
                    ndiag = gr.Slider(1, 16, value=e["number_diagnoses"], step=1, label="Number of diagnoses")
                with gr.Row():
                    d1 = gr.Textbox(e["diag_1"], label="Primary diagnosis (ICD-9)")
                    d2 = gr.Textbox(e["diag_2"], label="Secondary diagnosis")
                    d3 = gr.Textbox(e["diag_3"], label="Additional diagnosis")
                    spec = gr.Dropdown(SPECIALTIES, value=e["medical_specialty"], label="Admitting specialty",
                                       allow_custom_value=True)
                with gr.Row():
                    glu = gr.Dropdown(["None", "Norm", ">200", ">300"], value=e["max_glu_serum"], label="Max glucose serum")
                    a1c = gr.Dropdown(["None", "Norm", ">7", ">8"], value=e["A1Cresult"], label="HbA1c result")
                    insulin = gr.Dropdown(["No", "Steady", "Up", "Down"], value=e["insulin"], label="Insulin")
                    change = gr.Dropdown(["No", "Ch"], value=e["change"], label="Medication change")
                    dmed = gr.Dropdown(["Yes", "No"], value=e["diabetesMed"], label="On diabetes medication")
                btn = gr.Button("Predict readmission risk", variant="primary")
            with gr.Column(scale=2):
                label = gr.Label(label="Calibrated probability (Platt-scaled HealthGAT)", num_top_classes=1)
                expl = gr.Markdown()
                with gr.Accordion("Raw API response (JSON)", open=False):
                    raw = gr.JSON(label="POST /predict")
        inputs = [age, gender, race, adm_type, adm_source, discharge, los, labs, procs, meds, outp, emerg, inp,
                  ndiag, d1, d2, d3, spec, glu, a1c, insulin, change, dmed]
        btn.click(predict, inputs, [label, expl, raw])
        gr.Markdown("API: `POST /predict` · `POST /predict/batch` · `GET /model/info` · docs at `/docs`")
    return demo
