"""Request/response schemas with clinical input validation (ranges from the UCI data dictionary)."""
from typing import Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator

ICD9 = r"^(V\d{2}(\.\d{1,2})?|E\d{3}(\.\d)?|\d{1,3}(\.\d{1,2})?)$"
AgeBand = Literal["[0-10)", "[10-20)", "[20-30)", "[30-40)", "[40-50)", "[50-60)", "[60-70)",
                  "[70-80)", "[80-90)", "[90-100)"]
Dose = Literal["No", "Steady", "Up", "Down"]
Medication = Literal["metformin", "repaglinide", "nateglinide", "chlorpropamide", "glimepiride",
                     "acetohexamide", "glipizide", "glyburide", "tolbutamide", "pioglitazone",
                     "rosiglitazone", "acarbose", "miglitol", "troglitazone", "tolazamide", "examide",
                     "citoglipton", "glyburide-metformin", "glipizide-metformin",
                     "glimepiride-pioglitazone", "metformin-rosiglitazone", "metformin-pioglitazone"]
EXCLUDED_DISPOSITIONS = {11, 13, 14, 19, 20, 21}

EXAMPLE = {
    "age": "[70-80)", "gender": "Female", "race": "Caucasian",
    "admission_type_id": 1, "discharge_disposition_id": 1, "admission_source_id": 7,
    "time_in_hospital": 6, "num_lab_procedures": 58, "num_procedures": 1, "num_medications": 21,
    "number_outpatient": 0, "number_emergency": 1, "number_inpatient": 2, "number_diagnoses": 9,
    "diag_1": "428", "diag_2": "250.02", "diag_3": "403",
    "medical_specialty": "InternalMedicine", "max_glu_serum": "None", "A1Cresult": ">8",
    "insulin": "Up", "change": "Ch", "diabetesMed": "Yes", "medications": {"metformin": "Steady"},
}


class Patient(BaseModel):
    """One inpatient encounter at discharge, in the UCI Diabetes 130-US Hospitals format."""
    model_config = ConfigDict(extra="forbid", json_schema_extra={"examples": [EXAMPLE]})

    age: AgeBand
    gender: Literal["Female", "Male", "Unknown/Invalid"]
    race: Optional[Literal["Caucasian", "AfricanAmerican", "Hispanic", "Asian", "Other"]] = None
    admission_type_id: int = Field(ge=1, le=8, description="1 emergency, 2 urgent, 3 elective, ...")
    discharge_disposition_id: int = Field(ge=1, le=30, description="1 = discharged home")
    admission_source_id: int = Field(ge=1, le=26, description="7 = emergency room")
    time_in_hospital: int = Field(ge=1, le=14, description="length of stay in days")
    num_lab_procedures: int = Field(ge=0, le=150)
    num_procedures: int = Field(ge=0, le=6)
    num_medications: int = Field(ge=0, le=100)
    number_outpatient: int = Field(ge=0, le=60, description="outpatient visits in the prior year")
    number_emergency: int = Field(ge=0, le=80, description="emergency visits in the prior year")
    number_inpatient: int = Field(ge=0, le=30, description="inpatient admissions in the prior year")
    number_diagnoses: int = Field(ge=1, le=16)
    diag_1: str = Field(pattern=ICD9, description="primary diagnosis, ICD-9 (e.g. 428, 250.02, V57)")
    diag_2: Optional[str] = Field(default=None, pattern=ICD9)
    diag_3: Optional[str] = Field(default=None, pattern=ICD9)
    medical_specialty: Optional[str] = Field(default=None, max_length=60)
    max_glu_serum: Literal["None", "Norm", ">200", ">300"] = "None"
    A1Cresult: Literal["None", "Norm", ">7", ">8"] = "None"
    insulin: Dose = "No"
    change: Literal["No", "Ch"] = "No"
    diabetesMed: Literal["Yes", "No"] = "No"
    medications: dict[Medication, Dose] = Field(default_factory=dict,
                                                description="other diabetes drugs; omitted = 'No'")

    @field_validator("discharge_disposition_id")
    @classmethod
    def not_expired_or_hospice(cls, v):
        if v in EXCLUDED_DISPOSITIONS:
            raise ValueError("model is not applicable to patients who died or were discharged to hospice "
                             f"(disposition {v}); readmission is not a meaningful outcome for them")
        return v


class Driver(BaseModel):
    feature: str
    contribution: float


class Neighbour(BaseModel):
    relation: str
    attention: float
    age: str
    gender: str
    primary_dx: str
    primary_dx_group: str
    specialty: str
    prior_inpatient: int
    readmitted_30d: int = Field(description="1/0 for training patients, -1 if held out")


class Prediction(BaseModel):
    readmission_risk: float = Field(description="calibrated probability of readmission within 30 days")
    risk_band: Literal["low", "elevated", "high"]
    decision_threshold: float
    flag_for_follow_up: bool
    cohort_baseline_risk: float
    graph_neighbours: dict[str, int]
    subgraph_nodes: int
    top_risk_drivers: Optional[list[Driver]] = None
    most_attended_similar_patients: Optional[list[Neighbour]] = None
    self_attention: Optional[float] = None
    latency_ms: float


class BatchRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    patients: list[Patient] = Field(min_length=1, max_length=100)
    explain: bool = False
