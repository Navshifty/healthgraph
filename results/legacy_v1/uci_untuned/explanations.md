# HealthGAT explanations

## Attention budget by relation (test patients, last layer, head-averaged)

| | all | event=1 | event=0 |
|---|---|---|---|
| clinical-similarity | 0.478 | 0.477 | 0.478 |
| shared-diagnoses | 0.262 | 0.261 | 0.262 |
| shared-provider | 0.221 | 0.223 | 0.221 |
| self | 0.038 | 0.039 | 0.038 |

## Attention lift by neighbour outcome (1.0 = uniform)

{
  "to_positive_neighbours": 1.007,
  "to_negative_neighbours": 0.998
}

## Case studies

### Patient 17694: predicted risk 28.5%, actual outcome = 1
Profile: age=[20-30), gender=Female, medical_specialty=nan, diag_1=250.6, time_in_hospital=5, number_inpatient=12, number_emergency=1, num_medications=8

Top-5 attended neighbours (of 17; self-loop weight 0.057):

| neighbour | relation | attention | neighbour risk | neighbour outcome |
|---|---|---|---|---|
| 10347 | shared-diagnoses | 0.067 | 14.1% | 0 |
| 17199 | shared-diagnoses | 0.067 | 10.2% | 0 |
| 3070 | shared-diagnoses | 0.063 | 7.7% | 0 |
| 4062 | clinical-similarity | 0.061 | 23.3% | 0 |
| 149 | shared-diagnoses | 0.059 | 6.4% | 0 |

Own-feature drivers (gradient x input, + raises risk):

- number_inpatient: +0.841
- number_emergency: +0.049
- discharge_disposition_id_1: -0.035
- age: -0.023
- insulin_Steady: +0.021
- diabetesMed_Yes: +0.020
- ICD9 536: +0.014
- gender_Male: +0.011

### Patient 17211: predicted risk 23.8%, actual outcome = 1
Profile: age=[80-90), gender=Female, medical_specialty=InternalMedicine, diag_1=491, time_in_hospital=9, number_inpatient=6, number_emergency=1, num_medications=25

Top-5 attended neighbours (of 39; self-loop weight 0.027):

| neighbour | relation | attention | neighbour risk | neighbour outcome |
|---|---|---|---|---|
| 9758 | shared-provider | 0.032 | 9.8% | 0 |
| 4396 | shared-provider | 0.031 | 10.0% | 0 |
| 19729 | shared-provider | 0.030 | 6.4% | 0 |
| 18827 | shared-diagnoses | 0.029 | 7.6% | 0 |
| 10111 | shared-provider | 0.028 | 9.9% | 0 |

Own-feature drivers (gradient x input, + raises risk):

- number_inpatient: +0.485
- discharge_disposition_id_1: +0.082
- number_emergency: +0.076
- admission_type_id_2: +0.058
- ICD9 491: -0.052
- discharge_disposition_id_6: -0.042
- ICD9 428: +0.038
- diabetesMed_Yes: +0.026

### Patient 9904: predicted risk 23.2%, actual outcome = 1
Profile: age=[80-90), gender=Female, medical_specialty=nan, diag_1=153, time_in_hospital=6, number_inpatient=2, number_emergency=0, num_medications=38

Top-5 attended neighbours (of 15; self-loop weight 0.061):

| neighbour | relation | attention | neighbour risk | neighbour outcome |
|---|---|---|---|---|
| 10501 | shared-diagnoses | 0.084 | 9.0% | 0 |
| 5968 | clinical-similarity | 0.082 | 15.6% | 0 |
| 19527 | clinical-similarity | 0.072 | 14.5% | 0 |
| 2212 | clinical-similarity | 0.071 | 12.2% | 0 |
| 549 | clinical-similarity | 0.070 | 17.9% | 0 |

Own-feature drivers (gradient x input, + raises risk):

- discharge_disposition_id_22: +0.571
- number_inpatient: +0.235
- ICD9 153: -0.105
- discharge_disposition_id_1: +0.072
- insulin_Up: -0.048
- admission_type_id_3: -0.043
- num_medications: -0.042
- ICD9 707: +0.041
