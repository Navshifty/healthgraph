# HealthGAT explanations

## Attention budget by relation (test patients, last layer, head-averaged)

| | all | event=1 | event=0 |
|---|---|---|---|
| clinical-similarity | 0.353 | 0.335 | 0.356 |
| shared-diagnoses | 0.182 | 0.213 | 0.176 |
| shared-provider | 0.441 | 0.428 | 0.444 |
| self | 0.024 | 0.023 | 0.024 |

## Attention lift by neighbour outcome (1.0 = uniform)

{
  "to_positive_neighbours": 0.925,
  "to_negative_neighbours": 1.017
}

## Case studies

### Patient 1074: predicted risk 61.7%, actual outcome = 1
Profile: age=70.59, phenotype=frail-elderly, provider=30, provider_quality=1.60, prior_admissions_1y=1, albumin=3.43, creatinine=0.92, n_medications=14, followup_scheduled=False

Top-5 attended neighbours (of 35; self-loop weight 0.018):

| neighbour | relation | attention | neighbour risk | neighbour outcome |
|---|---|---|---|---|
| 1410 | shared-diagnoses | 0.070 | 6.4% | 0 |
| 1888 | shared-diagnoses | 0.057 | 7.0% | 0 |
| 1902 | shared-diagnoses | 0.038 | 27.7% | 0 |
| 4329 | clinical-similarity | 0.038 | 7.6% | 1 |
| 5821 | shared-provider | 0.036 | 33.6% | 1 |

Own-feature drivers (gradient x input, + raises risk):

- med_reconciliation: -0.132
- age: +0.112
- ed_visits_1y: +0.104
- hba1c: +0.086
- prior_admissions_1y: +0.078
- W19 Fall: +0.064
- n_medications: -0.059
- sodium: -0.057

### Patient 938: predicted risk 60.3%, actual outcome = 1
Profile: age=74.02, phenotype=frail-elderly, provider=30, provider_quality=1.60, prior_admissions_1y=1, albumin=nan, creatinine=1.05, n_medications=14, followup_scheduled=False

Top-5 attended neighbours (of 40; self-loop weight 0.022):

| neighbour | relation | attention | neighbour risk | neighbour outcome |
|---|---|---|---|---|
| 2269 | shared-provider | 0.052 | 12.7% | 0 |
| 5771 | shared-diagnoses | 0.046 | 12.2% | 0 |
| 2376 | shared-diagnoses | 0.034 | 31.0% | 1 |
| 5765 | clinical-similarity | 0.032 | 11.1% | 0 |
| 433 | shared-diagnoses | 0.030 | 42.6% | 1 |

Own-feature drivers (gradient x input, + raises risk):

- age: +0.175
- med_reconciliation: -0.138
- n_medications: -0.113
- heart_rate: -0.099
- prior_admissions_1y: +0.097
- R26 Gait abnormality: +0.082
- W19 Fall: +0.077
- followup_scheduled: +0.076

### Patient 1922: predicted risk 58.1%, actual outcome = 1
Profile: age=67.40, phenotype=renal, provider=82, provider_quality=0.98, prior_admissions_1y=1, albumin=3.71, creatinine=1.13, n_medications=14, followup_scheduled=False

Top-5 attended neighbours (of 39; self-loop weight 0.020):

| neighbour | relation | attention | neighbour risk | neighbour outcome |
|---|---|---|---|---|
| 1366 | shared-provider | 0.057 | 2.1% | 0 |
| 244 | clinical-similarity | 0.047 | 15.8% | 0 |
| 1120 | shared-provider | 0.040 | 32.2% | 0 |
| 4543 | shared-provider | 0.038 | 13.9% | 0 |
| 4161 | shared-diagnoses | 0.033 | 14.9% | 0 |

Own-feature drivers (gradient x input, + raises risk):

- ed_visits_1y: +0.122
- prior_admissions_1y: +0.109
- age: +0.093
- F17 Nicotine dependence: +0.090
- sodium: -0.075
- med_reconciliation: +0.058
- F03 Dementia: +0.044
- followup_scheduled: +0.039
