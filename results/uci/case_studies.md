# HealthGAT case studies (seed 0, test set)

### Patient 1211: predicted risk 30.8%, actual outcome = 1
Profile: age=[40-50), gender=Female, medical_specialty=nan, diag_1=198, primary_dx_group=neoplasms, time_in_hospital=10, number_inpatient=7, number_emergency=0, num_medications=19

Top-5 attended neighbours (of 26; self-loop weight 0.027):

| neighbour | relation | attention | neighbour risk | neighbour outcome |
|---|---|---|---|---|
| 11589 | shared-diagnoses | 0.052 | 12.8% | 0 |
| 6083 | shared-diagnoses | 0.048 | 19.5% | 0 |
| 11105 | shared-diagnoses | 0.047 | 13.7% | 0 |
| 7385 | shared-diagnoses | 0.045 | 12.6% | 0 |
| 16822 | shared-diagnoses | 0.044 | 16.7% | 0 |

Own-feature drivers (gradient x input, + raises risk):

- number_inpatient: +0.630
- n_dx_neoplasms: +0.183
- total_prior_visits: +0.168
- ICD9 198: +0.143
- admission_source_id=20: -0.143
- diag_1_target_enc: +0.122
- time_in_hospital: +0.116
- ICD9 162: +0.102

### Patient 16628: predicted risk 29.6%, actual outcome = 1
Profile: age=[60-70), gender=Male, medical_specialty=InternalMedicine, diag_1=996, primary_dx_group=injury, time_in_hospital=14, number_inpatient=6, number_emergency=0, num_medications=31

Top-5 attended neighbours (of 39; self-loop weight 0.029):

| neighbour | relation | attention | neighbour risk | neighbour outcome |
|---|---|---|---|---|
| 414 | shared-diagnoses | 0.035 | 6.6% | 0 |
| 13435 | shared-diagnoses | 0.034 | 9.9% | 0 |
| 15148 | shared-diagnoses | 0.032 | 13.6% | 0 |
| 12932 | shared-diagnoses | 0.030 | 10.3% | 0 |
| 16892 | shared-diagnoses | 0.029 | 7.9% | 0 |

Own-feature drivers (gradient x input, + raises risk):

- number_inpatient: +0.624
- time_in_hospital: +0.211
- total_prior_visits: +0.151
- discharge_disposition_id=3: +0.108
- diag_2_freq: -0.066
- ICD9 996: -0.043
- insulin=Up: +0.040
- A1Cresult=None: +0.033

### Patient 9101: predicted risk 27.1%, actual outcome = 1
Profile: age=[80-90), gender=Female, medical_specialty=InternalMedicine, diag_1=491, primary_dx_group=respiratory, time_in_hospital=9, number_inpatient=6, number_emergency=1, num_medications=25

Top-5 attended neighbours (of 54; self-loop weight 0.019):

| neighbour | relation | attention | neighbour risk | neighbour outcome |
|---|---|---|---|---|
| 4255 | shared-provider | 0.024 | 11.8% | 1 |
| 2117 | shared-diagnoses | 0.023 | 10.6% | 0 |
| 16320 | shared-diagnoses | 0.023 | 9.2% | 0 |
| 2053 | shared-provider | 0.023 | 10.5% | 0 |
| 19240 | shared-provider | 0.022 | 6.0% | 0 |

Own-feature drivers (gradient x input, + raises risk):

- number_inpatient: +0.681
- total_prior_visits: +0.186
- time_in_hospital: +0.112
- number_outpatient: -0.072
- admission_type_id=2: +0.069
- discharge_disposition_id=6: -0.048
- ICD9 428: +0.043
- ICD9 491: -0.042
