// IEEE-format project report (.docx) for HealthGraph, following the CAT II guideline structure (Sections 1-8).
const fs = require("fs");
const path = require("path");
const {
  Document, Packer, Paragraph, TextRun, ImageRun, Table, TableRow, TableCell, AlignmentType, WidthType,
  BorderStyle, SectionType, HeadingLevel, ShadingType, VerticalAlign, LineRuleType, ExternalHyperlink,
} = require("docx");

const ROOT = "E:/AML/lab/healthgraph";
const HERE = __dirname;
const D = JSON.parse(fs.readFileSync(path.join(HERE, "report_data.json"), "utf8"));
const M = D.macros;
const FIG = (f) => path.join(ROOT, "results/uci/figures", f);
const SYN = (f) => path.join(ROOT, "results/synthetic/figures", f);
const HPO = (f) => path.join(ROOT, "results/uci/hpo", f);

// ---------- page geometry (A4, IEEE-like margins) ----------
const PAGE = { size: { width: 11906, height: 16838 }, margin: { top: 1080, bottom: 1080, left: 1008, right: 1008 } };
const FULL = 11906 - 2 * 1008;           // 9890 DXA
const GAP = 360;
const COL = (FULL - GAP) / 2;            // 4765 DXA
const DXA_PER_IN = 1440;
const FONT = "Times New Roman";

// ---------- inline markup: **bold**, *italic* ----------
function runs(text, base = {}) {
  const out = [];
  const re = /(\*\*[^*]+\*\*|\*[^*]+\*)/g;
  let last = 0, m;
  while ((m = re.exec(text)) !== null) {
    if (m.index > last) out.push(new TextRun({ text: text.slice(last, m.index), ...base }));
    const t = m[0];
    if (t.startsWith("**")) out.push(new TextRun({ text: t.slice(2, -2), bold: true, ...base }));
    else out.push(new TextRun({ text: t.slice(1, -1), italics: true, ...base }));
    last = m.index + t.length;
  }
  if (last < text.length) out.push(new TextRun({ text: text.slice(last), ...base }));
  return out;
}
const P = (text, o = {}) => new Paragraph({
  children: runs(text, o.run || {}), alignment: o.align || AlignmentType.JUSTIFIED,
  indent: o.noIndent ? undefined : { firstLine: 245 }, spacing: { after: o.after ?? 0, before: o.before ?? 0 },
  keepNext: o.keepNext,
});
const LEAD = (lead, text) => new Paragraph({
  children: [new TextRun({ text: lead + " ", bold: true }), ...runs(text)],
  alignment: AlignmentType.JUSTIFIED, indent: { firstLine: 245 }, spacing: { after: 0 },
});

let secNo = 0, subNo = 0, figNo = 0, tabNo = 0;
const ROMAN = ["I", "II", "III", "IV", "V", "VI", "VII", "VIII", "IX", "X"];
const H1 = (title) => { subNo = 0; return new Paragraph({
  heading: HeadingLevel.HEADING_1, alignment: AlignmentType.CENTER, spacing: { before: 200, after: 100 }, keepNext: true,
  children: [new TextRun({ text: `${ROMAN[secNo++]}. ${title}`, smallCaps: true, size: 20, font: FONT, color: "000000" })] }); };
const H2 = (title) => new Paragraph({
  heading: HeadingLevel.HEADING_2, spacing: { before: 120, after: 60 }, keepNext: true,
  children: [new TextRun({ text: `${String.fromCharCode(65 + subNo++)}. ${title}`, italics: true, size: 20, font: FONT, color: "000000" })] });

// ---------- images ----------
function pngSize(file) {
  const b = fs.readFileSync(file);
  return { w: b.readUInt32BE(16), h: b.readUInt32BE(20) };
}
function img(file, widthIn) {
  const { w, h } = pngSize(file);
  const wpx = Math.round(widthIn * 96);
  return new ImageRun({ type: "png", data: fs.readFileSync(file), transformation: { width: wpx, height: Math.round(wpx * h / w) },
                        altText: { title: path.basename(file), description: path.basename(file), name: path.basename(file) } });
}
function figure(file, widthIn, caption) {
  figNo++;
  return [
    new Paragraph({ children: [img(file, widthIn)], alignment: AlignmentType.CENTER, spacing: { before: 80, after: 40 }, keepNext: true }),
    new Paragraph({ children: [new TextRun({ text: `Fig. ${figNo}. `, size: 16 }), ...runs(caption, { size: 16 })],
                    alignment: AlignmentType.JUSTIFIED, spacing: { after: 120 } }),
  ];
}
const NOB = { style: BorderStyle.NONE, size: 0, color: "FFFFFF" };
function figurePair(f1, f2, totalDxa, caption) {
  figNo++;
  const half = totalDxa / 2, wIn = (half - 120) / DXA_PER_IN;
  const cell = (f) => new TableCell({ width: { size: half, type: WidthType.DXA }, borders: { top: NOB, bottom: NOB, left: NOB, right: NOB },
    children: [new Paragraph({ children: [img(f, wIn)], alignment: AlignmentType.CENTER, keepNext: true })] });
  return [
    new Table({ width: { size: totalDxa, type: WidthType.DXA }, columnWidths: [half, half],
                borders: { top: NOB, bottom: NOB, left: NOB, right: NOB, insideHorizontal: NOB, insideVertical: NOB },
                rows: [new TableRow({ children: [cell(f1), cell(f2)] })] }),
    new Paragraph({ children: [new TextRun({ text: `Fig. ${figNo}. `, size: 16 }), ...runs(caption, { size: 16 })],
                    alignment: AlignmentType.JUSTIFIED, spacing: { before: 40, after: 120 } }),
  ];
}

// ---------- IEEE tables: caption above, horizontal rules only ----------
const RULE = { style: BorderStyle.SINGLE, size: 6, color: "000000" };
function table(title, header, rows, widths, opts = {}) {
  tabNo++;
  const total = widths.reduce((a, b) => a + b, 0);
  const fs8 = opts.size || 15;
  const cell = (t, i, r, isHead) => new TableCell({
    width: { size: widths[i], type: WidthType.DXA }, verticalAlign: VerticalAlign.CENTER,
    margins: { top: 20, bottom: 20, left: 50, right: 50 },
    borders: { left: NOB, right: NOB, top: isHead ? RULE : NOB,
               bottom: isHead || r === rows.length - 1 ? RULE : NOB },
    children: [new Paragraph({ alignment: i === 0 ? AlignmentType.LEFT : AlignmentType.CENTER, keepNext: r < rows.length - 1,
      children: [new TextRun({ text: t, size: fs8, bold: isHead || (opts.boldRow && opts.boldRow(rows[r] || [])) && i === 0 })] })],
  });
  return [
    new Paragraph({ alignment: AlignmentType.CENTER, spacing: { before: 120, after: 20 }, keepNext: true,
                    children: [new TextRun({ text: `TABLE ${ROMAN[tabNo - 1]}`, size: 16, smallCaps: true })] }),
    new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 60 }, keepNext: true,
                    children: [new TextRun({ text: title, size: 16, smallCaps: true })] }),
    new Table({ width: { size: total, type: WidthType.DXA }, columnWidths: widths,
      rows: [new TableRow({ tableHeader: true, cantSplit: true, children: header.map((h, i) => cell(h, i, -1, true)) }),
             ...rows.map((r, ri) => new TableRow({ cantSplit: true, children: r.map((t, i) => cell(t, i, ri, false)) }))] }),
    new Paragraph({ children: [], spacing: { after: 100 } }),
  ];
}

// ---------- section assembly ----------
const sections = [];
let buf = [];
function flush(cols, first = false) {
  if (!buf.length) return;
  sections.push({ properties: { type: first ? SectionType.NEXT_PAGE : SectionType.CONTINUOUS, page: PAGE,
                                column: cols === 2 ? { count: 2, space: GAP, equalWidth: true } : { count: 1 } },
                  children: buf });
  buf = [];
}
const add = (...xs) => xs.flat().forEach((x) => buf.push(x));
function wide(...xs) { flush(2); add(...xs); flush(1); }

// =================== TITLE BLOCK ===================
add(new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 160 },
  children: [new TextRun({ text: "HealthGraph: Patient Risk Prediction Using Relation-Aware Graph Attention Networks", size: 44 })] }));
add(new Paragraph({ alignment: AlignmentType.CENTER, children: [new TextRun({ text: "Naveenya V K", size: 22 })] }));
add(new Paragraph({ alignment: AlignmentType.CENTER, children: [new TextRun({ text: "Roll No. 2403717674422030", size: 20 })] }));
add(new Paragraph({ alignment: AlignmentType.CENTER, children: [new TextRun({ text: "III M.Sc. AI & ML, Department of Computing – AI & ML", size: 20, italics: true })] }));
add(new Paragraph({ alignment: AlignmentType.CENTER, children: [new TextRun({ text: "Coimbatore Institute of Technology, Coimbatore – 641 014", size: 20, italics: true })] }));
add(new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 240 },
  children: [new TextRun({ text: "19MAM56 Advanced Machine Learning Lab – CAT II Project", size: 20, italics: true })] }));
flush(1, true);

// =================== ABSTRACT ===================
const absText = `Thirty-day readmission is a key quality and cost indicator. We present HealthGraph, which models a cohort of diabetic inpatients as a multi-relational graph (clinical similarity, shared diagnoses, shared specialty) and predicts readmission with HealthGAT, a relation-aware GATv2 network implemented from scratch. On the UCI Diabetes-130 dataset we apply leakage-safe feature engineering (out-of-fold target encoding of high-cardinality ICD-9 codes, PCA), tune all models with Optuna, and track experiments in MLflow. Under the published evaluation protocol our pipeline reaches AUROC ${M.benchPOneXGB}, matching or exceeding prior work (0.667). On a strict first-encounter cohort, HealthGAT (AUROC ${M.aurocGAT}, ECE ${M.eceGAT}) performs on par with logistic regression and gradient boosting; no difference is significant under DeLong tests. Homophily analysis, relation ablation and a controlled synthetic study, where HealthGAT clearly wins (${M.synGAT} against ${M.synMLP}), show that the graph helps only when its edges carry outcome signal beyond the node features. The model is served through a validated FastAPI/Gradio application that scores unseen patients inductively, containerised with Docker and deployed on AWS EC2.`;
add(new Paragraph({ alignment: AlignmentType.JUSTIFIED, spacing: { after: 80 },
  children: [new TextRun({ text: "Abstract—", bold: true, italics: true, size: 18 }), new TextRun({ text: absText, bold: true, size: 18 })] }));
add(new Paragraph({ alignment: AlignmentType.JUSTIFIED, spacing: { after: 120 },
  children: [new TextRun({ text: "Index Terms—", bold: true, italics: true, size: 18 }),
             new TextRun({ text: "graph attention networks, GATv2, hospital readmission, patient similarity graph, diabetes, calibration, Optuna, MLflow, FastAPI, Docker, AWS.", bold: true, size: 18 })] }));

// =================== I. INTRODUCTION ===================
add(H1("Introduction"));
add(P(`Unplanned hospital readmission within 30 days is a widely used quality indicator. It is costly and often preventable with better discharge planning. Diabetic inpatients are a large, high-risk group. Strack et al. [1] released 101,766 diabetic encounters from 130 US hospitals (1999–2008) and found that inpatient HbA1c testing, done in only 18.4% of encounters, was associated with readmission. The data has become the standard public benchmark for readmission modelling, yet published models plateau at an AUROC of about 0.63–0.69 [2], [3], and clinical scores such as LACE reach about 0.61 on comparable tasks [4], [5].`));
add(P(`Almost all published models score each encounter independently. Many of the strongest risk factors, though, are *latent and shared*: a disease phenotype, the practice style of a care team, the burden of a comorbidity cluster. A single record shows only a noisy trace of each, while patients who share diagnoses or a care team pool that evidence. Graph neural networks (GNNs) use exactly this structure: message passing averages noisy evidence over related patients, and attention [7], [8] learns *which* neighbours and *which* relation to trust.`));
add(P(`**Real-world impact.** A well-calibrated risk score at discharge lets hospitals target follow-up calls, medication reconciliation and early clinic visits to the patients most likely to return, which reduces cost and avoidable harm.`));
add(P(`**Objectives.** (i) Build a reproducible, leakage-free pipeline on a real public dataset, with advanced feature engineering and dimensionality reduction. (ii) Design HealthGAT, a relation-aware GATv2 over a multi-relational patient graph, implemented from scratch. (iii) Tune every model with Optuna and track all runs in MLflow. (iv) Benchmark against published results *under their own evaluation protocols*, with significance tests and error analysis. (v) Deploy a validated real-time inference API and a containerised web UI to the cloud.`));

// =================== II. LITERATURE REVIEW ===================
add(H1("Literature Review"));
add(LEAD("Readmission prediction on Diabetes-130.", `Strack et al. [1] used multivariable logistic regression to study HbA1c testing. Recent machine-learning studies report AUROCs of 0.579–0.667: XGBoost 0.667, LR 0.642, RF 0.630 and a deep network 0.579 with an 80/20 split [2]; and XGBoost 0.664, LightGBM 0.660, LR 0.657 and a stacking ensemble 0.665 with nested cross-validation [3]. Both studies use *all* encounters, so the same patient can appear in training and test folds; [3] acknowledges this explicitly. Gradient-boosted trees [14], [15] are consistently the strongest family, and deep networks have not outperformed them.`));
add(LEAD("Clinical scores.", `LACE [5] and similar indices are simple and interpretable but discriminate poorly; a systematic review found most readmission models reach C-statistics of only 0.6–0.7 [6].`));
add(LEAD("Graph learning in healthcare.", `GCN [9] introduced neighbourhood averaging, R-GCN [10] added relation-specific weights, and GAT [7] learned neighbour weights with attention. GATv2 [8] showed that GAT's attention is *static* and fixed this by applying the non-linearity before the attention vector. In healthcare, GRAM [11] and the Graph Convolutional Transformer [12] model medical-code graphs, and MM-STGNN [4] builds an admission-similarity graph that reaches AUROC 0.79 for 30-day readmission on Mayo Clinic and MIMIC-IV data, versus 0.61 for LACE+.`));
add(LEAD("Research gaps.", `(1) No published GNN study on Diabetes-130 that we found. (2) Benchmarks mix evaluation protocols with possible patient leakage. (3) GNN papers rarely test whether the graph is *homophilous* for the outcome, although GNN gains depend on it strongly [13]. (4) Few systems address inductive deployment for unseen patients.`));

// =================== III. METHODOLOGY ===================
add(H1("Methodology"));
add(P(`The complete pipeline is shown in Fig. 1. Every fitted statistic uses training rows only, and the same fitted preprocessor and reference graph are reused at inference time.`));
wide(figure(path.join(HERE, "fig_arch.png"), 6.8, "Block diagram of the HealthGraph pipeline: data and cohort definition, split-aware feature engineering, PCA similarity view, multi-relational graph, HealthGAT, calibration, evaluation, hyperparameter search and deployment."));
add(H2("Data Preprocessing Pipeline"));
add(P(`The target is y = 1 if readmitted = "<30". The main cohort keeps the *first encounter per patient* and removes encounters ending in death or hospice, giving 69,973 patients with prevalence 9.0%. Full-batch GAT training on CPU limits the graph experiments to a stratified 20,000-patient subsample; tabular models are also evaluated on the full cohort (Sec. V-B). Splits are stratified 70/15/15 and repeated over three seeds.`));
add(P(`Feature engineering is fitted on training rows only. *Numeric*: eight utilisation counts, log(1+x) for heavy-tailed prior-visit counts, engineered totals (prior visits, number of diabetes drugs, dose changes), z-scored. *Low-cardinality categoricals*: one-hot, rare levels (<20) dropped. *High-cardinality*: the admitting specialty (72 levels) and three ICD-9 diagnoses (700+ codes) receive log-frequency encoding and smoothed target encoding μ̂ₖ = (Σᵢ∈ₖ yᵢ + m·ȳ)/(nₖ + m), m = 30, computed *out-of-fold* (5 folds) for training rows so no row sees its own label. *Domain grouping*: ICD-9 chapters following [1]. *Codes*: multi-hot of the 100 most frequent 3-character ICD-9 categories. In total there are ${M.nFeatures} input features. Missing values (49% of specialties; HbA1c not measured in 83% of encounters) are kept as explicit categories because absence is informative.`));
add(P(`**Dimensionality reduction.** PCA on the label-free block yields a 16-dimensional *similarity view* that keeps ${M.pcaVar}% of the variance (Fig. 2, left) and is used to build k-NN edges. t-SNE [25] and UMAP [26] project the learned embeddings for analysis.`));
add(H2("Multi-Relational Patient Graph"));
add(P(`Each patient is a node, linked by three relations: r₁ clinical similarity (k_s-NN in PCA space), r₂ shared diagnoses (k_c-NN, cosine, on ICD codes) and r₃ shared admitting specialty (k_g random peers, a care-team proxy). Edges are symmetrised and self-loops get their own relation type. Edges use features only, so they cannot leak labels. We measure *outcome homophily* per relation, P(yⱼ=1 | yᵢ=1, (j,i)∈r), against the base rate (Fig. 2, right).`));
add(H2("Mathematical Formulation of HealthGAT"));
add(P(`An encoder maps xᵢ to hᵢ⁽⁰⁾ = ELU(W₀xᵢ). Each relation-aware GATv2 layer computes, per head h:`));
add(new Paragraph({ children: [img(path.join(HERE, "eq_gat.png"), 3.1)], alignment: AlignmentType.CENTER, spacing: { before: 60, after: 60 } }));
add(P(`where r_τ is a learned embedding of the edge's relation, letting each head weigh similarity, diagnosis and specialty neighbours differently at far fewer parameters than R-GCN [10]. Applying the non-linearity before aₕ gives dynamic attention [8]. Each layer uses a residual connection, LayerNorm and dropout, and a jumping-knowledge head maps [h⁽⁰⁾ ‖ h⁽ᴸ⁾] to a logit. Training minimises class-weighted binary cross-entropy with AdamW, gradient clipping, DropEdge [16] and early stopping on validation AUPRC. Because the class weighting shifts all logits, Platt scaling [18] p = σ(az+b) is fitted on the validation split (temperature scaling [17] cannot correct a shift); the threshold maximises validation F1.`));
add(H2("Hyperparameter Search Strategy"));
add(P(`Optuna [19] with a TPE sampler maximises validation AUPRC on the seed-0 split. For HealthGAT the space covers hidden size {16, 32}, heads {2, 4}, layers {1, 2, 3}, dropout [0.2, 0.6], DropEdge [0, 0.4], learning rate [3×10⁻⁴, 5×10⁻³] (log), weight decay [10⁻⁴, 2×10⁻²] (log) and graph density k_s ∈ {5, 10, 20}, k_c ∈ {3, 5, 10}, k_g ∈ {0, 4, 8}, with median pruning after 30 epochs. XGBoost, LightGBM and RF have their own TPE studies; logistic regression uses exhaustive grid search over C. The test split is never used for selection. Every trial, per-epoch curve, metric and figure is logged to MLflow [20].`));

// =================== IV. IMPLEMENTATION & EXPERIMENTS ===================
add(H1("Implementation and Experiments"));
add(LEAD("Environment and computational setup.", `AMD Ryzen 7 7445HS (6 cores / 12 threads), 15 GB RAM, no GPU; Windows 11; Python 3.14, PyTorch 2.14 (CPU), scikit-learn 1.8, XGBoost 3.2, LightGBM 4.6, Optuna 5.0, MLflow 3.17, SHAP 0.52. All models train full-batch on CPU.`));
add(LEAD("Experimental workflow.", `A Python package driven by YAML configs runs the stages tune → experiment → ablation → analyze → benchmark (python -m healthgraph all). Each seed re-fits the preprocessor and rebuilds the graph on its training split, and every stage resumes from checkpoints. Twenty-two pytest tests check the layer mathematics, the exactness of inductive inference and the API contract.`));
add(LEAD("Tuning outcome.", `The HealthGAT study ran ${M.nTrials} trials (${M.nComplete} completed, ${M.nPruned} pruned, ${M.nInterrupted} interrupted). The best configuration (validation AUPRC ${M.bestValAuprc}) uses ${M.besthidden} hidden units × ${M.bestheads} heads, ${M.bestlayers} layers, dropout ${M.bestdropout}, DropEdge ${M.bestedgedropout}, learning rate ${M.bestlr}, weight decay ${M.bestweightdecay}, and k_s=${M.bestKsim}, k_c=${M.bestKcode}, k_g=${M.bestKgroup} (Fig. 3). The tuned graph has ${M.nEdges} directed edges over ${M.nRef} patients.`));
add(figurePair(FIG("pca_variance.png"), FIG("homophily.png"), COL, `Left: PCA spectrum of the similarity view. Right: outcome homophily per relation against the ${M.prevalence}% base rate.`));
wide(figurePair(HPO("history_gat.png"), HPO("importance_gat.png"), FULL * 0.8, "Optuna search for HealthGAT: optimisation history (left) and fANOVA hyperparameter importance (right)."),
     figure(FIG("eda.png"), 6.8, "Exploratory analysis of the 20k-patient cohort: 9.0% prevalence; readmission rises with age and steeply with prior inpatient stays; HbA1c is not measured in most encounters and the specialty is missing for about half."),
     figure(FIG("training_curves.png"), 6.8, "Training dynamics (seed 0): training loss, validation loss and validation AUPRC (the early-stopping criterion) for MLP, GCN and HealthGAT."));
add(LEAD("Feature importance and model behaviour.", `SHAP [24] for ${M.shapModel} (Fig. 6) and gradient×input for HealthGAT rank the same drivers at the top: prior inpatient stays, discharge home (protective), length of stay, discharge to nursing or rehabilitation, and the engineered diagnosis target/frequency encodings, confirming the value of the high-cardinality encodings. The embedding projections (Fig. 7) show HealthGAT ordering patients along a risk axis not visible in the raw features.`));
add(figure(FIG("shap_tree_model.png"), 3.2, `SHAP summary for ${M.shapModel} (2,000 seed-0 test patients).`));
wide(figure(FIG("embeddings.png"), 6.6, "t-SNE of raw input features against t-SNE and UMAP of the learned HealthGAT representation (2,500 test patients; red = readmitted)."));
add(LEAD("Deployment.", `A trained GNN must score patients it has never seen. The deployment bundle stores the fitted preprocessor, the reference cohort's features and edges, the weights, Platt parameters and threshold. A new encounter is preprocessed with the same transform and attached to the reference graph with the training rules; because an L-layer GNN only sees L hops, the forward pass runs on that subgraph alone, and a unit test confirms it equals a full-graph pass to within 10⁻⁴. A FastAPI service exposes POST /predict, POST /predict/batch, GET /model/info and GET /health, with Pydantic input validation (enumerated codes, numeric ranges from the data dictionary, an ICD-9 regex, rejection of unknown fields, and refusal of death/hospice dispositions) returning HTTP 422 per-field errors. Responses include the calibrated risk, risk band, follow-up flag, top risk drivers and the most-attended similar patients. A Gradio UI (Fig. 8) is mounted at /ui. The Docker image (python:3.12-slim, CPU PyTorch, non-root, health check) is deployed on AWS EC2 (free-tier t3.micro, Amazon Linux 2023, ap-south-1) by deploy/deploy_aws_ec2.sh, which uses a private S3 bucket and an HTTP-only security group. The live service returns the same predictions as the local container in about 0.4 s. Live UI: http://15.252.73.160/ui — API docs: http://15.252.73.160/docs.`));
wide(figure(path.join(ROOT, "report/figures/ui_screenshot.png"), 4.1, "HealthGraph web UI: encounter form, calibrated risk and band, the patient's own risk drivers, and the most-attended similar past patients."));

// =================== V. RESULTS ===================
add(H1("Results and Comparative Analysis"));
add(H2("Discrimination, Calibration and Significance"));
add(P(`Table I gives test-set results on the strict cohort. All seven models land in a narrow band (AUROC ${M.aurocXGB}–${M.aurocLR}). The best AUROC is logistic regression's (${M.aurocLR}±${M.aurocsdLR}), with the MLP (${M.aurocMLP}), GCN (${M.aurocGCN}) and HealthGAT (${M.aurocGAT}±${M.aurocsdGAT}) close behind; random forest has the best AUPRC (${M.auprcRF}). Seed-to-seed spread (sd ≈ 0.02–0.03) is as large as the gaps between models. Paired DeLong tests [21], [22] (Table II) find no significant difference between HealthGAT and any baseline in any seed, apart from one borderline p = 0.047 against the MLP, expected by chance across 18 tests. Tuned XGBoost had the best *validation* AUPRC yet the lowest test AUROC (${M.aurocXGB}): single-split tuning overfits that split.`));
add(P(`After Platt scaling every model is well calibrated (ECE ≤ 0.006, Fig. 9). At the tuned threshold HealthGAT flags ${M.sensGAT}% of readmissions with PPV ${M.ppvGAT}% (≈1.5× the base rate), and its top risk decile captures ${M.topdecGAT}% of readmissions. Decision-curve analysis [23] (Fig. 13) shows positive net benefit over "treat all" for thresholds of roughly 5–20%.`));
const mainHead = ["Model", "AUROC", "AUPRC", "LogLoss", "Brier", "ECE", "F1", "Sens.", "PPV", "Top-10%"];
wide(table("Test-set results on UCI Diabetes-130 (strict cohort, 20k patients; mean±sd over 3 seeds)", mainHead, D.main,
           [2050, 1060, 1060, 820, 760, 760, 760, 760, 760, 1100], { boldRow: (r) => r[0] === "HealthGAT" }),
     table("Uncertainty and paired significance tests (95% bootstrap CIs on the seed-0 test set; ΔAUROC = HealthGAT − model, mean over seeds; DeLong p per seed)",
           ["Model", "AUROC (seed 0)", "AUROC 95% CI", "AUPRC 95% CI", "ΔAUROC", "DeLong p (seeds 0,1,2)"], D.sig,
           [2050, 1300, 1600, 1600, 1100, 2240]),
     figure(FIG("roc_pr_calibration.png"), 6.8, "Seed-0 test set: ROC (left), precision–recall (middle) and reliability diagram after Platt scaling (right)."));
add(figure(FIG("confusion_matrices.png"), 3.2, "Confusion matrices at validation-tuned (max-F1) thresholds, seed-0 test set."));
add(H2("Comparison with Published Benchmarks"));
add(P(`Published numbers use all encounters with encounter-level splits [2], [3], so we re-ran our tuned tabular models under four protocols (Table III). Under the published protocol (P1, ${M.benchNPOne} encounters) our XGBoost reaches ${M.benchPOneXGB} and logistic regression ${M.benchPOneLR}, matching or exceeding the best published results (0.667 and 0.664–0.665). A patient-grouped split (P2; under P1 ${M.benchOverlapPOne} test patients also appear in training) changes AUROC only slightly (${M.benchPTwoXGB}). The cohort definition matters far more: first encounters only (P3) lower AUROC to ${M.benchPThreeXGB}–${M.benchPThreeLGBM}, and the 20k subsample (P4) costs a further 0.01–0.03. On P4, logistic regression gives ${M.benchPFourLR} against ${M.aurocGAT} for HealthGAT on the same splits. Cohort definition, not model choice, explains most of the spread in published AUROCs.`));
add(table("Comparison with published results (AUROC; ours: mean±sd, 3 seeds)", ["Source / protocol", "Model", "AUROC"], D.bench, [2650, 1300, 815], { size: 14 }));
add(H2("Error Analysis: Why the Graph Does Not Help Here, and When It Does"));
add(P(`**Homophily.** A readmitted patient's clinical-similarity neighbour is readmitted ${M.homSim} of the time (${M.homLiftSim}× the base rate), a diagnosis neighbour ${M.homDx} (${M.homLiftDx}×) and a same-specialty neighbour ${M.homSpec} (${M.homLiftSpec}×). The graph is barely more homophilous than random, and its signal is largely already in the node features, so message passing has little to add [13].`));
add(P(`**Ablation and attention.** Retraining without each relation (Table IV, Fig. 11) changes AUROC by less than the seed spread (all relations ${M.ablAll}; self-loops only ${M.ablNoEdges}). Removing only the clinical-similarity edges was slightly worse in all three seeds (${M.ablNoSim}), suggesting they mainly offset noise from the other relations. HealthGAT gives ${M.attSim}% of its attention to similarity neighbours, ${M.attDx}% to diagnosis and ${M.attSpec}% to specialty neighbours, with attention lift towards later-readmitted neighbours of only ${M.attLiftPos} (1.00 = uniform).`));
add(table("Relation ablation for HealthGAT on UCI (retrained per variant, 3 seeds)", ["Variant", "AUROC", "AUPRC"], D.abl, [2165, 1300, 1300]));
add(figure(FIG("ablation.png"), 3.2, "Relation ablation on UCI (test AUROC, mean±sd over seeds)."));
add(P(`**Controlled synthetic study.** We generated 6,000 synthetic patients with eight latent phenotypes and 120 providers whose hidden quality affects the outcome but leaves only noisy traces in individual records; same-provider edges carry ${M.synHomLiftSpec}× outcome enrichment. Here the graph models clearly beat every tabular model (Table V): HealthGAT reaches ${M.synGAT} against ${M.synMLP} for the identical network without message passing and ${M.synLGBM} for LightGBM (DeLong p < 0.01 in two of three seeds), closing about a third of the gap to the oracle (${M.synOracle}). Removing provider edges is the most costly ablation (${M.synablAll} → ${M.synablNoSpec}), about as costly as removing every edge (${M.synablNoEdges}). The architecture works; whether it helps depends on whether the graph carries outcome signal beyond the node features, which the homophily diagnostic checks before training.`));
add(table("Controlled synthetic study (6,000 patients, known ground-truth risk; mean±sd over 3 seeds)", ["Model", "AUROC", "AUPRC", "Brier", "ECE", "Top-10%"], D.syn,
          [1465, 900, 900, 500, 500, 500], { boldRow: (r) => r[0] === "HealthGAT", size: 14 }));
add(P(`**Subgroups and error profile.** Calibration holds in every subgroup (largest gap ${M.subMaxCalGap} percentage points), and discrimination is similar for women and men and for Caucasian and African-American patients (Fig. 12). Performance is weakest *within* patients who already have prior inpatient stays (AUROC ${M.subWorstAuc} for ${M.subWorst}). False negatives (${M.errFNN}) average ${M.errFNInp} prior inpatient stays against ${M.errTPInp} for true positives: the model mostly misses *first* readmissions without utilisation history. False positives (${M.errFPN}) are patients with prior stays (${M.errFPInp}) who were not readmitted. Both point to information the dataset lacks: social factors, post-discharge care and adherence.`));
wide(figure(FIG("subgroups.png"), 6.8, "HealthGAT test AUROC by subgroup (test sets pooled over seeds; n per bar; dotted line = chance)."));
add(figure(FIG("decision_curve.png"), 3.2, "Decision-curve analysis: net benefit against threshold probability."));

// =================== VI. CONCLUSION ===================
add(H1("Conclusion and Future Scope"));
add(P(`We built HealthGraph, an end-to-end system for 30-day readmission risk around a relation-aware GATv2 over a multi-relational patient graph, with leakage-safe feature engineering, automated Optuna tuning, MLflow tracking, a protocol-controlled benchmark and a validated, containerised cloud deployment. Under the published protocol our pipeline matches or exceeds the best reported AUROC on Diabetes-130 (${M.benchPOneXGB} against 0.667).`));
add(P(`On the real data HealthGAT is as accurate and well calibrated as strong tabular baselines but not significantly better (${M.aurocGAT} against ${M.aurocLR}). Its graph is only weakly outcome-homophilous, and the ablation, attention budgets and synthetic study agree: patient graphs help when they encode latent shared factors that individual records do not show (${M.synGAT} against ${M.synMLP} on synthetic data), and add nothing otherwise. Measure homophily before committing to a GNN, and report results under an explicit cohort definition.`));
add(P(`**Deployment considerations.** The model must be recalibrated and its reference graph refreshed for each new site and period: the data is from 1999–2008, uses ICD-9 and has no post-discharge information. Scoring cost grows with depth L; three-layer configurations led the search by only 0.004 validation AUPRC, so a 2-layer model is a reasonable latency trade-off.`));
add(P(`**Future scope.** (i) Richer, more homophilous relations such as true provider or ward identifiers, shared medications or temporal encounter chains in MIMIC-IV; (ii) heterophily-aware or label-propagation variants [13]; (iii) mini-batch neighbour sampling to train on the full 70k cohort; (iv) prospective validation with fairness auditing across subgroups.`));

// =================== REFERENCES ===================
add(new Paragraph({ alignment: AlignmentType.CENTER, spacing: { before: 200, after: 100 }, keepNext: true,
  children: [new TextRun({ text: "References", smallCaps: true, size: 20 })] }));
const REFS = [
  `B. Strack, J. P. DeShazo, C. Gennings, J. L. Olmo, S. Ventura, K. J. Cios, and J. N. Clore, "Impact of HbA1c measurement on hospital readmission rates: Analysis of 70,000 clinical database patient records," *BioMed Research International*, vol. 2014, Art. no. 781670, 2014.`,
  `O. G. Emi-Johnson and K. J. Nkrumah, "Predicting 30-day hospital readmission in patients with diabetes using machine learning on electronic health record data," *Cureus*, vol. 17, no. 4, Art. no. e82437, 2025.`,
  `S. S. Salim and A. A. Ibrahim, "A machine learning approach for predicting 30-day hospital readmission in patients with diabetes," *Healthcare*, vol. 14, no. 9, Art. no. 1185, 2026.`,
  `S. Tang, A. Tariq, J. A. Dunnmon, U. Sharma, P. Elugunti, D. L. Rubin, B. N. Patel, and I. Banerjee, "Predicting 30-day all-cause hospital readmission using multimodal spatiotemporal graph neural networks," *IEEE J. Biomed. Health Inform.*, vol. 27, no. 4, pp. 2071–2082, 2023.`,
  `C. van Walraven *et al.*, "Derivation and validation of an index to predict early death or unplanned readmission after discharge from hospital to the community," *CMAJ*, vol. 182, no. 6, pp. 551–557, 2010.`,
  `D. Kansagara *et al.*, "Risk prediction models for hospital readmission: A systematic review," *JAMA*, vol. 306, no. 15, pp. 1688–1698, 2011.`,
  `P. Veličković, G. Cucurull, A. Casanova, A. Romero, P. Liò, and Y. Bengio, "Graph attention networks," in *Proc. ICLR*, 2018.`,
  `S. Brody, U. Alon, and E. Yahav, "How attentive are graph attention networks?" in *Proc. ICLR*, 2022.`,
  `T. N. Kipf and M. Welling, "Semi-supervised classification with graph convolutional networks," in *Proc. ICLR*, 2017.`,
  `M. Schlichtkrull, T. N. Kipf, P. Bloem, R. van den Berg, I. Titov, and M. Welling, "Modeling relational data with graph convolutional networks," in *Proc. ESWC*, 2018, pp. 593–607.`,
  `E. Choi, M. T. Bahadori, L. Song, W. F. Stewart, and J. Sun, "GRAM: Graph-based attention model for healthcare representation learning," in *Proc. ACM SIGKDD*, 2017, pp. 787–795.`,
  `E. Choi *et al.*, "Learning the graphical structure of electronic health records with graph convolutional transformer," in *Proc. AAAI*, 2020, pp. 606–613.`,
  `J. Zhu, Y. Yan, L. Zhao, M. Heimann, L. Akoglu, and D. Koutra, "Beyond homophily in graph neural networks: Current limitations and effective designs," in *Proc. NeurIPS*, 2020.`,
  `T. Chen and C. Guestrin, "XGBoost: A scalable tree boosting system," in *Proc. ACM SIGKDD*, 2016, pp. 785–794.`,
  `G. Ke *et al.*, "LightGBM: A highly efficient gradient boosting decision tree," in *Proc. NeurIPS*, 2017, pp. 3146–3154.`,
  `Y. Rong, W. Huang, T. Xu, and J. Huang, "DropEdge: Towards deep graph convolutional networks on node classification," in *Proc. ICLR*, 2020.`,
  `C. Guo, G. Pleiss, Y. Sun, and K. Q. Weinberger, "On calibration of modern neural networks," in *Proc. ICML*, 2017, pp. 1321–1330.`,
  `J. Platt, "Probabilistic outputs for support vector machines and comparisons to regularized likelihood methods," in *Advances in Large Margin Classifiers*. MIT Press, 1999, pp. 61–74.`,
  `T. Akiba, S. Sano, T. Yanase, T. Ohta, and M. Koyama, "Optuna: A next-generation hyperparameter optimization framework," in *Proc. ACM SIGKDD*, 2019, pp. 2623–2631.`,
  `M. Zaharia *et al.*, "Accelerating the machine learning lifecycle with MLflow," *IEEE Data Eng. Bull.*, vol. 41, no. 4, pp. 39–45, 2018.`,
  `E. R. DeLong, D. M. DeLong, and D. L. Clarke-Pearson, "Comparing the areas under two or more correlated receiver operating characteristic curves: A nonparametric approach," *Biometrics*, vol. 44, no. 3, pp. 837–845, 1988.`,
  `X. Sun and W. Xu, "Fast implementation of DeLong's algorithm for comparing the areas under correlated receiver operating characteristic curves," *IEEE Signal Process. Lett.*, vol. 21, no. 11, pp. 1389–1393, 2014.`,
  `A. J. Vickers and E. B. Elkin, "Decision curve analysis: A novel method for evaluating prediction models," *Med. Decis. Making*, vol. 26, no. 6, pp. 565–574, 2006.`,
  `S. M. Lundberg and S.-I. Lee, "A unified approach to interpreting model predictions," in *Proc. NeurIPS*, 2017, pp. 4765–4774.`,
  `L. van der Maaten and G. Hinton, "Visualizing data using t-SNE," *J. Mach. Learn. Res.*, vol. 9, pp. 2579–2605, 2008.`,
  `L. McInnes, J. Healy, and J. Melville, "UMAP: Uniform manifold approximation and projection for dimension reduction," arXiv:1802.03426, 2018.`,
];
REFS.forEach((r, i) => add(new Paragraph({ alignment: AlignmentType.LEFT, spacing: { after: 30 },
  indent: { left: 400, hanging: 400 }, children: [new TextRun({ text: `[${i + 1}]\t`, size: 16 }), ...runs(r, { size: 16 })],
  tabStops: [{ type: "left", position: 400 }] })));
flush(2);

const doc = new Document({
  creator: "Naveenya V K", title: "HealthGraph: Patient Risk Prediction Using Relation-Aware Graph Attention Networks",
  styles: { default: { document: { run: { font: FONT, size: 20 } } },
            paragraphStyles: [
              { id: "Heading1", name: "Heading 1", basedOn: "Normal", next: "Normal", quickFormat: true,
                run: { font: FONT, size: 20 }, paragraph: { outlineLevel: 0 } },
              { id: "Heading2", name: "Heading 2", basedOn: "Normal", next: "Normal", quickFormat: true,
                run: { font: FONT, size: 20, italics: true }, paragraph: { outlineLevel: 1 } }] },
  sections,
});
const out = process.argv[2] || path.join(HERE, "HealthGraph_IEEE_Report.docx");
Packer.toBuffer(doc).then((buf) => { fs.writeFileSync(out, buf); console.log("wrote", out, sections.length, "sections,", figNo, "figures,", tabNo, "tables"); });
