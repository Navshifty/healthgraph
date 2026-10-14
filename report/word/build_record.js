// Lab-record style report (max 4 pages) for HealthGraph, single column.
const fs = require("fs");
const path = require("path");
const {
  Document, Packer, Paragraph, TextRun, ImageRun, Table, TableRow, TableCell, AlignmentType, WidthType,
  BorderStyle, HeadingLevel, ShadingType, VerticalAlign, LevelFormat,
} = require("docx");

const ROOT = "E:/AML/lab/healthgraph";
const HERE = __dirname;
const D = JSON.parse(fs.readFileSync(path.join(HERE, "report_data.json"), "utf8"));
const M = D.macros;
const FONT = "Times New Roman";
const PAGE = { size: { width: 11906, height: 16838 }, margin: { top: 1000, bottom: 1000, left: 1134, right: 1134 } };
const W = 11906 - 2 * 1134;   // 9638 DXA

function runs(text, base = {}) {
  const out = [], re = /(\*\*[^*]+\*\*|\*[^*]+\*)/g;
  let last = 0, m;
  while ((m = re.exec(text)) !== null) {
    if (m.index > last) out.push(new TextRun({ text: text.slice(last, m.index), ...base }));
    const t = m[0];
    out.push(t.startsWith("**") ? new TextRun({ text: t.slice(2, -2), bold: true, ...base })
                                : new TextRun({ text: t.slice(1, -1), italics: true, ...base }));
    last = m.index + t.length;
  }
  if (last < text.length) out.push(new TextRun({ text: text.slice(last), ...base }));
  return out;
}
const P = (t, o = {}) => new Paragraph({ children: runs(t), alignment: AlignmentType.JUSTIFIED, spacing: { after: o.after ?? 80 } });
const H = (t) => new Paragraph({ heading: HeadingLevel.HEADING_1, spacing: { before: 160, after: 60 }, keepNext: true,
  children: [new TextRun({ text: t, bold: true, size: 24, font: FONT, color: "000000" })] });
const BUL = (t) => new Paragraph({ numbering: { reference: "bul", level: 0 }, children: runs(t), alignment: AlignmentType.JUSTIFIED, spacing: { after: 30 } });
const NUM = (t) => new Paragraph({ numbering: { reference: "num", level: 0 }, children: runs(t), alignment: AlignmentType.JUSTIFIED, spacing: { after: 30 } });

function pngSize(f) { const b = fs.readFileSync(f); return { w: b.readUInt32BE(16), h: b.readUInt32BE(20) }; }
function img(f, wIn) {
  const { w, h } = pngSize(f), px = Math.round(wIn * 96);
  return new ImageRun({ type: "png", data: fs.readFileSync(f), transformation: { width: px, height: Math.round(px * h / w) },
                        altText: { title: path.basename(f), description: path.basename(f), name: path.basename(f) } });
}
let fig = 0;
const FIGURE = (f, wIn, cap) => { fig++; return [
  new Paragraph({ children: [img(f, wIn)], alignment: AlignmentType.CENTER, spacing: { before: 60, after: 30 }, keepNext: true }),
  new Paragraph({ children: [new TextRun({ text: `Fig. ${fig}: ${cap}`, italics: true, size: 18 })], alignment: AlignmentType.CENTER, spacing: { after: 100 } })]; };

const LINE = { style: BorderStyle.SINGLE, size: 4, color: "808080" };
const BORD = { top: LINE, bottom: LINE, left: LINE, right: LINE };
function TABLE(header, rows, widths, cap, boldRow) {
  const total = widths.reduce((a, b) => a + b, 0);
  const cell = (t, i, head, bold) => new TableCell({
    width: { size: widths[i], type: WidthType.DXA }, borders: BORD, verticalAlign: VerticalAlign.CENTER,
    margins: { top: 30, bottom: 30, left: 80, right: 80 },
    shading: head ? { fill: "D9E2F3", type: ShadingType.CLEAR, color: "auto" } : undefined,
    children: [new Paragraph({ alignment: i === 0 ? AlignmentType.LEFT : AlignmentType.CENTER, keepNext: true,
      children: [new TextRun({ text: t, size: 19, bold: head || bold })] })] });
  return [
    new Paragraph({ children: [new TextRun({ text: cap, bold: true, size: 19 })], alignment: AlignmentType.CENTER, spacing: { before: 80, after: 40 }, keepNext: true }),
    new Table({ width: { size: total, type: WidthType.DXA }, columnWidths: widths, alignment: AlignmentType.CENTER,
      rows: [new TableRow({ tableHeader: true, cantSplit: true, children: header.map((h, i) => cell(h, i, true, false)) }),
             ...rows.map((r) => new TableRow({ cantSplit: true, children: r.map((t, i) => cell(t, i, false, boldRow && boldRow(r))) }))] }),
    new Paragraph({ children: [], spacing: { after: 60 } }),
  ];
}

// ---------- header block ----------
const info = [["Name", "Naveenya V K"], ["Roll No.", "2403717674422030"],
              ["Course", "19MAM56 – Advanced Machine Learning Lab (III M.Sc. AI & ML)"],
              ["Assessment", "CAT II – Project Work"], ["Date", "14.10.2026"]];
const infoTable = new Table({ width: { size: W, type: WidthType.DXA }, columnWidths: [2000, W - 2000],
  rows: info.map(([k, v]) => new TableRow({ children: [
    new TableCell({ width: { size: 2000, type: WidthType.DXA }, borders: BORD, shading: { fill: "F2F2F2", type: ShadingType.CLEAR, color: "auto" },
                    margins: { top: 30, bottom: 30, left: 100, right: 100 }, children: [new Paragraph({ children: [new TextRun({ text: k, bold: true })] })] }),
    new TableCell({ width: { size: W - 2000, type: WidthType.DXA }, borders: BORD, margins: { top: 30, bottom: 30, left: 100, right: 100 },
                    children: [new Paragraph({ children: [new TextRun({ text: v })] })] })] })) });

const children = [
  new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 40 },
    children: [new TextRun({ text: "HealthGraph: Patient Risk Prediction Using Graph Attention Networks", bold: true, size: 30 })] }),
  new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 140 },
    children: [new TextRun({ text: "Coimbatore Institute of Technology · Department of Computing – AI & ML", italics: true, size: 20 })] }),
  infoTable,

  H("1. Aim"),
  P("To design, implement, tune and deploy an advanced machine-learning model that predicts whether a diabetic inpatient will be readmitted to hospital within 30 days of discharge. The model is a graph attention network over a patient-similarity graph; it is compared with state-of-the-art baselines and published benchmarks, and served as a containerised web application on the cloud."),

  H("2. Dataset"),
  BUL("**Source:** UCI Machine Learning Repository – *Diabetes 130-US Hospitals for years 1999–2008* (Strack et al., 2014)."),
  BUL("**Size:** 101,766 encounters of 71,518 patients, 50 attributes (demographics, admission details, lab/procedure counts, three ICD-9 diagnoses, 23 diabetes drugs)."),
  BUL(`**Target:** readmitted within 30 days (positive class ${M.prevalence}% after cohort filtering → highly imbalanced).`),
  BUL("**Cohort:** first encounter per patient, deaths/hospice discharges removed (69,973 patients); a stratified 20,000-patient sample is used for the graph models, split 70/15/15 (train/validation/test) over three random seeds."),

  H("3. Tools and Environment"),
  P("Python 3.14, PyTorch 2.14 (GAT written from scratch), scikit-learn 1.8, XGBoost, LightGBM, Optuna (hyperparameter search), MLflow (experiment tracking), SHAP, UMAP, FastAPI + Pydantic (API), Gradio (web UI), Docker, AWS EC2 + S3. Hardware: AMD Ryzen 7 7445HS CPU, 15 GB RAM (no GPU)."),

  H("4. Procedure"),
  NUM("**Load and filter** the data: keep each patient's first encounter, drop expired/hospice discharges, define the 30-day label."),
  NUM("**Split** into stratified train/validation/test sets (three seeds). Every statistic below is fitted on the training rows only, to avoid leakage."),
  NUM(`**Feature engineering:** log-transform and standardise counts; one-hot encode categoricals; encode the high-cardinality specialty (72 levels) and ICD-9 codes (700+) with frequency and out-of-fold smoothed target encoding; group ICD-9 codes into clinical chapters; keep "not measured" as its own category (${M.nFeatures} features).`),
  NUM(`**Dimensionality reduction:** PCA to 16 components (${M.pcaVar}% variance) gives a compact clinical-similarity space; t-SNE/UMAP are used for visual analysis.`),
  NUM("**Build the patient graph** with three edge types: k-nearest neighbours in PCA space, patients sharing diagnosis codes, and patients admitted under the same specialty."),
  NUM("**Train HealthGAT** (relation-aware GATv2) with class-weighted loss, dropout, DropEdge and early stopping on validation AUPRC; train baselines (Logistic Regression, Random Forest, XGBoost, LightGBM, MLP, GCN) on the same splits."),
  NUM("**Tune** all models with Optuna (TPE sampler + pruning; grid search for logistic regression) and log every trial to MLflow."),
  NUM("**Calibrate** probabilities with Platt scaling and choose the decision threshold on the validation set."),
  NUM("**Evaluate** on the test set: AUROC, AUPRC, F1, calibration (ECE), DeLong significance tests, bootstrap confidence intervals, subgroup and error analysis, relation ablation, SHAP; compare with published results under their protocol."),
  NUM("**Deploy:** wrap the model in a validated FastAPI service with a Gradio UI, containerise with Docker, and host on AWS EC2."),

  H("5. Model Architecture"),
  P("Each patient's features are encoded, then passed through stacked relation-aware GATv2 layers. For a patient i, every neighbour j receives an attention score that also depends on the type of link τ between them; the scores are normalised with a softmax and used to aggregate the neighbours' messages:"),
  new Paragraph({ children: [img(path.join(HERE, "eq_gat.png"), 3.4)], alignment: AlignmentType.CENTER, spacing: { after: 60 } }),
  P(`Residual connections, LayerNorm and a final MLP give the readmission logit. The tuned model uses ${M.bestlayers} layers × ${M.bestheads} heads × ${M.besthidden} hidden units (Optuna, ${M.nTrials} trials, best validation AUPRC ${M.bestValAuprc}). Core of the attention layer (healthgraph/models.py):`),
  ...[
    "xs, xd = self.lin_src(x).view(n, H, C), self.lin_dst(x).view(n, H, C)",
    "g = leaky_relu(xs[src] + xd[dst] + self.rel(edge_type).view(-1, H, C))",
    "alpha = edge_softmax((g * self.att).sum(-1), dst, n)        # attention per edge",
    "out = zeros_like(xs).index_add_(0, dst, xs[src] * alpha.unsqueeze(-1))",
  ].map((l) => new Paragraph({ spacing: { after: 0 }, indent: { left: 360 }, shading: { fill: "F2F2F2", type: ShadingType.CLEAR, color: "auto" },
                              children: [new TextRun({ text: l, font: "Consolas", size: 17 })] })),
  new Paragraph({ children: [], spacing: { after: 60 } }),

  H("6. Results"),
  ...TABLE(["Model", "AUROC", "AUPRC", "ECE", "F1"],
           D.main.map((r) => [r[0], r[1], r[2], r[5], r[6]]), [3000, 1800, 1800, 1100, 1100],
           "Table 1: Test-set performance on UCI Diabetes-130 (mean ± sd over 3 seeds)", (r) => r[0] === "HealthGAT"),
  ...TABLE(["Source / protocol", "Best model", "AUROC"],
           [["Emi-Johnson & Nkrumah (2025), published", "XGBoost", "0.667"],
            ["Salim & Ibrahim (2026), published", "Stacking ensemble", "0.665"],
            ["Ours – same protocol as the papers (P1)", "XGBoost", `${M.benchPOneXGB}`],
            ["Ours – strict first-encounter cohort (P3)", "LightGBM", `${M.benchPThreeLGBM}`]],
           [4700, 2600, 1500], "Table 2: Comparison with published benchmarks"),
  ...FIGURE(path.join(ROOT, "results/uci/figures/roc_pr_calibration.png"), 6.4, "ROC, precision–recall and calibration curves of all models (seed-0 test set)"),
  P("**Observations:**", { after: 30 }),
  BUL(`Under the published evaluation protocol our pipeline **exceeds the best published AUROC** (${M.benchPOneXGB} vs 0.667).`),
  BUL(`On the strict cohort all models perform similarly; HealthGAT (AUROC ${M.aurocGAT}) is on par with logistic regression (${M.aurocLR}) — DeLong tests show no significant difference. Every model is well calibrated (ECE ≈ ${M.eceGAT}).`),
  BUL(`The patient graph is only weakly informative here (a readmitted patient's neighbours are readmitted only ${M.homLiftSim}× as often as average), and removing all edges does not hurt (${M.ablNoEdges} vs ${M.ablAll}).`),
  BUL(`In a controlled synthetic study where the graph *does* carry hidden risk information, HealthGAT clearly wins (${M.synGAT} vs ${M.synMLP} without the graph): graph models help only when the links carry outcome signal.`),
  BUL(`Top risk factors (SHAP): prior inpatient stays, discharge destination, length of stay and the engineered diagnosis encodings. The model is weakest for first-time patients with no prior visits (most missed cases).`),

  H("7. Deployment"),
  BUL("**API:** FastAPI with endpoints POST /predict, POST /predict/batch, GET /model/info, GET /health; Pydantic validation rejects invalid inputs (HTTP 422)."),
  BUL("**Web UI:** Gradio form showing the calibrated risk, risk band, the patient's risk drivers and the most similar past patients."),
  BUL("**Container & cloud:** Docker image (non-root, health check) deployed on AWS EC2 free tier (t3.micro, Mumbai region); ≈0.4 s per prediction."),
  BUL("**Live link:** http://15.252.73.160/ui   ·   **Code:** https://github.com/Navshifty/healthgraph"),
  ...FIGURE(path.join(ROOT, "report/figures/ui_screenshot.png"), 4.6, "HealthGraph web UI on AWS: input form, calibrated risk, risk drivers and similar past patients"),

  H("8. Result"),
  P(`Thus, a graph-attention-based patient risk prediction system (HealthGraph) was successfully designed, tuned with Optuna, tracked with MLflow, benchmarked against published work and deployed as a Dockerised FastAPI/Gradio application on AWS. The pipeline outperformed published results under their protocol (AUROC ${M.benchPOneXGB}), and the study showed that graph attention improves prediction only when the patient graph carries outcome information.`),
];

const doc = new Document({
  creator: "Naveenya V K", title: "HealthGraph – Lab Record",
  styles: { default: { document: { run: { font: FONT, size: 22 } } },
            paragraphStyles: [{ id: "Heading1", name: "Heading 1", basedOn: "Normal", next: "Normal", quickFormat: true,
                                run: { font: FONT, size: 24, bold: true }, paragraph: { outlineLevel: 0 } }] },
  numbering: { config: [
    { reference: "bul", levels: [{ level: 0, format: LevelFormat.BULLET, text: "•", alignment: AlignmentType.LEFT,
                                   style: { paragraph: { indent: { left: 500, hanging: 260 } } } }] },
    { reference: "num", levels: [{ level: 0, format: LevelFormat.DECIMAL, text: "%1.", alignment: AlignmentType.LEFT,
                                   style: { paragraph: { indent: { left: 500, hanging: 320 } } } }] }] },
  sections: [{ properties: { page: PAGE }, children }],
});
const out = process.argv[2] || path.join(HERE, "HealthGraph_Lab_Record.docx");
Packer.toBuffer(doc).then((b) => { fs.writeFileSync(out, b); console.log("wrote", out); });
