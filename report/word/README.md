# Word (.docx) report generators

* `build_ieee.js`: IEEE two-column project report following the CAT II guideline structure.
* `build_record.js`: lab-record report (max 4 pages).
* `to_pdf.ps1`: renders a .docx to PDF through Microsoft Word and prints the page count.

```bash
python report/export_report_data.py report/word/report_data.json   # numbers + tables from results/
cd report/word && npm install docx@9.5.1
node build_ieee.js ../HealthGraph_IEEE_Report.docx
node build_record.js ../HealthGraph_Lab_Record.docx
```
`fig_arch.png` and `eq_gat.png` are cropped from the compiled LaTeX PDF (block diagram and GATv2 equations).
