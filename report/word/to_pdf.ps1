param([string]$Docx, [string]$Pdf)
# Render a .docx to PDF with Microsoft Word (also refreshes layout) and report the page count.
$word = New-Object -ComObject Word.Application
$word.Visible = $false
$word.DisplayAlerts = 0
try {
    $doc = $word.Documents.Open($Docx, $false, $true)
    $doc.Repaginate()
    $pages = $doc.ComputeStatistics(2)   # wdStatisticPages
    $doc.ExportAsFixedFormat($Pdf, 17)   # wdExportFormatPDF
    $doc.Close($false)
    "pages=$pages"
} finally {
    $word.Quit()
}
