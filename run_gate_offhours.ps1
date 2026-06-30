# ColdCallAI — Off-hours latency gate runner
# Scheduled by schtasks to run at 2am when BizFinder traffic is zero.
# Results written to tests/e2e/gate_offhours_YYYYMMDD_HHMM.txt

$timestamp = Get-Date -Format "yyyyMMdd_HHmm"
$outFile   = "D:\office\coldCall\tests\e2e\gate_offhours_$timestamp.txt"

Set-Location D:\office\coldCall

"=== ColdCallAI Off-Hours Gate Run ===" | Out-File $outFile
"Started: $(Get-Date)" | Out-File $outFile -Append
"" | Out-File $outFile -Append

# T1.9 latency — 20 turns
"--- T1.9 Latency (20 turns) ---" | Out-File $outFile -Append
& C:\Python311\python.exe tests/spike/latency_spike.py --mode llm --iterations 20 2>&1 |
    Out-File $outFile -Append

"" | Out-File $outFile -Append
"Finished: $(Get-Date)" | Out-File $outFile -Append
