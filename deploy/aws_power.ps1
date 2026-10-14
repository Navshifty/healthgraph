# Switch the HealthGraph AWS deployment on/off from Windows PowerShell.
#   .\deploy\aws_power.ps1 status
#   .\deploy\aws_power.ps1 stop
#   .\deploy\aws_power.ps1 start     # waits until the app answers; the Elastic IP keeps the URL fixed
param([ValidateSet("status", "start", "stop")][string]$Action = "status")
$ErrorActionPreference = "Stop"
$env:AWS_DEFAULT_REGION = "ap-south-1"

$iid = aws ec2 describe-instances --filters "Name=tag:Project,Values=healthgraph" `
    "Name=instance-state-name,Values=pending,running,stopping,stopped" `
    --query "Reservations[0].Instances[0].InstanceId" --output text
if (-not $iid -or $iid -eq "None") { Write-Host "No HealthGraph instance found."; exit 1 }

function Show-State {
    $info = aws ec2 describe-instances --instance-ids $iid `
        --query "Reservations[0].Instances[0].[State.Name,PublicIpAddress]" --output text
    $state, $ip = $info -split "\s+"
    Write-Host "instance ${iid}: $state"
    if ($ip -and $ip -ne "None") { Write-Host "UI:   http://$ip/ui"; Write-Host "Docs: http://$ip/docs" }
    return $ip
}

switch ($Action) {
    "status" { $null = Show-State }
    "stop" {
        aws ec2 stop-instances --instance-ids $iid | Out-Null
        Write-Host "stopping..."
        aws ec2 wait instance-stopped --instance-ids $iid
        $null = Show-State
    }
    "start" {
        aws ec2 start-instances --instance-ids $iid | Out-Null
        Write-Host "starting..."
        aws ec2 wait instance-running --instance-ids $iid
        $ip = Show-State
        Write-Host -NoNewline "waiting for the app"
        for ($i = 0; $i -lt 40; $i++) {
            try {
                Invoke-WebRequest -Uri "http://$ip/health" -TimeoutSec 5 -UseBasicParsing | Out-Null
                Write-Host " - ready."; exit 0
            } catch { Write-Host -NoNewline "."; Start-Sleep -Seconds 6 }
        }
        Write-Host " - not responding yet; try again in a minute."
    }
}
